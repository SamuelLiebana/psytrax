"""Extract and preprocess behavioural data for all mice.

Reads the empirical CSV, applies the standard cleaning pipeline for each mouse,
subtracts a non-decision time t_nd, and saves a psytrax-ready data dict to
psytrax/data/<mouse>_data.npy.

Usage:
    python extract_data.py
    python extract_data.py --csv /path/to/other.csv
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from itertools import groupby

_DEFAULT_CSV = os.path.join(
    os.path.dirname(__file__),
    '../learning_race_model/mouse_data/empirical_behav_normalised.csv',
)


def load_mouse(df, mouse_name):
    """Clean one mouse's data from the empirical DataFrame.

    Applies:
      - Drop repeat trials (repeatNumber > 1)
      - Drop first 5 trials of each session
      - Drop NoGo choices
      - Signed contrast = contrastRight − contrastLeft
      - Keep only sessions with a consistent contrast set
        (same unique contrasts as the first session)

    Returns a dict with keys: inputs, responses, times, session_lengths.
    Returns None if the mouse has no usable trials.
    """
    names = np.array([df['expRef'].iloc[i][-6:] for i in range(len(df))])
    idx = np.where(names == mouse_name)[0]
    if not len(idx):
        print(f'  WARNING: {mouse_name} not found in CSV — skipping')
        return None

    sub = df.iloc[idx].copy()

    # Drop repeat trials and first 5 of each session
    mask = (sub['repeatNumber'] > 1) | (sub['trialNumber'] <= 5)
    sub = sub[~mask]

    # Drop NoGo trials
    sub = sub[sub['choice'] != 'NoGo']
    if len(sub) == 0:
        print(f'  WARNING: {mouse_name} has no usable trials after cleaning — skipping')
        return None

    c = (sub['contrastRight'] - sub['contrastLeft']).to_numpy()
    T = (sub['choiceCompleteTime'] - sub['stimulusOnsetTime']).to_numpy()
    r = sub['choice'].map({'Right': 1.0, 'Left': 0.0}).to_numpy()

    sess = np.array([s[0:12] for s in sub['expRef'].to_numpy()])
    session_lengths = np.array([sum(1 for _ in g) for _, g in groupby(sess)])
    boundaries = np.cumsum(session_lengths, dtype=int)

    # Keep only sessions with consistent contrast set
    first_contrasts = np.unique(c[:boundaries[0]])
    max_sessions = 1
    for i in range(len(boundaries) - 1):
        if not np.array_equal(np.unique(c[boundaries[i]:boundaries[i + 1]]), first_contrasts):
            break
        max_sessions += 1

    end = boundaries[max_sessions - 1]
    c, T, r = c[:end], T[:end], r[:end]
    day_lengths = session_lengths[:max_sessions]

    return {
        'inputs': {'c': c},
        'responses': r,
        'times': T,
        'session_lengths': day_lengths,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', default=_DEFAULT_CSV,
                        help='Path to empirical_behav_normalised.csv')
    parser.add_argument('--out', default=os.path.join(os.path.dirname(__file__), 'data'),
                        help='Output directory (default: psytrax/data)')
    args = parser.parse_args()

    csv_path = os.path.abspath(args.csv)
    out_dir = os.path.abspath(args.out)

    if not os.path.exists(csv_path):
        sys.exit(f'ERROR: CSV not found at {csv_path}')

    os.makedirs(out_dir, exist_ok=True)

    print(f'Loading CSV: {csv_path}')
    df = pd.read_csv(csv_path)

    mice = sorted(set(df['expRef'].str[-6:].tolist()))
    print(f'Found {len(mice)} mice: {mice}\n')

    saved, skipped = [], []
    for mouse in mice:
        print(f'Processing {mouse}...')
        data = load_mouse(df, mouse)
        if data is None:
            skipped.append(mouse)
            continue

        N = len(data['responses'])
        n_sessions = len(data['session_lengths'])

        # Non-decision time: min RT minus 50 ms buffer
        T_all = data['times']
        finite_rts = T_all[np.isfinite(T_all)]

        out_path = os.path.join(out_dir, f'{mouse}_data.npy')
        data_adj = dict(data)
        if len(finite_rts) == 0:
            # No valid RTs — drop times entirely so psytrax treats as choice-only
            del data_adj['times']
            data_adj['t_nd'] = None
            print(f'  {N} trials, {n_sessions} sessions, no valid RTs (times dropped) → {out_path}')
        else:
            t_nd = float(np.min(finite_rts) - 0.05)
            T_adj = T_all - t_nd

            # Drop trials with non-finite RTs after subtraction
            rt_mask = np.isfinite(T_adj)
            n_dropped = int(np.sum(~rt_mask))
            if n_dropped > 0:
                data_adj['inputs'] = {k: v[rt_mask] for k, v in data['inputs'].items()}
                data_adj['responses'] = data['responses'][rt_mask]
                data_adj['times'] = T_adj[rt_mask]
                # Recompute session_lengths from scratch isn't straightforward, so warn
                del data_adj['session_lengths']
                print(f'  WARNING: dropped {n_dropped} trials with NaN RTs; '
                      f'session_lengths removed (cannot be recomputed reliably)')
            else:
                data_adj['times'] = T_adj

            data_adj['t_nd'] = t_nd
            N_final = len(data_adj['responses'])
            print(f'  {N_final} trials, {n_sessions} sessions, t_nd={t_nd:.3f}s → {out_path}')

        np.save(out_path, data_adj)
        saved.append(mouse)

    print(f'\nDone. Saved {len(saved)} mice, skipped {len(skipped)}.')
    if skipped:
        print(f'Skipped: {skipped}')


if __name__ == '__main__':
    main()

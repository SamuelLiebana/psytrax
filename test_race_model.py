"""End-to-end test of psytrax using the race model on a subset of one mouse."""

import sys
import numpy as np
import pandas as pd
from itertools import groupby

sys.path.insert(0, '/Users/sam/Documents/repos/psytrax')

import psytrax
from psytrax.models.race import log_lik_trial, N_PARAMS, PARAM_NAMES, default_hyper, default_E0


def load_mouse(csv_path, mouse_name):
    """Load and clean one mouse's data from the empirical CSV."""
    df = pd.read_csv(csv_path)
    names = np.array([df['expRef'][i][-6:] for i in range(len(df))])
    idx = np.where(names == mouse_name)[0]
    if not len(idx):
        raise ValueError(f'Mouse {mouse_name} not found')

    sub = df.iloc[idx].copy()

    # Drop repeats and first 5 trials of each session
    mask = (sub['repeatNumber'] > 1) | (sub['trialNumber'] <= 5)
    sub = sub[~mask]

    # Drop NoGo trials
    sub = sub[sub['choice'] != 'NoGo']

    # Signed contrast (+ve = rightward stimulus)
    c = (sub['contrastRight'] - sub['contrastLeft']).to_numpy()
    T = (sub['choiceCompleteTime'] - sub['stimulusOnsetTime']).to_numpy()
    r = sub['choice'].map({'Right': 1.0, 'Left': 0.0}).to_numpy()

    # Session lengths
    sess = np.array([s[0:12] for s in sub['expRef'].to_numpy()])
    session_lengths = np.array([sum(1 for _ in g) for _, g in groupby(sess)])
    boundaries = np.cumsum(session_lengths, dtype=int)

    # Keep only sessions with consistent contrast set
    first_contrasts = np.unique(c[:boundaries[0]])
    max_sessions = 1
    for i in range(len(boundaries) - 1):
        if not np.array_equal(np.unique(c[boundaries[i]:boundaries[i+1]]), first_contrasts):
            break
        max_sessions += 1

    end = boundaries[max_sessions - 1]
    c, T, r = c[:end], T[:end], r[:end]
    day_lengths = session_lengths[:max_sessions]

    print(f'Mouse {mouse_name}: {len(r)} total trials across {max_sessions} sessions')

    return {
        'inputs': {'c': c},
        'responses': r,
        'times': T,
        'session_lengths': day_lengths,
    }


if __name__ == '__main__':
    csv = '/Users/sam/Documents/repos/learning_race_model/mouse_data/empirical_behav_normalised.csv'
    mouse = 'DAP009'
    N_TEST = 300   # subset size for fast testing

    data = load_mouse(csv, mouse)
    N = N_TEST  # use first 300 trials

    print(f'\nFitting first {N} trials of {mouse} with psytrax race model...\n')

    result = psytrax.fit(
        data=data,
        log_lik_trial=log_lik_trial,
        n_params=N_PARAMS,
        param_names=PARAM_NAMES,
        hyper=default_hyper(),
        E0=default_E0(N),
        n_trials=N,
        session_boundaries=True,
        device='auto',
        subject_name=mouse,
        save=True,
    )

    print(f'\nSaved to: {result}')
    result = np.load(result, allow_pickle=True).item()

    print('\n--- Results ---')
    print(f"Log evidence : {result['log_evidence']:.4f}")
    print(f"Duration     : {result['duration']}")
    print(f"Params shape : {result['params'].shape}")
    print(f"Final sigma (log2): {np.log2(result['hyper']['sigma']).round(3)}")
    for i, name in enumerate(result['param_names']):
        p = result['params'][i]
        print(f"  {name:8s}: mean={p.mean():.3f}  std={p.std():.3f}")

    print(f'\nTo visualise: streamlit run app.py')
    print(f'Then upload:  fits/{mouse}_N{N}_fit.npy')

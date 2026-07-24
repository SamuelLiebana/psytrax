"""Batch-fit the race model to all preprocessed mice.

Reads every data/<mouse>_data.npy file, fits the race model with session
boundaries enabled, and saves results to example_fits/<mouse>_race_fit.npy.

Usage:
    python fit_all.py                        # fit all mice
    python fit_all.py --mice DAP009 DAP011   # fit specific mice
    python fit_all.py --skip-existing        # skip mice already fitted

Fitting one mouse takes 30–120 min depending on trial count.
Run overnight (or on a remote server) for all 26 mice.
"""

import os
import sys
import time
import argparse
import numpy as np

import psytrax
from psytrax.models import race

_REPO_DIR = os.path.dirname(__file__)
_DATA_DIR = os.path.join(_REPO_DIR, 'data')
_OUT_DIR  = os.path.join(_REPO_DIR, 'example_fits')


def fit_mouse(mouse, verbose=True, precision='float64', device='auto',
              init_sig_i=None):
    """Fit the race model to one mouse.

    ``sig_i`` (within-trial accumulator noise) is now estimated jointly with
    ``sigma`` by Empirical Bayes — pass ``init_sig_i`` only to override the
    starting point for the EB optimiser; otherwise the model's
    ``default_model_hyper()`` value is used.
    """
    data_path = os.path.join(_DATA_DIR, f'{mouse}_data.npy')
    out_path  = os.path.join(_OUT_DIR,  f'{mouse}_race_fit.npy')

    raw = np.load(data_path, allow_pickle=True).item()
    n_trials = len(raw['responses'] if 'responses' in raw else raw['r'])

    # session_lengths may have been dropped for mice with NaN RTs (e.g. DAP044)
    has_sessions = 'session_lengths' in raw or 'dayLength' in raw

    model_hyper = None if init_sig_i is None else {'sig_i': float(init_sig_i)}

    result = psytrax.fit(
        data               = raw,
        log_lik_trial      = race.log_lik_trial,
        n_params           = race.N_PARAMS,
        param_names        = race.PARAM_NAMES,
        hyper              = race.default_hyper(),
        E0                 = race.default_E0(n_trials),
        session_boundaries = has_sessions,
        hess_calc          = 'weights',
        device             = device,
        precision          = precision,
        verbose            = verbose,
        model_hyper        = model_hyper,
    )

    np.save(out_path, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mice', nargs='+', default=None,
                        help='Mice to fit (default: all in data/)')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip mice whose fit file already exists')
    parser.add_argument('--device', default='auto',
                        choices=['auto', 'cpu', 'gpu', 'tpu'],
                        help='Execution device policy (default: auto)')
    parser.add_argument('--precision', default='float64',
                        choices=['float32', 'float64'],
                        help='Requested JAX precision (default: float64)')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress per-iteration output')
    args = parser.parse_args()

    os.makedirs(_OUT_DIR, exist_ok=True)

    if args.mice:
        mice = args.mice
    else:
        # Sort ascending by trial count so smaller (faster) fits complete first
        all_mice = [f.replace('_data.npy', '') for f in os.listdir(_DATA_DIR)
                    if f.endswith('_data.npy')]
        def _n_trials(m):
            try:
                raw = np.load(os.path.join(_DATA_DIR, f'{m}_data.npy'), allow_pickle=True).item()
                r_key = 'responses' if 'responses' in raw else 'r'
                return len(raw[r_key])
            except Exception:
                return 0
        mice = sorted(all_mice, key=_n_trials)

    print(
        f'Fitting {len(mice)} mice (sorted by trial count, '
        f'device={args.device}, precision={args.precision}, '
        f'sig_i estimated by Empirical Bayes from initial value '
        f'{race.DEFAULT_SIG_I:.4f}): {mice}'
    )
    print()

    results_summary = []
    for i, mouse in enumerate(mice):
        out_path = os.path.join(_OUT_DIR, f'{mouse}_race_fit.npy')

        if args.skip_existing and os.path.exists(out_path):
            print(f'[{i+1}/{len(mice)}] {mouse} — skipping (fit already exists)')
            results_summary.append((mouse, None, 'skipped', None, None))
            continue

        data_path = os.path.join(_DATA_DIR, f'{mouse}_data.npy')
        if not os.path.exists(data_path):
            print(f'[{i+1}/{len(mice)}] {mouse} — WARNING: data file not found, skipping')
            results_summary.append((mouse, None, 'missing data', None, None))
            continue

        raw = np.load(data_path, allow_pickle=True).item()
        r_key = 'responses' if 'responses' in raw else 'r'
        N = len(raw[r_key])
        print(f'[{i+1}/{len(mice)}] {mouse} — {N} trials, starting fit...')
        t0 = time.time()

        try:
            result = fit_mouse(mouse, verbose=not args.quiet,
                               precision=args.precision,
                               device=args.device)
            elapsed = time.time() - t0
            log_evd = result['log_evidence']
            execution = (result.get('execution') or {}).get('description', 'unknown execution')
            print(
                f'  Done in {elapsed/60:.1f} min — log evidence: {log_evd:.2f} '
                f'[{execution}] → {out_path}'
            )
            results_summary.append((mouse, N, 'ok', log_evd, elapsed))
        except Exception as e:
            print(f'  ERROR: {e}')
            results_summary.append((mouse, N, f'error: {e}', None, None))

    # Final summary
    print('\n' + '='*60)
    print(f'{"Mouse":<12} {"N trials":>10} {"Status":<12} {"Log evd":>10} {"Time (min)":>12}')
    print('-'*60)
    for mouse, N_trials, status, log_evd, elapsed in results_summary:
        n_str    = str(N_trials) if N_trials is not None else '—'
        evd_str  = f'{log_evd:.1f}'    if log_evd  is not None else '—'
        time_str = f'{elapsed/60:.1f}' if elapsed  is not None else '—'
        print(f'{mouse:<12} {n_str:>10} {status:<12} {evd_str:>10} {time_str:>12}')
    print('='*60)

    n_ok = sum(1 for _, _, s, _, _ in results_summary if s == 'ok')
    print(f'\nCompleted {n_ok}/{len(mice)} fits successfully.')

    # Save timing stats to CSV
    import csv
    stats_path = os.path.join(_REPO_DIR, 'fit_all_stats.csv')
    with open(stats_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['mouse', 'n_trials', 'status', 'log_evidence', 'time_min', 'precision'])
        for mouse, N_trials, status, log_evd, elapsed in results_summary:
            writer.writerow([
                mouse,
                N_trials if N_trials is not None else '',
                status,
                f'{log_evd:.4f}' if log_evd is not None else '',
                f'{elapsed/60:.2f}' if elapsed is not None else '',
                args.precision,
            ])
    print(f'Timing stats saved to {stats_path}')


if __name__ == '__main__':
    main()

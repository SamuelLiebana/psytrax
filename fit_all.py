"""Batch-fit the race model to all preprocessed mice.

Reads every data/<mouse>_data.npy file, fits the race model with session
boundaries enabled, and saves results to example_fits/<mouse>_race_fit.npy.

With --push, each fit is committed and pushed to GitHub as soon as it
finishes, so the Streamlit app is updated incrementally.

Usage:
    python fit_all.py                        # fit all mice
    python fit_all.py --mice DAP009 DAP011   # fit specific mice
    python fit_all.py --skip-existing        # skip mice already fitted
    python fit_all.py --push                 # git-commit + push each fit

Fitting one mouse takes 30–120 min depending on trial count.
Run overnight (or on a remote server) for all 26 mice.
"""

import os
import sys
import time
import subprocess
import argparse
import numpy as np

import psytrax
from psytrax.models.race import (
    log_lik_trial, N_PARAMS, PARAM_NAMES, default_hyper,
)

_REPO_DIR = os.path.dirname(__file__)
_DATA_DIR = os.path.join(_REPO_DIR, 'data')
_OUT_DIR  = os.path.join(_REPO_DIR, 'example_fits')


def _git_push(mouse, out_path):
    """Stage the new fit file, commit, and push."""
    try:
        subprocess.run(['git', 'add', out_path], cwd=_REPO_DIR, check=True)
        msg = f'Add race model fit for {mouse}'
        subprocess.run(['git', 'commit', '-m', msg], cwd=_REPO_DIR, check=True)
        subprocess.run(['git', 'push', 'origin', 'main'], cwd=_REPO_DIR, check=True)
        print(f'  Pushed {mouse} fit to GitHub.')
    except subprocess.CalledProcessError as e:
        print(f'  WARNING: git push failed: {e}')


def fit_mouse(mouse, verbose=True):
    data_path = os.path.join(_DATA_DIR, f'{mouse}_data.npy')
    out_path  = os.path.join(_OUT_DIR,  f'{mouse}_race_fit.npy')

    raw = np.load(data_path, allow_pickle=True).item()

    # session_lengths may have been dropped for mice with NaN RTs (e.g. DAP044)
    has_sessions = 'session_lengths' in raw or 'dayLength' in raw

    result = psytrax.fit(
        data               = raw,
        log_lik_trial      = log_lik_trial,
        n_params           = N_PARAMS,
        param_names        = PARAM_NAMES,
        hyper              = default_hyper(),
        session_boundaries = has_sessions,
        hess_calc          = 'weights',
        verbose            = verbose,
    )

    np.save(out_path, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mice', nargs='+', default=None,
                        help='Mice to fit (default: all in data/)')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip mice whose fit file already exists')
    parser.add_argument('--push', action='store_true',
                        help='git commit + push each fit as it completes')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress per-iteration output')
    args = parser.parse_args()

    os.makedirs(_OUT_DIR, exist_ok=True)

    if args.mice:
        mice = args.mice
    else:
        mice = sorted(
            f.replace('_data.npy', '')
            for f in os.listdir(_DATA_DIR)
            if f.endswith('_data.npy')
        )

    print(f'Fitting {len(mice)} mice: {mice}')
    if args.push:
        print('Auto-push enabled: each fit will be pushed to GitHub on completion.')
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
            result = fit_mouse(mouse, verbose=not args.quiet)
            elapsed = time.time() - t0
            log_evd = result['log_evidence']
            print(f'  Done in {elapsed/60:.1f} min — log evidence: {log_evd:.2f} → {out_path}')
            results_summary.append((mouse, N, 'ok', log_evd, elapsed))
            if args.push:
                _git_push(mouse, out_path)
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
        writer.writerow(['mouse', 'n_trials', 'status', 'log_evidence', 'time_min'])
        for mouse, N_trials, status, log_evd, elapsed in results_summary:
            writer.writerow([
                mouse,
                N_trials if N_trials is not None else '',
                status,
                f'{log_evd:.4f}' if log_evd is not None else '',
                f'{elapsed/60:.2f}' if elapsed is not None else '',
            ])
    print(f'Timing stats saved to {stats_path}')


if __name__ == '__main__':
    main()

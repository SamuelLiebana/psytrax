"""Batch re-fit every per-mouse file in data/ with the joint
choice + RT + dopamine race model and save the result to
example_fits/<mouse>_race_fit.npy.

Run from the repo root:

    python examples/refit_all_with_dopamine.py

Optional flags:
    --mice DAP022 DAP039        only fit these mice
    --skip-existing             leave files in example_fits/ alone if
                                already present (default: overwrite)
    --hess-calc {None|weights|All}
                                Hessian level (default: weights — enough for
                                trajectory credible bands in the app)

Each fit takes a few minutes per mouse on a single CPU.  ~21 DLS mice ⇒
expect 30-90 minutes for the whole batch depending on your hardware.
The script prints per-mouse progress and a final summary.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR  = REPO_ROOT / 'data'
OUT_DIR   = REPO_ROOT / 'example_fits'
TIMING_CSV = REPO_ROOT / 'fit_all_stats.csv'

# Make the local psytrax package importable when the script is run
# directly from the repo (e.g. `python examples/refit_all_with_dopamine.py`)
# without needing PYTHONPATH=. or `pip install -e .`.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--mice', type=str, nargs='*', default=None,
                   help='Optional whitelist of mouse IDs (e.g. DAP022 DAP039).')
    p.add_argument('--skip-existing', action='store_true',
                   help='Skip mice that already have a fit on disk.')
    p.add_argument('--hess-calc', type=str, default='weights',
                   choices=['None', 'weights', 'Weights', 'All'],
                   help="Hessian level passed to psytrax.fit.")
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    if not DATA_DIR.is_dir():
        print(f'ERROR: {DATA_DIR} does not exist.  Run '
              '`python examples/extract_dopamine_to_data_files.py` first.',
              file=sys.stderr)
        return 1
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(DATA_DIR.glob('*_data.npy'))
    if not files:
        print(f'ERROR: no per-mouse files under {DATA_DIR}', file=sys.stderr)
        return 1

    if args.mice:
        wanted = set(args.mice)
        files = [f for f in files if f.name.replace('_data.npy', '') in wanted]
        if not files:
            print(f'No data files match --mice {args.mice}', file=sys.stderr)
            return 1

    # Lazy imports so --help is fast.
    import psytrax
    from psytrax.models import race

    hess_calc = None if args.hess_calc == 'None' else args.hess_calc
    if hess_calc == 'Weights':
        hess_calc = 'weights'

    summary = []
    t_total = time.time()
    for i, f in enumerate(files, 1):
        mouse = f.name.replace('_data.npy', '')
        out_path = OUT_DIR / f'{mouse}_race_fit.npy'
        if args.skip_existing and out_path.exists():
            print(f'[{i}/{len(files)}] {mouse}: skipping (already exists)')
            continue

        data = np.load(f, allow_pickle=True).item()
        N = len(data['responses'])
        n_da = int(np.sum(np.isfinite(np.asarray(data.get('dopamine', [])))))
        print(f'[{i}/{len(files)}] {mouse}: N = {N} trials, '
              f'dopamine valid = {n_da}/{N}')

        t0 = time.time()
        try:
            result = psytrax.fit(
                data           = data,
                log_lik_trial  = race.log_lik_trial,
                n_params       = race.N_PARAMS,
                param_names    = list(race.PARAM_NAMES),
                hyper          = race.default_hyper(),
                E0             = race.default_E0(N),
                model_hyper    = race.default_model_hyper_with_dopamine(),
                session_boundaries = True,
                hess_calc      = hess_calc,
                verbose        = False,
                save           = False,  # we save manually with our naming
                subject_name   = mouse,
            )
        except Exception as exc:
            elapsed = time.time() - t0
            print(f'    ✗ failed after {elapsed:.0f}s: {exc}')
            summary.append((mouse, N, n_da, None, elapsed, str(exc), {}))
            continue

        elapsed = time.time() - t0
        np.save(out_path, result, allow_pickle=True)
        log_evd = float(result.get('log_evidence', float('nan')))
        rec_mh = {k: round(float(v), 4)
                  for k, v in (result.get('model_hyper') or {}).items()}
        print(f'    ✓ {elapsed:.0f}s  log_evd={log_evd:.2f}  '
              f'model_hyper={rec_mh}  → {out_path.name}')
        summary.append((mouse, N, n_da, log_evd, elapsed, None, rec_mh))

    total = time.time() - t_total
    with open(TIMING_CSV, 'w', newline='') as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                'mouse', 'n_trials', 'dopamine_valid', 'status',
                'log_evidence', 'time_min', 'precision', 'hess_calc',
                'sig_i', 'sig_DA', 'da_beta', 'da_offset', 'error',
            ],
        )
        writer.writeheader()
        for mouse, N, n_da, log_evd, elapsed, err, rec_mh in summary:
            writer.writerow({
                'mouse': mouse,
                'n_trials': N,
                'dopamine_valid': n_da,
                'status': 'ok' if err is None else 'failed',
                'log_evidence': '' if log_evd is None else f'{log_evd:.4f}',
                'time_min': f'{elapsed / 60:.2f}',
                'precision': 'float64',
                'hess_calc': hess_calc if hess_calc is not None else 'None',
                'sig_i': rec_mh.get('sig_i', ''),
                'sig_DA': rec_mh.get('sig_DA', ''),
                'da_beta': rec_mh.get('da_beta', ''),
                'da_offset': rec_mh.get('da_offset', ''),
                'error': err or '',
            })

    ok    = sum(1 for s in summary if s[5] is None)
    fail  = sum(1 for s in summary if s[5] is not None)
    print(f'\n[done] {ok} succeeded, {fail} failed in {total:.0f}s '
          f'(out: {OUT_DIR})')
    print(f'Timing summary written to {TIMING_CSV}')
    if fail:
        print('Failures:')
        for mouse, _, _, _, _, err, _ in summary:
            if err is not None:
                print(f'  {mouse}: {err}')
    return 0 if fail == 0 else 2


if __name__ == '__main__':
    sys.exit(main())

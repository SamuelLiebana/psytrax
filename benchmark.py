"""Benchmark psytrax fitting time across models.

Supports two benchmark modes:

- synthetic: generate synthetic datasets at fixed trial counts
- real: fit a spread of bundled example mice with lower and higher trial counts

Usage:
    python benchmark.py                          # synthetic benchmark
    python benchmark.py --dataset real           # real example mice
    python benchmark.py --precision float32      # force float32
    python benchmark.py --n-repeats 3            # median over N runs per cell

To benchmark CPU explicitly when Metal is installed:
    JAX_PLATFORMS=cpu python benchmark.py
"""

import argparse
import copy
import importlib
import os
import time
import numpy as np

# Trial counts to sweep
N_TRIALS = [250, 500, 1000, 2000, 5000, 10_000]
REAL_MICE = ['DAP044', 'DAP027', 'DAP009', 'DAP011', 'DAP031', 'DAP007']
_REPO_DIR = os.path.dirname(__file__)
_DATA_DIR = os.path.join(_REPO_DIR, 'data')

# Models to benchmark: (label, module_path, needs_RT)
MODELS = [
    ('Logistic (K=2)',  'psytrax.models.logistic',   False),
    ('DDM approx (K=3)', 'psytrax.models.ddm_approx', True),
    ('Race fixed sig_i (K=5)', 'psytrax.models.race', True),
]


def make_data(N, rng, with_rt=True):
    """Synthetic data dict compatible with all built-in models."""
    contrasts = rng.choice([-0.5, -0.25, -0.125, 0.125, 0.25, 0.5], size=N)
    responses = (contrasts + rng.normal(0, 0.3, size=N) > 0).astype(int)
    dat = {
        'r':      responses,
        'inputs': {'c': contrasts},
    }
    if with_rt:
        dat['T'] = np.abs(rng.normal(0.5, 0.15, size=N)).clip(0.05)
    return dat


def fit_once_synthetic(mod, N, rng, precision='float64'):
    import psytrax
    m = importlib.import_module(mod[1])
    # Retry with different seeds if a rare degenerate posterior is encountered.
    for seed_offset in range(5):
        seed_rng = np.random.default_rng(rng.integers(1_000_000) + seed_offset)
        dat = make_data(N, seed_rng, with_rt=mod[2])
        try:
            t0 = time.perf_counter()
            fit_kwargs = _fit_kwargs_for_model(mod[0], m, len(dat['r']))
            psytrax.fit(data=dat, precision=precision, verbose=False, **fit_kwargs)
            return time.perf_counter() - t0
        except Exception:
            if seed_offset == 4:
                raise
            continue


def fit_once_real(mod, mouse, precision='float64'):
    import psytrax

    m = importlib.import_module(mod[1])
    dat = _load_mouse_data(mouse, needs_rt=mod[2])
    n_trials = len(dat.get('responses', dat.get('r')))
    fit_kwargs = _fit_kwargs_for_model(mod[0], m, n_trials)

    t0 = time.perf_counter()
    psytrax.fit(
        data=dat,
        session_boundaries='session_lengths' in dat or 'dayLength' in dat,
        hess_calc='weights',
        precision=precision,
        verbose=False,
        **fit_kwargs,
    )
    return time.perf_counter() - t0


def _load_mouse_data(mouse, needs_rt):
    path = os.path.join(_DATA_DIR, f'{mouse}_data.npy')
    raw = np.load(path, allow_pickle=True).item()
    dat = copy.deepcopy(raw)
    if not needs_rt:
        dat.pop('times', None)
        dat.pop('T', None)
    return dat


def _fit_kwargs_for_model(label, module, n_trials):
    if label.startswith('Race fixed sig_i'):
        log_lik_trial, n_params, param_names, default_hyper, _ = module.make_fixed_sig_i_model(
            module.DEFAULT_FIXED_SIG_I
        )
        return {
            'log_lik_trial': log_lik_trial,
            'n_params': n_params,
            'param_names': param_names,
            'hyper': default_hyper(),
            'E0': module.default_E0_fixed_sig_i(n_trials),
        }

    return {
        'log_lik_trial': module.log_lik_trial,
        'n_params': module.N_PARAMS,
        'param_names': module.PARAM_NAMES,
        'hyper': module.default_hyper(),
        'E0': module.default_E0(n_trials),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='synthetic',
                        choices=['synthetic', 'real'],
                        help='Benchmark synthetic trial counts or bundled example mice')
    parser.add_argument('--n-repeats', type=int, default=2,
                        help='Fits per cell (result = median)')
    parser.add_argument('--precision', default=None,
                        choices=['float32', 'float64'],
                        help='Override precision (default: float64 on CPU, float32 on Metal)')
    args = parser.parse_args()

    import jax
    device = jax.devices()[0]
    device_str = str(device)
    is_metal = 'METAL' in device_str.upper()
    if is_metal:
        print('\nApple Metal detected — Metal is float32-only and is not compatible')
        print('with psytrax (requires float64 for stable Hessian computation).')
        print('Run with JAX_PLATFORMS=cpu to benchmark the CPU backend.')
        print('For GPU acceleration use NVIDIA CUDA (jax[cuda12]).')
        raise SystemExit(1)
    precision = args.precision or 'float64'
    print(f'\nDevice: {device_str}')
    print(f'Precision: {precision}')
    print(f'Dataset mode: {args.dataset}')
    print(f'Repeats per cell: {args.n_repeats}\n')

    rng = np.random.default_rng(42)

    # Warm up JAX JIT (first fit is always slower due to compilation)
    print('Warming up JIT... ', end='', flush=True)
    if args.dataset == 'real':
        fit_once_real(MODELS[0], REAL_MICE[0], precision=precision)
    else:
        fit_once_synthetic(MODELS[0], 500, rng, precision=precision)
    print('done.\n')

    # Header
    if args.dataset == 'real':
        columns = REAL_MICE
        column_labels = [f'{mouse} ({_mouse_n_trials(mouse):,})' for mouse in REAL_MICE]
    else:
        columns = N_TRIALS
        column_labels = [f'{N:,}' for N in N_TRIALS]

    header = f"| {'Model':<22} | " + ' | '.join(f'{c:>14}' for c in column_labels) + ' |'
    sep    = f"| {'-'*22} | " + ' | '.join('-'*14 for _ in columns) + ' |'
    print(header)
    print(sep)

    results = {}   # (model_label, column) -> median_time

    for mod in MODELS:
        label = mod[0]
        row_times = []
        for column in columns:
            times = []
            for _ in range(args.n_repeats):
                if args.dataset == 'real':
                    t = fit_once_real(mod, column, precision=precision)
                else:
                    t = fit_once_synthetic(mod, column, rng, precision=precision)
                times.append(t)
            med = float(np.median(times))
            row_times.append(med)
            results[(label, column)] = med
            if args.dataset == 'real':
                print(f'\r  {label:<22}  {column:>8}  {med:.1f}s   ', end='', flush=True)
            else:
                print(f'\r  {label:<22}  N={column:>6,}  {med:.1f}s   ', end='', flush=True)

        cells = ' | '.join(f'{t:>6.1f}s' for t in row_times)
        print(f'\r| {label:<22} | {cells} |')

    # Also emit copy-pasteable Markdown
    print('\n\n--- Markdown table ---\n')
    if args.dataset == 'real':
        table_header = ' | '.join(f'**{mouse}** ({_mouse_n_trials(mouse):,})' for mouse in REAL_MICE)
    else:
        table_header = ' | '.join(f'**{N:,} trials**' for N in N_TRIALS)
    print(f'| Model | {table_header} |')
    print('|' + '---|' * (len(columns) + 1))
    for mod in MODELS:
        label = mod[0]
        cells = ' | '.join(f'{results[(label, column)]:.1f}s' for column in columns)
        print(f'| {label} | {cells} |')

    print(f'\n*Device: {device_str}, {precision}. JAX L-BFGS optimizer.*')


def _mouse_n_trials(mouse):
    raw = np.load(os.path.join(_DATA_DIR, f'{mouse}_data.npy'), allow_pickle=True).item()
    return len(raw.get('responses', raw.get('r')))


if __name__ == '__main__':
    main()

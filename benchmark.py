"""Benchmark psytrax fitting time across models and trial counts.

Generates synthetic data, fits each model at several trial counts, and
prints a Markdown table suitable for the README.

Usage:
    python benchmark.py
    python benchmark.py --n-repeats 3   # average over N runs per cell
"""

import argparse
import time
import numpy as np

# Trial counts to sweep
N_TRIALS = [250, 500, 1000, 2000, 5000, 10_000]

# Models to benchmark: (label, module_path, needs_RT)
MODELS = [
    ('Logistic (K=2)',  'psytrax.models.logistic',   False),
    ('DDM approx (K=3)', 'psytrax.models.ddm_approx', True),
    ('Race (K=6)',      'psytrax.models.race',        True),
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


def fit_once(mod, N, rng):
    import psytrax
    needs_rt = mod[2]
    dat = make_data(N, rng, with_rt=needs_rt)
    import importlib
    m = importlib.import_module(mod[1])
    t0 = time.perf_counter()
    psytrax.fit(
        data          = dat,
        log_lik_trial = m.log_lik_trial,
        n_params      = m.N_PARAMS,
        param_names   = m.PARAM_NAMES,
        hyper         = m.default_hyper(),
        hess_calc     = 'weights',
        verbose       = False,
    )
    return time.perf_counter() - t0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-repeats', type=int, default=2,
                        help='Fits per cell (result = median)')
    args = parser.parse_args()

    import jax
    device = jax.devices()[0]
    device_str = str(device)
    print(f'\nDevice: {device_str}')
    print(f'Repeats per cell: {args.n_repeats}\n')

    rng = np.random.default_rng(42)

    # Warm up JAX JIT (first fit is always slower due to compilation)
    print('Warming up JIT... ', end='', flush=True)
    fit_once(MODELS[0], 100, rng)
    print('done.\n')

    # Header
    n_cols = [f'{N:,}' for N in N_TRIALS]
    header = f"| {'Model':<22} | " + ' | '.join(f'{c:>8}' for c in n_cols) + ' |'
    sep    = f"| {'-'*22} | " + ' | '.join('-'*8 for _ in N_TRIALS) + ' |'
    print(header)
    print(sep)

    results = {}   # (model_label, N) -> median_time

    for mod in MODELS:
        label = mod[0]
        row_times = []
        for N in N_TRIALS:
            times = []
            for _ in range(args.n_repeats):
                t = fit_once(mod, N, rng)
                times.append(t)
            med = float(np.median(times))
            row_times.append(med)
            results[(label, N)] = med
            print(f'\r  {label:<22}  N={N:>6,}  {med:.1f}s   ', end='', flush=True)

        cells = ' | '.join(f'{t:>6.1f}s' for t in row_times)
        print(f'\r| {label:<22} | {cells} |')

    # Also emit copy-pasteable Markdown
    print('\n\n--- Markdown table ---\n')
    trial_header = ' | '.join(f'**{N:,} trials**' for N in N_TRIALS)
    print(f'| Model | {trial_header} |')
    print('|' + '---|' * (len(N_TRIALS) + 1))
    for mod in MODELS:
        label = mod[0]
        cells = ' | '.join(f'{results[(label, N)]:.1f}s' for N in N_TRIALS)
        print(f'| {label} | {cells} |')

    print(f'\n*Device: {device_str}. JAX L-BFGS optimizer, float64.*')


if __name__ == '__main__':
    main()

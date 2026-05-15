"""Compare learning-rule priors on DAP011 choice + RT + dopamine fits.

This script compares two Gaussian-walk mean choices for the race model:

1. REINFORCE, using the native psytrax score-function rule.
2. A tutor-executor rule adapted from ``da_long_term_learning`` by
   precomputing the sequential deep-linear-network update trajectory, then
   passing those updates to psytrax as fixed trial-aligned walk means.

Run from the repo root:

    python examples/compare_dap011_learning_rules.py

The tutor-executor adapter uses the simulation defaults from the companion
repo's model notebook: deep, diagonal, positive tutor-executor, squared-RPE
gradient descent with beta=9, w_0=0.05, k=1, init_stim_assoc=0.2, and
learning_rate=0.0026.  It uses the observed animal choice as the action.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# This script imports JAX directly while precomputing the tutor-executor
# trajectory.  Pin to CPU before that import so Apple Metal does not get
# initialized accidentally on machines where the plugin is installed but no
# supported GPU is visible.
os.environ.setdefault('JAX_PLATFORMS', 'cpu')

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / 'data' / 'DAP011_data.npy'
BASE_FIT_PATH = REPO_ROOT / 'example_fits' / 'DAP011_race_fit.npy'
OUT_DIR = REPO_ROOT / 'fits'

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--n-trials', type=int, default=None,
                        help='Use only the first N trials, for smoke tests.')
    parser.add_argument('--map-tol', type=float, default=1e-6,
                        help='MAP L-BFGS tolerance passed to psytrax.fit.')
    parser.add_argument('--device', type=str, default='cpu',
                        help="psytrax device setting, e.g. 'cpu' or 'auto'.")
    parser.add_argument('--precision', type=str, default='float64',
                        choices=['float32', 'float64'])
    parser.add_argument('--hess-calc', type=str, default='None',
                        choices=['None', 'weights', 'All'],
                        help='Credible-interval level. Evidence Hessian is always computed.')
    parser.add_argument('--skip-reinforce', action='store_true')
    parser.add_argument('--skip-tutor-executor', action='store_true')
    parser.add_argument('--no-base-warm-start', action='store_true',
                        help='Start from race defaults rather than example_fits/DAP011_race_fit.npy.')
    return parser.parse_args()


def _clone_dict(d: dict) -> dict:
    out = {}
    for key, val in d.items():
        out[key] = np.array(val, copy=True) if isinstance(val, np.ndarray) else val
    return out


def _load_data() -> dict:
    data = np.load(DATA_PATH, allow_pickle=True).item()
    inputs = dict(data['inputs'])
    c = np.asarray(inputs['c'], dtype=float)
    r = np.asarray(data['responses'], dtype=float)

    # DAP011 does not include explicit outcome.  For signed contrasts we infer
    # correctness from the stimulus side; for zero contrast we use expected
    # reward 0.5, matching the squared-RPE treatment in da_long_term_learning.
    reward = np.where(c > 0.0, r, np.where(c < 0.0, 1.0 - r, 0.5))
    inputs['reward'] = reward.astype(float)

    te_updates = _precompute_tutor_executor_updates(c, r)
    for i, name in enumerate(['te_wr', 'te_wl', 'te_br', 'te_bl', 'te_z']):
        inputs[name] = te_updates[:, i]

    out = dict(data)
    out['inputs'] = inputs
    return out


def _precompute_tutor_executor_updates(c: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Return fixed tutor-executor raw updates mapped to race params.

    Output columns are [wr, wl, br, bl, z].  The race threshold z has no direct
    counterpart in the tutor-executor network, so its update is set to zero.
    """
    from jax import config as jax_config
    jax_config.update('jax_enable_x64', True)
    import jax
    import jax.numpy as jnp

    beta = 9.0
    lr = 0.0026
    seed = 0
    n = len(c)

    def trial_arrays(c_t, r_t):
        if c_t < 0.0:
            x = np.array([1.0, 0.0, 1.0])
            y = np.array([1.0, 0.0])
        elif c_t > 0.0:
            x = np.array([0.0, 1.0, 1.0])
            y = np.array([0.0, 1.0])
        else:
            x = np.array([0.0, 0.0, 1.0])
            y = np.array([0.5, 0.5])
        action_left = 1.0 - float(r_t)  # psytrax r=1 is right; TE action=1 is left.
        return x, y, action_left

    xs = np.zeros((n, 3), dtype=np.float64)
    ys = np.zeros((n, 2), dtype=np.float64)
    actions = np.zeros(n, dtype=np.float64)
    for t in range(n):
        xs[t], ys[t], actions[t] = trial_arrays(c[t], r[t])

    def loss_for_input(w1, w2, x, y, action_left):
        yh = w2 @ (w1 @ x)
        left_target = jnp.where(y[0] > y[1], 1.0, jnp.where(y[0] < y[1], 0.0, y[0]))
        right_target = jnp.where(y[0] > y[1], 0.0, jnp.where(y[0] < y[1], 1.0, y[1]))
        pred = jnp.where(action_left == 1.0, yh[0], yh[1])
        target = jnp.where(action_left == 1.0, left_target, right_target)
        return 0.5 * (target - pred) ** 2

    def loss_total(w1, w2, x, y, action_left):
        return loss_for_input(w1, w2, x, y, action_left)

    def loss_stim(w1, w2, x, y, action_left):
        return loss_for_input(w1, w2, x * jnp.array([1.0, 1.0, 0.0]), y, action_left)

    def loss_bias(w1, w2, x, y, action_left):
        return loss_for_input(w1, w2, jnp.array([0.0, 0.0, 1.0]), y, action_left)

    grad_w1_total = jax.grad(loss_total, argnums=0)
    grad_w2_stim = jax.grad(loss_stim, argnums=1)
    grad_w2_bias = jax.grad(loss_bias, argnums=1)

    @jax.jit
    def step(w1, w2, x, y, action_left):
        gw1 = grad_w1_total(w1, w2, x, y, action_left)
        gw2 = grad_w2_stim(w1, w2, x, y, action_left) + grad_w2_bias(w1, w2, x, y, action_left)

        w1_new = w1 - lr * gw1
        w2_new = w2 - lr * gw2

        # Match the positive, diagonal deep-network simulations.
        w1_new = jnp.abs(w1_new)
        w2_new = jnp.abs(w2_new)
        w1_new = jnp.diag(jnp.diag(w1_new)) + 1e-6

        # Raw direction for psytrax.  Alpha in the EB fit learns the scale.
        raw_w2 = (w2_new - w2) / lr
        mapped = jnp.array([
            raw_w2[1, 1],  # wr: right output, right-stim pathway
            raw_w2[0, 0],  # wl: left output, left-stim pathway
            raw_w2[1, 2],  # br: right output bias
            raw_w2[0, 2],  # bl: left output bias
            0.0,           # z has no TE-network analogue
        ])
        return w1_new, w2_new, mapped

    key = jax.random.PRNGKey(seed)
    w1 = jnp.diag(jnp.array([0.05, 0.05, 1.0], dtype=jnp.float64))
    w1 = w1 + 1e-5 * jax.random.normal(key, (3, 3), dtype=jnp.float64)
    key, _ = jax.random.split(key)
    w2 = jnp.array(
        [[0.2, 0.2, 0.0],
         [0.2, 0.2, 0.0]],
        dtype=jnp.float64,
    )
    w2 = w2 + 1e-5 * jax.random.normal(key, (2, 3), dtype=jnp.float64)

    updates = np.zeros((n, 5), dtype=np.float64)
    for t in range(max(0, n - 1)):
        w1, w2, mapped = step(
            w1,
            w2,
            jnp.asarray(xs[t], dtype=jnp.float64),
            jnp.asarray(ys[t], dtype=jnp.float64),
            jnp.asarray(actions[t], dtype=jnp.float64),
        )
        updates[t] = np.asarray(mapped, dtype=np.float64)
    return updates


def _make_fixed_update_rule():
    import jax.numpy as jnp

    keys = ['te_wr', 'te_wl', 'te_br', 'te_bl', 'te_z']

    def learning_rule(params, dat_trial, model_hyper=None):
        del params, model_hyper
        return jnp.array([dat_trial['inputs'][key] for key in keys])

    learning_rule.required_data_keys = {
        key: {
            'description': 'Precomputed tutor-executor update component',
            'required': True,
        }
        for key in keys
    }
    return learning_rule


def _fit_learning_rule(name: str, data: dict, learning_rule, args: argparse.Namespace) -> dict:
    import psytrax
    from psytrax.models import race

    n_total = len(data['responses'])
    n = min(args.n_trials, n_total) if args.n_trials is not None else n_total
    use_base = BASE_FIT_PATH.exists() and not args.no_base_warm_start and args.n_trials is None

    if use_base:
        base = np.load(BASE_FIT_PATH, allow_pickle=True).item()
        # The no-learning example fit often recovers very tight process noise.
        # Reusing those sigmas with a newly introduced learning-rule mean can
        # make the first MAP step brittle, so keep the useful trajectory and
        # model-hyper warm start but reset sigma/sigDay to the looser defaults.
        hyper = race.default_hyper()
        model_hyper = _clone_dict(base['model_hyper'])
        e0 = np.array(base['params'], copy=True)
        print(f'  warm start: {BASE_FIT_PATH.relative_to(REPO_ROOT)}')
    else:
        hyper = race.default_hyper()
        model_hyper = race.default_model_hyper_with_dopamine()
        e0 = race.default_E0(n)
        print('  warm start: race defaults')

    if e0.shape[1] != n:
        e0 = e0[:, :n]

    hyper['alpha'] = np.full(race.N_PARAMS, 2 ** -3, dtype=float)
    hess_calc = None if args.hess_calc == 'None' else args.hess_calc

    def status(payload):
        stage = payload.get('stage', '?')
        msg = payload.get('message', '')
        print(f'    [{stage}] {msg}', flush=True)

    start = time.time()
    result = psytrax.fit(
        data=data,
        log_lik_trial=race.log_lik_trial,
        n_params=race.N_PARAMS,
        param_names=list(race.PARAM_NAMES),
        hyper=hyper,
        E0=e0,
        n_trials=args.n_trials,
        model_hyper=model_hyper,
        session_boundaries=True,
        learning_rule=learning_rule,
        hess_calc=hess_calc,
        device=args.device,
        precision=args.precision,
        map_tol=args.map_tol,
        verbose=False,
        save=False,
        status_callback=status,
    )
    result['comparison_learning_rule'] = name
    result['wall_time_seconds'] = time.time() - start
    return result


def _summarize_result(name: str, result: dict) -> dict:
    hyper = result.get('hyper') or {}
    model_hyper = result.get('model_hyper') or {}
    alpha = np.asarray(hyper.get('alpha', np.full(5, np.nan)), dtype=float)
    sigma = np.asarray(hyper.get('sigma', np.full(5, np.nan)), dtype=float)
    return {
        'learning_rule': name,
        'n_trials': int(result.get('n_trials', 0)),
        'log_evidence': float(result.get('log_evidence', np.nan)),
        'wall_time_min': float(result.get('wall_time_seconds', np.nan)) / 60.0,
        'duration': str(result.get('duration', '')),
        'sigma_wr': float(sigma[0]),
        'sigma_wl': float(sigma[1]),
        'sigma_br': float(sigma[2]),
        'sigma_bl': float(sigma[3]),
        'sigma_z': float(sigma[4]),
        'alpha_wr': float(alpha[0]),
        'alpha_wl': float(alpha[1]),
        'alpha_br': float(alpha[2]),
        'alpha_bl': float(alpha[3]),
        'alpha_z': float(alpha[4]),
        'sig_i': float(model_hyper.get('sig_i', np.nan)),
        'sig_DA': float(model_hyper.get('sig_DA', np.nan)),
        'da_beta': float(model_hyper.get('da_beta', np.nan)),
        'da_offset': float(model_hyper.get('da_offset', np.nan)),
    }


def main() -> int:
    args = _parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = _load_data()

    n_total = len(data['responses'])
    n = min(args.n_trials, n_total) if args.n_trials is not None else n_total
    print(f'DAP011: using {n}/{n_total} trials')
    print('Reward assumption: signed-contrast correctness, zero contrast = 0.5 expected reward')

    te_matrix = np.column_stack([
        data['inputs'][key] for key in ['te_wr', 'te_wl', 'te_br', 'te_bl', 'te_z']
    ])
    te_rms = np.sqrt(np.nanmean(te_matrix[:max(1, n - 1)] ** 2, axis=0))
    print('Tutor-executor raw update RMS [wr, wl, br, bl, z]:',
          np.array2string(te_rms, precision=4))

    results = {}
    if not args.skip_reinforce:
        from psytrax.models import race
        print('\nFitting REINFORCE prior mean')
        results['reinforce'] = _fit_learning_rule(
            'reinforce',
            data,
            race.default_learning_rule(reward_key='reward'),
            args,
        )
        out = OUT_DIR / f'DAP011_reinforce_learning_rule{"_N" + str(n) if args.n_trials else ""}_fit.npy'
        np.save(out, results['reinforce'], allow_pickle=True)
        print(f'  saved: {out.relative_to(REPO_ROOT)}')

    if not args.skip_tutor_executor:
        print('\nFitting tutor-executor prior mean')
        results['tutor_executor'] = _fit_learning_rule(
            'tutor_executor',
            data,
            _make_fixed_update_rule(),
            args,
        )
        out = OUT_DIR / f'DAP011_tutor_executor_learning_rule{"_N" + str(n) if args.n_trials else ""}_fit.npy'
        np.save(out, results['tutor_executor'], allow_pickle=True)
        print(f'  saved: {out.relative_to(REPO_ROOT)}')

    rows = [_summarize_result(name, result) for name, result in results.items()]
    csv_path = OUT_DIR / f'DAP011_learning_rule_comparison{"_N" + str(n) if args.n_trials else ""}.csv'
    json_path = OUT_DIR / f'DAP011_learning_rule_comparison{"_N" + str(n) if args.n_trials else ""}.json'

    if rows:
        with open(csv_path, 'w', newline='') as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        with open(json_path, 'w') as fh:
            json.dump(rows, fh, indent=2)

        print('\nSummary')
        for row in rows:
            print(
                f"  {row['learning_rule']:<15} "
                f"log_evidence={row['log_evidence']:.3f} "
                f"wall={row['wall_time_min']:.2f} min "
                f"alpha=[{row['alpha_wr']:.4g}, {row['alpha_wl']:.4g}, "
                f"{row['alpha_br']:.4g}, {row['alpha_bl']:.4g}, {row['alpha_z']:.4g}]"
            )
        print(f'  wrote: {csv_path.relative_to(REPO_ROOT)}')
        print(f'  wrote: {json_path.relative_to(REPO_ROOT)}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())

"""Behaviour-only learning-rule comparison on example animals.

This is a model-matched version of the learning-rule comparison:

1. A shallow side/zero GLM choice model.
2. The same GLM with a REINFORCE prior mean.
3. The same GLM with the tutor-executor deep-network update projected into
   the GLM's effective stimulus-weight/bias coordinates.
4. The same GLM with the full deep-network total-RPE gradient descent update
   projected into the GLM coordinates.

Run DAP011 from the repo root:

    python examples/compare_dap011_behaviour_learning_rules.py

Run every example animal:

    python examples/compare_dap011_behaviour_learning_rules.py --all-example-subjects --te-input-encoding side

Only choices are evaluated: RT and dopamine are deliberately dropped from the
data before fitting.
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

# Pin direct JAX imports to CPU before the Metal plugin can initialize.
os.environ.setdefault('JAX_PLATFORMS', 'cpu')

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / 'data'
DEFAULT_DATA_PATH = DATA_DIR / 'DAP011_data.npy'
OUT_DIR = REPO_ROOT / 'fits'

MODEL_ORDER = [
    'glm_side_zero_mean',
    'glm_side_reinforce',
    'glm_side_projected_te_rule',
    'glm_side_projected_total_rpe_gd',
]

MODEL_CHOICES = [
    'glm_shallow_zero_mean',
    'glm_shallow_reinforce',
    'glm_side_zero_mean',
    'glm_side_reinforce',
    'glm_side_projected_te_rule',
    'glm_side_projected_total_rpe_gd',
    'deep_zero_mean',
    'deep_te_rule',
    'deep_te_rule_glm_init_bias',
]

MODEL_DISPLAY = {
    'glm_shallow_zero_mean': 'GLM (shallow)',
    'glm_shallow_reinforce': 'GLM (shallow) + REINFORCE',
    'glm_side_zero_mean': 'GLM (shallow)',
    'glm_side_reinforce': 'GLM (shallow) + REINFORCE',
    'glm_side_projected_te_rule': 'GLM (shallow) + projected TE rule',
    'glm_side_projected_total_rpe_gd': 'GLM (shallow) + projected deep GD',
    'deep_zero_mean': 'Deep, 0-mean walk',
    'deep_te_rule': 'Deep + TE rule',
    'deep_te_rule_glm_init_bias': 'Deep + TE rule (GLM init bias)',
}

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--n-trials', type=int, default=None,
                        help='Use only the first N trials, for smoke tests.')
    parser.add_argument('--map-tol', type=float, default=1e-6)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--precision', type=str, default='float64',
                        choices=['float32', 'float64'])
    parser.add_argument('--hess-calc', type=str, default='None',
                        choices=['None', 'weights', 'All'])
    parser.add_argument('--data-path', type=Path, default=None,
                        help='Single *_data.npy file to fit. Defaults to DAP011.')
    parser.add_argument('--subject', type=str, default=None,
                        help='Single subject ID such as DAP011. Ignored with --data-path.')
    parser.add_argument('--subjects', nargs='+', default=None,
                        help='Subject IDs to fit, e.g. DAP011 DAP013.')
    parser.add_argument('--all-example-subjects', action='store_true',
                        help='Fit every data/*_data.npy example subject.')
    parser.add_argument('--models', nargs='+', default=MODEL_ORDER,
                        choices=MODEL_CHOICES,
                        help='Subset/order of models to fit.')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Load existing per-subject fit files instead of refitting.')
    parser.add_argument('--summary-tag', type=str, default='',
                        help='Optional suffix for summary CSV/JSON files only.')
    parser.add_argument('--show-status', action='store_true',
                        help='Print every psytrax status callback during fitting.')
    parser.add_argument('--te-input-encoding', type=str, default='magnitude',
                        choices=['magnitude', 'side'],
                        help=(
                            'magnitude: x=[max(-c,0), max(c,0), 1]; '
                            'side: x=[c<0, c>0, 1], matching the categorical '
                            'simulation task more literally.'
                        ))
    parser.add_argument('--skip-baselines', action='store_true',
                        help='Only fit logistic+REINFORCE and tutor-executor+rule.')
    parser.add_argument('--te-alpha-init', type=float, default=0.0026,
                        help=(
                            'Initial tutor-executor learning-rate alpha. Defaults '
                            'to the deep tutor-executor simulation value used in '
                            'da_long_term_learning.'
                        ))
    parser.add_argument('--te-init-bias-left', type=float, default=0.0,
                        help='Initial W2 left-output bias association for the TE model.')
    parser.add_argument('--te-init-bias-right', type=float, default=0.0,
                        help='Initial W2 right-output bias association for the TE model.')
    return parser.parse_args()


def _subject_from_path(path: Path) -> str:
    name = path.name
    return name[:-len('_data.npy')] if name.endswith('_data.npy') else path.stem


def _example_data_paths(args: argparse.Namespace) -> list[Path]:
    if args.all_example_subjects:
        return sorted(DATA_DIR.glob('*_data.npy'))
    if args.subjects:
        return [DATA_DIR / f'{subject}_data.npy' for subject in args.subjects]
    if args.data_path is not None:
        return [args.data_path]
    if args.subject is not None:
        return [DATA_DIR / f'{args.subject}_data.npy']
    return [DEFAULT_DATA_PATH]


def _load_behaviour_data(data_path: Path, te_input_encoding: str) -> dict:
    raw = np.load(data_path, allow_pickle=True).item()
    c = np.asarray(raw['inputs']['c'], dtype=float)
    r = np.asarray(raw['responses'], dtype=float)

    reward = np.where(c > 0.0, r, np.where(c < 0.0, 1.0 - r, 0.5))
    if te_input_encoding == 'magnitude':
        x_left = np.maximum(-c, 0.0)
        x_right = np.maximum(c, 0.0)
    elif te_input_encoding == 'side':
        x_left = (c < 0.0).astype(float)
        x_right = (c > 0.0).astype(float)
    else:
        raise ValueError(f'unknown TE input encoding: {te_input_encoding}')

    return {
        'inputs': {
            'c': c,
            'glm_side': np.sign(c).astype(float),
            'reward': reward.astype(float),
            'te_x_left': x_left.astype(float),
            'te_x_right': x_right.astype(float),
            'te_x_bias': np.ones_like(c, dtype=float),
        },
        'responses': r,
        'session_lengths': np.asarray(raw['session_lengths'], dtype=int),
    }


def make_tutor_executor_choice_model(beta: float = 9.0,
                                     alpha_init: float = 0.0026,
                                     init_bias_left: float = 0.0,
                                     init_bias_right: float = 0.0,
                                     init_bias_source: str = 'fixed'):
    """Return a deep diagonal tutor-executor choice model and learning rule.

    Parameters are:
        [w1_l, w1_r, w1_b,
         w2_ll, w2_lr, w2_lb,
         w2_rl, w2_rr, w2_rb]

    where W1 is diagonal and W2 maps hidden units to [left, right] output
    values.  The prior-mean rule is the tutor-executor separated-pathway
    gradient direction for instantaneous squared RPE.  A single fitted
    psytrax alpha therefore plays the role of the network learning rate.
    """
    import jax
    import jax.numpy as jnp

    K = 9
    param_names = [
        'w1_l', 'w1_r', 'w1_b',
        'w2_ll', 'w2_lr', 'w2_lb',
        'w2_rl', 'w2_rr', 'w2_rb',
    ]

    def unpack(params):
        w1 = jnp.diag(params[:3])
        w2 = jnp.reshape(params[3:], (2, 3))
        return w1, w2

    def trial_x(dat_trial, stim_only=False, bias_only=False):
        x = jnp.array([
            dat_trial['inputs']['te_x_left'],
            dat_trial['inputs']['te_x_right'],
            dat_trial['inputs']['te_x_bias'],
        ])
        if stim_only:
            x = x * jnp.array([1.0, 1.0, 0.0])
        if bias_only:
            x = jnp.array([0.0, 0.0, dat_trial['inputs']['te_x_bias']])
        return x

    def trial_target(dat_trial):
        x_left = dat_trial['inputs']['te_x_left']
        x_right = dat_trial['inputs']['te_x_right']
        left_target = jnp.where(x_left > x_right, 1.0,
                                jnp.where(x_right > x_left, 0.0, 0.5))
        right_target = jnp.where(x_right > x_left, 1.0,
                                 jnp.where(x_left > x_right, 0.0, 0.5))
        return jnp.array([left_target, right_target])

    def forward_values(params, dat_trial, stim_only=False, bias_only=False):
        w1, w2 = unpack(params)
        return w2 @ (w1 @ trial_x(dat_trial, stim_only=stim_only,
                                  bias_only=bias_only))

    def log_lik_trial(params, dat_trial, model_hyper=None):
        del model_hyper
        values = forward_values(params, dat_trial)
        logit_right = beta * (values[1] - values[0])
        log_p_right = jax.nn.log_sigmoid(logit_right)
        log_p_left = jax.nn.log_sigmoid(-logit_right)
        return dat_trial['r'] * log_p_right + (1.0 - dat_trial['r']) * log_p_left

    def chosen_squared_rpe(params, dat_trial, stim_only=False, bias_only=False):
        values = forward_values(params, dat_trial, stim_only=stim_only,
                                bias_only=bias_only)
        target = trial_target(dat_trial)
        choice_right = dat_trial['r']
        pred = jnp.where(choice_right == 1.0, values[1], values[0])
        chosen_target = jnp.where(choice_right == 1.0, target[1], target[0])
        return 0.5 * (chosen_target - pred) ** 2

    grad_w1_total = jax.grad(chosen_squared_rpe, argnums=0)
    grad_w2_stim = jax.grad(
        lambda params, dat_trial: chosen_squared_rpe(params, dat_trial, stim_only=True),
        argnums=0,
    )
    grad_w2_bias = jax.grad(
        lambda params, dat_trial: chosen_squared_rpe(params, dat_trial, bias_only=True),
        argnums=0,
    )

    def tutor_executor_rule(params, dat_trial, model_hyper=None):
        del model_hyper
        g_w1 = grad_w1_total(params, dat_trial)
        g_w2 = grad_w2_stim(params, dat_trial) + grad_w2_bias(params, dat_trial)

        grad = jnp.zeros_like(params)
        grad = grad.at[:3].set(g_w1[:3])
        grad = grad.at[3:].set(g_w2[3:])
        return -grad

    tutor_executor_rule.required_data_keys = {
        'te_x_left': {'description': 'Tutor-executor left stimulus input', 'required': True},
        'te_x_right': {'description': 'Tutor-executor right stimulus input', 'required': True},
        'te_x_bias': {'description': 'Tutor-executor bias input', 'required': True},
    }

    def default_hyper():
        # Use scalar sigma/sigDay/alpha: the original network has one learning
        # rate, and scalar process noise keeps this high-dimensional comparison
        # tractable.
        return {
            'sigma': 2 ** -1,
            'sigInit': 2 ** 2,
            'sigDay': 2 ** -2,
            'alpha': np.full(K, alpha_init, dtype=float),
        }

    def default_init_mean():
        # Match the deterministic centre of the da_long_term_learning deep
        # tutor-executor simulations: w_0=0.05, k=1, init_stim_assoc=0.2,
        # init_bias_assoc=0.
        return np.array([
            0.05, 0.05, 1.0,
            0.2, 0.2, init_bias_left,
            0.2, 0.2, init_bias_right,
        ], dtype=float)

    def default_E0(N):
        rng = np.random.default_rng(0)
        init = default_init_mean()
        init = init + 1e-5 * rng.normal(size=K)
        return np.tile(init[:, None], (1, N))

    return {
        'log_lik_trial': log_lik_trial,
        'learning_rule': tutor_executor_rule,
        'n_params': K,
        'param_names': param_names,
        'default_hyper': default_hyper,
        'default_E0': default_E0,
        'default_init_mean': default_init_mean,
        'init_description': (
            'da_long_term_learning deep TE centre: W1=diag([0.05,0.05,1]), '
            f'W2 stim assoc=0.2, W2 bias assoc=({init_bias_left:g}, '
            f'{init_bias_right:g}), beta=9'
        ),
        'te_init_bias_left': float(init_bias_left),
        'te_init_bias_right': float(init_bias_right),
        'te_init_bias_source': init_bias_source,
        'beta': float(beta),
    }


def make_projected_deep_glm_model(beta: float = 9.0,
                                  alpha_init: float = 2 ** -3,
                                  rule_kind: str = 'te') -> dict:
    """Return a side/zero GLM with projected deep-network learning dynamics.

    The fitted state is only the effective GLM parameter vector ``[w, b]`` for
    ``logit_right = w * sign(contrast) + b``.  To compute the TE update, we map
    ``[w, b]`` onto the diagonal tutor-executor manifold centred on the
    da_long_term_learning initialisation, apply either the TE squared-RPE rule
    or full deep-network total-RPE gradient descent, and project the resulting
    deep update back through the effective GLM coordinates.  This keeps the
    observation model matched to the shallow GLM while testing the learning-rule
    direction.
    """
    import jax
    import jax.numpy as jnp
    from psytrax.models import logistic

    log_lik_trial, _sample, K, param_names, _default_hyper, default_E0, _lr, _spec = (
        logistic.make_model(['glm_side'])
    )
    deep_model = make_tutor_executor_choice_model(beta=beta, alpha_init=alpha_init)
    deep_rule = deep_model['learning_rule']
    if rule_kind not in {'te', 'total_rpe_gd'}:
        raise ValueError(f'unknown projected deep rule_kind: {rule_kind}')

    def theta_to_deep(theta):
        w, b = theta
        w1_l = 0.05
        w1_r = 0.05
        w1_b = 1.0
        stim_delta = w / (beta * 0.05)
        bias_delta = b / (beta * w1_b)
        return jnp.array([
            w1_l, w1_r, w1_b,
            0.2 + 0.5 * stim_delta,
            0.2 - 0.5 * stim_delta,
            -0.5 * bias_delta,
            0.2 - 0.5 * stim_delta,
            0.2 + 0.5 * stim_delta,
            0.5 * bias_delta,
        ])

    def deep_to_theta(params):
        w1_l, w1_r, w1_b = params[:3]
        w2 = jnp.reshape(params[3:], (2, 3))
        left_logit = beta * (
            w1_l * (w2[1, 0] - w2[0, 0]) +
            w1_b * (w2[1, 2] - w2[0, 2])
        )
        right_logit = beta * (
            w1_r * (w2[1, 1] - w2[0, 1]) +
            w1_b * (w2[1, 2] - w2[0, 2])
        )
        zero_logit = beta * w1_b * (w2[1, 2] - w2[0, 2])
        w = 0.5 * (right_logit - left_logit)
        b = zero_logit
        return jnp.array([w, b])

    def deep_trial_x(dat_trial):
        return jnp.array([
            dat_trial['inputs']['te_x_left'],
            dat_trial['inputs']['te_x_right'],
            dat_trial['inputs']['te_x_bias'],
        ])

    def deep_trial_target(dat_trial):
        x_left = dat_trial['inputs']['te_x_left']
        x_right = dat_trial['inputs']['te_x_right']
        left_target = jnp.where(x_left > x_right, 1.0,
                                jnp.where(x_right > x_left, 0.0, 0.5))
        right_target = jnp.where(x_right > x_left, 1.0,
                                 jnp.where(x_left > x_right, 0.0, 0.5))
        return jnp.array([left_target, right_target])

    def total_rpe_loss(params, dat_trial):
        w1 = jnp.diag(params[:3])
        w2 = jnp.reshape(params[3:], (2, 3))
        values = w2 @ (w1 @ deep_trial_x(dat_trial))
        target = deep_trial_target(dat_trial)
        choice_right = dat_trial['r']
        pred = jnp.where(choice_right == 1.0, values[1], values[0])
        chosen_target = jnp.where(choice_right == 1.0, target[1], target[0])
        return 0.5 * (chosen_target - pred) ** 2

    jac_deep_to_theta = jax.jacfwd(deep_to_theta)
    total_rpe_grad = jax.grad(total_rpe_loss, argnums=0)

    def projected_deep_rule(theta, dat_trial, model_hyper=None):
        del model_hyper
        deep_params = theta_to_deep(theta)
        if rule_kind == 'te':
            deep_update = deep_rule(deep_params, dat_trial, {})
        else:
            deep_update = -total_rpe_grad(deep_params, dat_trial)
        return jac_deep_to_theta(deep_params) @ deep_update

    projected_deep_rule.required_data_keys = {
        'te_x_left': {'description': 'Tutor-executor left stimulus input', 'required': True},
        'te_x_right': {'description': 'Tutor-executor right stimulus input', 'required': True},
        'te_x_bias': {'description': 'Tutor-executor bias input', 'required': True},
    }
    rule_label = (
        'TE rule' if rule_kind == 'te'
        else 'deep total-RPE gradient descent'
    )

    return {
        'log_lik_trial': log_lik_trial,
        'learning_rule': projected_deep_rule,
        'n_params': K,
        'param_names': list(param_names),
        'default_hyper': lambda: {
            'sigma': 2 ** -1,
            'sigInit': np.full(K, 2 ** 4),
            'sigDay': 2 ** -2,
            'alpha': np.full(K, alpha_init, dtype=float),
        },
        'default_E0': default_E0,
        'init_description': (
            f'{rule_label} projected onto side/zero GLM coordinates centred '
            'on da_long_term_learning deep-network initialisation'
        ),
        'beta': float(beta),
    }


def make_projected_te_glm_model(beta: float = 9.0,
                                alpha_init: float = 2 ** -3) -> dict:
    return make_projected_deep_glm_model(
        beta=beta,
        alpha_init=alpha_init,
        rule_kind='te',
    )


def make_projected_total_rpe_gd_glm_model(beta: float = 9.0,
                                          alpha_init: float = 2 ** -3) -> dict:
    return make_projected_deep_glm_model(
        beta=beta,
        alpha_init=alpha_init,
        rule_kind='total_rpe_gd',
    )


def _fit(name: str, subject: str, data: dict, model: dict, learning_rule,
         args: argparse.Namespace) -> dict:
    import psytrax

    n_total = len(data['responses'])
    n = min(args.n_trials, n_total) if args.n_trials is not None else n_total
    hess_calc = None if args.hess_calc == 'None' else args.hess_calc

    hyper = model['default_hyper']()
    if learning_rule is None:
        hyper.pop('alpha', None)
    default_init_mean = model.get('default_init_mean')
    init_mean = default_init_mean() if callable(default_init_mean) else None

    def status(payload):
        stage = payload.get('stage', '?')
        msg = payload.get('message', '')
        print(f'    [{stage}] {msg}', flush=True)

    start = time.time()
    result = psytrax.fit(
        data=data,
        log_lik_trial=model['log_lik_trial'],
        n_params=model['n_params'],
        param_names=model['param_names'],
        hyper=hyper,
        E0=model['default_E0'](n),
        init_mean=init_mean,
        n_trials=args.n_trials,
        session_boundaries=True,
        learning_rule=learning_rule,
        hess_calc=hess_calc,
        device=args.device,
        precision=args.precision,
        map_tol=args.map_tol,
        verbose=False,
        save=False,
        status_callback=status if args.show_status else None,
    )
    result['subject'] = subject
    result['comparison_model'] = name
    result['comparison_label'] = MODEL_DISPLAY[name]
    result['wall_time_seconds'] = time.time() - start
    result['te_input_encoding'] = args.te_input_encoding
    result['te_alpha_init'] = args.te_alpha_init
    if init_mean is not None:
        result['init_mean'] = init_mean
    if 'init_description' in model:
        result['init_description'] = model['init_description']
    for key in ('te_init_bias_left', 'te_init_bias_right',
                'te_init_bias_source', 'beta'):
        if key in model:
            result[key] = model[key]
    return result


def _logistic_model_bundle():
    from psytrax.models import logistic

    return {
        'log_lik_trial': logistic.log_lik_trial,
        'learning_rule': logistic.default_learning_rule(reward_key='reward'),
        'n_params': logistic.N_PARAMS,
        'param_names': list(logistic.PARAM_NAMES),
        'default_hyper': lambda: {
            'sigma': 2 ** -1,
            'sigInit': 2 ** 4,
            'sigDay': 2 ** -2,
            'alpha': np.full(logistic.N_PARAMS, 2 ** -3, dtype=float),
        },
        'default_E0': logistic.default_E0,
    }


def _logistic_side_model_bundle():
    from psytrax.models import logistic

    log_lik_trial, _sample, K, param_names, _default_hyper, default_E0, default_lr, _spec = (
        logistic.make_model(['glm_side'])
    )
    return {
        'log_lik_trial': log_lik_trial,
        'learning_rule': default_lr(reward_key='reward'),
        'n_params': K,
        'param_names': list(param_names),
        'default_hyper': lambda: {
            'sigma': 2 ** -1,
            'sigInit': np.full(K, 2 ** 4),
            'sigDay': 2 ** -2,
            'alpha': np.full(K, 2 ** -3, dtype=float),
        },
        'default_E0': default_E0,
    }


def _summarize(subject: str, name: str, result: dict, fit_path: Path) -> dict:
    hyper = result.get('hyper') or {}
    return {
        'subject': subject,
        'model_id': name,
        'model': MODEL_DISPLAY[name],
        'fit_path': str(fit_path.relative_to(REPO_ROOT)),
        'te_input_encoding': str(result.get('te_input_encoding', '')),
        'n_params': int(result['params'].shape[0]),
        'n_trials': int(result.get('n_trials', 0)),
        'log_evidence': float(result.get('log_evidence', np.nan)),
        'wall_time_min': float(result.get('wall_time_seconds', np.nan)) / 60.0,
        'duration': str(result.get('duration', '')),
        'sigma': float(np.asarray(hyper.get('sigma', np.nan)).ravel()[0]),
        'sigDay': float(np.asarray(hyper.get('sigDay', np.nan)).ravel()[0]),
        'alpha': float(np.asarray(hyper.get('alpha', np.nan)).ravel()[0]),
        'te_alpha_init': float(result.get('te_alpha_init', np.nan)),
        'te_init_bias_left': float(result.get('te_init_bias_left', np.nan)),
        'te_init_bias_right': float(result.get('te_init_bias_right', np.nan)),
        'te_init_bias_source': str(result.get('te_init_bias_source', '')),
    }


def _make_glm_init_bias_te_model(glm_result: dict, args: argparse.Namespace) -> dict:
    """Initialise TE bias associations from the GLM's first-trial bias.

    The TE choice logit is beta * (V_right - V_left). With w1_bias initialized
    near one, a GLM bias b is matched by setting
        w2_rb - w2_lb = b / beta.
    We split that difference symmetrically across the two output bias weights.
    """
    params = np.asarray(glm_result['params'], dtype=float)
    glm_bias = float(params[-1, 0])
    beta = 9.0
    bias_delta = glm_bias / beta
    return make_tutor_executor_choice_model(
        beta=beta,
        alpha_init=args.te_alpha_init,
        init_bias_left=-0.5 * bias_delta,
        init_bias_right=0.5 * bias_delta,
        init_bias_source=f'glm_shallow_reinforce_first_bias={glm_bias:g}',
    )


def main() -> int:
    args = _parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    data_paths = _example_data_paths(args)
    missing = [path for path in data_paths if not path.exists()]
    if missing:
        missing_str = ', '.join(str(path.relative_to(REPO_ROOT)) for path in missing)
        raise FileNotFoundError(f'Missing example data file(s): {missing_str}')

    model_ids = list(args.models)
    if args.skip_baselines:
        model_ids = [
            model_id for model_id in model_ids
            if model_id in {
                'glm_shallow_reinforce',
                'glm_side_reinforce',
                'glm_side_projected_te_rule',
                'glm_side_projected_total_rpe_gd',
                'deep_te_rule',
            }
        ]

    print(f'Behaviour-only comparison: {len(data_paths)} subject(s)')
    print(f'Models: {", ".join(MODEL_DISPLAY[m] for m in model_ids)}')
    print(f'Tutor-executor input encoding: {args.te_input_encoding}')
    print(
        'Deep TE initialisation: W1=diag([0.05,0.05,1]), '
        f'W2 stim=0.2, W2 bias=({args.te_init_bias_left:g}, '
        f'{args.te_init_bias_right:g})'
    )
    print(f'TE alpha init: {args.te_alpha_init:g}')
    print('Reward assumption for REINFORCE: signed-contrast correctness, zero contrast = 0.5')

    logistic = _logistic_model_bundle()
    logistic_side = _logistic_side_model_bundle()
    projected_te_glm = make_projected_te_glm_model(
        beta=9.0,
        alpha_init=args.te_alpha_init,
    )
    projected_total_gd_glm = make_projected_total_rpe_gd_glm_model(
        beta=9.0,
        alpha_init=args.te_alpha_init,
    )
    tutor_executor = make_tutor_executor_choice_model(
        beta=9.0,
        alpha_init=args.te_alpha_init,
        init_bias_left=args.te_init_bias_left,
        init_bias_right=args.te_init_bias_right,
    )

    model_defs = {
        'glm_shallow_zero_mean': (logistic, None),
        'glm_shallow_reinforce': (logistic, logistic['learning_rule']),
        'glm_side_zero_mean': (logistic_side, None),
        'glm_side_reinforce': (logistic_side, logistic_side['learning_rule']),
        'glm_side_projected_te_rule': (
            projected_te_glm,
            projected_te_glm['learning_rule'],
        ),
        'glm_side_projected_total_rpe_gd': (
            projected_total_gd_glm,
            projected_total_gd_glm['learning_rule'],
        ),
        'deep_zero_mean': (tutor_executor, None),
        'deep_te_rule': (tutor_executor, tutor_executor['learning_rule']),
    }

    rows = []
    suffix = f'_N{args.n_trials}' if args.n_trials is not None else ''
    summary_tag = f'_{args.summary_tag}' if args.summary_tag else ''
    for data_path in data_paths:
        subject = _subject_from_path(data_path)
        data = _load_behaviour_data(data_path, args.te_input_encoding)
        n_total = len(data['responses'])
        n = min(args.n_trials, n_total) if args.n_trials is not None else n_total
        print(f'\n{subject}: using {n}/{n_total} trials')

        subject_rows = []
        subject_results = {}
        for name in model_ids:
            if name == 'deep_te_rule_glm_init_bias':
                glm_result = subject_results.get('glm_shallow_reinforce')
                if glm_result is None:
                    glm_suffix = f'_N{args.n_trials}' if args.n_trials is not None else ''
                    glm_path = OUT_DIR / (
                        f'{subject}_behaviour_glm_shallow_reinforce_'
                        f'{args.te_input_encoding}{glm_suffix}_fit.npy'
                    )
                    if not glm_path.exists():
                        raise FileNotFoundError(
                            f'{MODEL_DISPLAY[name]} requires a GLM + REINFORCE fit first; '
                            f'missing {glm_path.relative_to(REPO_ROOT)}'
                        )
                    glm_result = np.load(glm_path, allow_pickle=True).item()
                model = _make_glm_init_bias_te_model(glm_result, args)
                rule = model['learning_rule']
            else:
                model, rule = model_defs[name]
            fit_path = OUT_DIR / (
                f'{subject}_behaviour_{name}_{args.te_input_encoding}{suffix}_fit.npy'
            )
            if args.skip_existing and fit_path.exists():
                print(f'  loading {MODEL_DISPLAY[name]}: {fit_path.relative_to(REPO_ROOT)}')
                result = np.load(fit_path, allow_pickle=True).item()
            else:
                print(f'  fitting {MODEL_DISPLAY[name]}')
                result = _fit(name, subject, data, model, rule, args)
                np.save(fit_path, result, allow_pickle=True)
                print(f'    saved: {fit_path.relative_to(REPO_ROOT)}')
            subject_results[name] = result
            row = _summarize(subject, name, result, fit_path)
            subject_rows.append(row)
            rows.append(row)

        subject_csv = OUT_DIR / (
            f'{subject}_behaviour_learning_rule_comparison_'
            f'{args.te_input_encoding}{suffix}{summary_tag}.csv'
        )
        with open(subject_csv, 'w', newline='') as fh:
            writer = csv.DictWriter(fh, fieldnames=list(subject_rows[0].keys()))
            writer.writeheader()
            writer.writerows(subject_rows)
        print(f'  wrote: {subject_csv.relative_to(REPO_ROOT)}')

    prefix = 'example' if len(data_paths) > 1 else _subject_from_path(data_paths[0])
    csv_path = OUT_DIR / (
        f'{prefix}_behaviour_learning_rule_comparison_'
        f'{args.te_input_encoding}{suffix}{summary_tag}.csv'
    )
    json_path = OUT_DIR / (
        f'{prefix}_behaviour_learning_rule_comparison_'
        f'{args.te_input_encoding}{suffix}{summary_tag}.json'
    )

    with open(csv_path, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(json_path, 'w') as fh:
        json.dump(rows, fh, indent=2)

    print('\nSummary')
    for row in rows:
        print(
            f"  {row['subject']:<6} "
            f"{row['model']:<28} "
            f"K={row['n_params']:<2d} "
            f"log_evidence={row['log_evidence']:.3f} "
            f"wall={row['wall_time_min']:.2f} min "
            f"alpha={row['alpha']:.4g}"
        )
    print(f'  wrote: {csv_path.relative_to(REPO_ROOT)}')
    print(f'  wrote: {json_path.relative_to(REPO_ROOT)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

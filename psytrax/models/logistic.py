"""Logistic regression — choice-only baseline model.

A flexible logistic regression: ``P(right) = σ(w · x + b)``, where ``x`` is a
vector of trial-level features and ``w`` is a vector of trial-varying weights
(one per feature) evolving under the random-walk prior.  No RT modelled.

By default the model exposes a single feature ``c`` (signed contrast). Build
custom multi-feature variants with :func:`make_model`:

    >>> from psytrax.models.logistic import make_model
    >>> bundle = make_model(['c', 'prev_choice', 'prev_reward'])
    >>> log_lik_trial, sample_trial, K, names, *_ = bundle
    >>> K
    4

Parameters (K = n_inputs + 1)
-----------------------------
For a single-feature model:        ``[w, b]``
For multi-feature models:          ``[w_<feature1>, w_<feature2>, …, b]``

The single-feature naming is preserved so existing fits saved with K=2 and
``param_names = ['w', 'b']`` continue to load.
"""

import jax
import jax.numpy as jnp
import numpy as np


def make_model(input_keys=None):
    """Build a logistic regression model bundle.

    Args:
        input_keys: list of strings naming the features expected under
                    ``dat['inputs'][key]``. Defaults to ``['c']``.

    Returns:
        Tuple ``(log_lik_trial, sample_trial, N_PARAMS, PARAM_NAMES,
                 default_hyper, default_E0, default_learning_rule, DATA_SPEC)``.
        ``log_lik_trial`` is JAX-traceable, ``sample_trial`` is a numpy
        callable suitable for :func:`psytrax.simulate`.
    """
    if input_keys is None:
        input_keys = ['c']
    input_keys = list(input_keys)
    n_in = len(input_keys)
    if n_in == 0:
        raise ValueError("input_keys must contain at least one entry")
    K = n_in + 1

    # Preserve the legacy ['w', 'b'] naming when there's a single input so
    # already-saved fits keep loading; switch to disambiguated names when
    # multiple inputs are involved.
    if n_in == 1:
        param_names = ['w', 'b']
    else:
        param_names = [f'w_{k}' for k in input_keys] + ['b']

    def log_lik_trial(params, dat_trial, model_hyper=None):
        """Per-trial log-likelihood for the multi-input logistic.

        Args:
            params      : (K,) array — weights (n_in entries) followed by bias.
            dat_trial   : dict with scalar fields, including
                          ``dat_trial['inputs'][k]`` for each ``k`` in
                          ``input_keys``.
            model_hyper : unused.
        """
        w = params[:n_in]
        b = params[-1]
        x = jnp.stack([dat_trial['inputs'][k] for k in input_keys])
        logit = jnp.dot(w, x) + b
        log_p_right = jax.nn.log_sigmoid(logit)
        log_p_left  = jax.nn.log_sigmoid(-logit)
        return dat_trial['r'] * log_p_right + (1 - dat_trial['r']) * log_p_left

    def sample_trial(params, dat_trial, rng, model_hyper=None):
        """Sample one trial from the multi-input logistic."""
        w_arr = np.asarray([float(p) for p in params[:n_in]], dtype=float)
        b = float(params[-1])
        x = np.asarray([float(dat_trial['inputs'][k]) for k in input_keys],
                       dtype=float)
        p_right = 1.0 / (1.0 + np.exp(-(np.dot(w_arr, x) + b)))
        return {'r': float(rng.uniform() < p_right)}

    def default_hyper(n_params=K, shared_sigma=False):
        """Looser sigma so EB can escape the constant-trajectory local mode."""
        return {
            'sigma':   float(2 ** -1) if shared_sigma else np.full(n_params, 2 ** -1),
            'sigInit': np.full(n_params, 2 ** 4),
            'sigDay':  None,
        }

    def default_E0(N, n_params=K):
        return np.zeros((K, N))

    def default_learning_rule(reward_key='reward'):
        """Return a REINFORCE learning rule for this logistic model."""
        from psytrax.learning_rules import make_reinforce
        return make_reinforce(log_lik_trial, reward_key=reward_key)

    DATA_SPEC = {
        'inputs': {
            k: {
                'description': f'Input regressor "{k}"',
                'required': True,
            }
            for k in input_keys
        },
        'response': {
            'key': 'r',
            'description': 'Choice — discrete (0/1) or continuous in [0, 1]',
            'required': True,
        },
    }

    return (log_lik_trial, sample_trial, K, param_names,
            default_hyper, default_E0, default_learning_rule, DATA_SPEC)


# ---------------------------------------------------------------------------
# Default single-feature (contrast only) model — exported at module level so
# `from psytrax.models.logistic import log_lik_trial, ...` keeps working.
# ---------------------------------------------------------------------------
(log_lik_trial, sample_trial, N_PARAMS, PARAM_NAMES,
 default_hyper, default_E0, default_learning_rule, DATA_SPEC) = make_model(['c'])

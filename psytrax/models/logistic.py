"""Logistic regression — choice-only baseline model.

The simplest possible decision model: the probability of a rightward choice
is a sigmoid function of a weighted stimulus contrast plus a bias.  No RT
information is used.

Parameters (K=2)
----------------
w : contrast weight (positive = rightward bias for rightward stimuli)
b : bias (positive = rightward bias independent of contrast)
"""

import jax
import jax.numpy as jnp
import numpy as np

N_PARAMS = 2
PARAM_NAMES = ['w', 'b']

DATA_SPEC = {
    'inputs': {
        'c': {'description': 'Signed stimulus strength (e.g. contrast)', 'required': True},
    },
    'response': {
        'key': 'r',
        'description': 'Choice — discrete (0/1) or continuous in [0, 1]',
        'required': True,
    },
}


def log_lik_trial(params, dat_trial, model_hyper=None):
    """Per-trial log-likelihood for logistic regression.

    Args:
        params      : (2,) array [w, b]
        dat_trial   : dict with scalar fields
                      - inputs['c'] : signed contrast
                      - r           : response (1=right, 0=left)
        model_hyper : unused (logistic has no model-level hyperparameters).
    """
    w, b = params
    logit = w * dat_trial['inputs']['c'] + b
    log_p_right = jax.nn.log_sigmoid(logit)
    log_p_left  = jax.nn.log_sigmoid(-logit)
    return dat_trial['r'] * log_p_right + (1 - dat_trial['r']) * log_p_left


def default_hyper(n_params=N_PARAMS, shared_sigma=False):
    """Looser sigma so EB can escape the constant-trajectory local mode."""
    return {
        'sigma':   float(2 ** -1) if shared_sigma else np.full(n_params, 2 ** -1),
        'sigInit': np.full(n_params, 2 **  4),
        'sigDay':  None,
    }


def default_E0(N, n_params=N_PARAMS):
    return np.tile(np.array([0.5, 0.0])[:, None], N)


def sample_trial(params, dat_trial, rng, model_hyper=None):
    """Sample one trial from the logistic model.

    Args:
        params      : (2,) array [w, b]
        dat_trial   : dict with scalar field ``dat_trial['inputs']['c']``.
        rng         : numpy.random.Generator
        model_hyper : unused (logistic has no model-level hyperparameters).

    Returns:
        dict with key ``'r'`` (1 with probability sigmoid(w·c+b), else 0).
    """
    w, b = (float(p) for p in params)
    c = float(dat_trial['inputs']['c'])
    p_right = 1.0 / (1.0 + np.exp(-(w * c + b)))
    return {'r': float(rng.uniform() < p_right)}


def default_learning_rule(reward_key='reward'):
    """Return a REINFORCE learning rule for the logistic model.

    The update direction at trial t is the score function
    ∇_θ log p(y_t | x_t, θ) scaled by the reward signal.

    For this model, this reduces to the classic REINFORCE update:
        v̂_t = (y_t − p_right) · [c_t, 1] · reward_t

    The data dict must contain ``data['inputs']['reward']`` (or whichever
    key you pass as ``reward_key``), typically 1 for correct and 0 otherwise.

    Returns
    -------
    learning_rule : callable
        Suitable for ``psytrax.fit(..., learning_rule=...)``.
    """
    from psytrax.learning_rules import make_reinforce
    return make_reinforce(log_lik_trial, reward_key=reward_key)

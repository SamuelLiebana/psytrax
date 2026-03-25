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


def log_lik_trial(params, dat_trial):
    """Per-trial log-likelihood for logistic regression.

    Args:
        params    : (2,) array [w, b]
        dat_trial : dict with scalar fields
                    - inputs['c'] : signed contrast
                    - r           : response (1=right, 0=left)
    """
    w, b = params
    logit = w * dat_trial['inputs']['c'] + b
    log_p_right = jax.nn.log_sigmoid(logit)
    log_p_left  = jax.nn.log_sigmoid(-logit)
    return dat_trial['r'] * log_p_right + (1 - dat_trial['r']) * log_p_left


def default_hyper(n_params=N_PARAMS, shared_sigma=False):
    return {
        'sigma':   float(2 ** -3) if shared_sigma else np.full(n_params, 2 ** -3),
        'sigInit': np.full(n_params, 2 **  4),
        'sigDay':  None,
    }


def default_E0(N, n_params=N_PARAMS):
    return np.tile(np.array([0.5, 0.0])[:, None], N)

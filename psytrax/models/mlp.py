"""Multi-layer perceptron (MLP) decision model — choice only.

A small feed-forward network maps stimulus features to a choice probability.
Because the weights evolve under a Gaussian random-walk prior, the network
can capture nonlinear, non-stationary mappings from inputs to choices as the
animal learns.

Architecture: input (n_inputs) → hidden (H=4, tanh) → output (sigmoid)

Parameters (K = n_inputs*H + H + H + 1)
-----------------------------------------
Flattened as: [W1 (n_inputs × H), b1 (H), W2 (H), b2 (1)]

With default inputs=['c'] and H=4: K = 4 + 4 + 4 + 1 = 13

Extending inputs
----------------
Pass input_keys=['c', 'prev_r', 'prev_reward'] to include history features.
Your data dict must then contain those keys under 'inputs'.
"""

import jax
import jax.numpy as jnp
import numpy as np

_H = 4  # hidden units


def make_model(input_keys=None, hidden=_H):
    """Return (log_lik_trial, N_PARAMS, PARAM_NAMES, default_hyper, default_E0)
    for a given set of input features and hidden size.

    Args:
        input_keys : list[str], keys to read from dat_trial['inputs'].
                     Defaults to ['c'].
        hidden     : int, number of hidden units.
    """
    if input_keys is None:
        input_keys = ['c']
    n_in = len(input_keys)
    H    = hidden
    K    = n_in * H + H + H + 1

    param_names = (
        [f'W1_{i}_{j}' for i in range(n_in) for j in range(H)] +
        [f'b1_{j}'     for j in range(H)] +
        [f'W2_{j}'     for j in range(H)] +
        ['b2']
    )

    def log_lik_trial(params, dat_trial):
        # Unpack flat parameter vector
        W1 = params[:n_in * H].reshape(n_in, H)   # (n_in, H)
        b1 = params[n_in * H:n_in * H + H]        # (H,)
        W2 = params[n_in * H + H:n_in * H + 2*H]  # (H,)
        b2 = params[-1]                            # scalar

        # Build input vector from dat_trial
        x = jnp.stack([dat_trial['inputs'][k] for k in input_keys])  # (n_in,)

        # Forward pass
        h     = jnp.tanh(W1.T @ x + b1)           # (H,)
        logit = jnp.dot(W2, h) + b2                # scalar

        # Binary cross-entropy
        log_p_right = jax.nn.log_sigmoid(logit)
        log_p_left  = jax.nn.log_sigmoid(-logit)
        return dat_trial['r'] * log_p_right + (1 - dat_trial['r']) * log_p_left

    def default_hyper(n_params=K, shared_sigma=False):
        return {
            'sigma':   float(2 ** -3) if shared_sigma else np.full(n_params, 2 ** -3),
            'sigInit': np.full(n_params, 2 **  2),
            'sigDay':  None,
        }

    def default_E0(N, n_params=K):
        return np.zeros((n_params, N))

    return log_lik_trial, K, param_names, default_hyper, default_E0


# ---------------------------------------------------------------------------
# Default single-input (contrast only) model, ready to import directly
# ---------------------------------------------------------------------------
log_lik_trial, N_PARAMS, PARAM_NAMES, default_hyper, default_E0 = make_model(
    input_keys=['c'], hidden=_H
)

"""REINFORCE (policy-gradient) learning rules.

These rules implement the REINFORCE family of weight updates (Williams 1992),
generalised to work with *any* JAX-differentiable psytrax decision model.

The core idea is the policy-gradient identity:

    ∇_θ E[R]  =  E[ ∇_θ log π(a | s, θ) · R ]

so the per-trial update direction is simply the score function (gradient of the
model's log-likelihood with respect to its parameters) scaled by the reward.

Usage
-----
>>> from psytrax.learning_rules import make_reinforce
>>> from psytrax.models import logistic
>>>
>>> lr = make_reinforce(logistic.log_lik_trial)
>>> result = psytrax.fit(data, logistic.log_lik_trial, logistic.N_PARAMS,
...                      learning_rule=lr)

The data dict must include a trial-aligned ``reward`` array under
``data['inputs']['reward']`` (1 for correct / rewarded, 0 otherwise).

References
----------
Williams, R. J. (1992). Simple statistical gradient-following algorithms for
    connectionist reinforcement learning. Machine Learning, 8, 229–256.
Ashwood, Z., Roy, N. A., et al. (2020). Mice alternate between discrete
    strategies during perceptual decision-making. NeurIPS 2020.
"""

import jax
import jax.numpy as jnp


def make_reinforce(log_lik_trial, reward_key='reward'):
    """Create a general REINFORCE learning rule for any differentiable model.

    The update direction for trial *t* is:

        v̂_t  =  ∇_θ log p(y_t | x_t, θ_t) · r_t

    where r_t is the reward signal read from ``dat_trial['inputs'][reward_key]``.

    Parameters
    ----------
    log_lik_trial : callable
        The model's per-trial log-likelihood function with signature
        ``log_lik_trial(params, dat_trial) -> scalar``.
        Must be JAX-traceable.
    reward_key : str
        Key under ``dat_trial['inputs']`` that contains the scalar reward.

    Returns
    -------
    learning_rule : callable
        Function with signature ``(params, dat_trial) -> (K,)`` suitable
        for passing to ``psytrax.fit(..., learning_rule=...)``.
        The returned function has a ``required_data_keys`` attribute — a dict
        mapping each required ``data['inputs']`` key to a human-readable
        description, suitable for merging into a model's ``DATA_SPEC``.
    """
    grad_fn = jax.grad(log_lik_trial, argnums=0)

    def learning_rule(params, dat_trial):
        score = grad_fn(params, dat_trial)
        reward = dat_trial['inputs'][reward_key]
        return score * reward

    learning_rule.required_data_keys = {
        reward_key: {
            'description': 'Reward signal (1 = rewarded, 0 = unrewarded)',
            'required': True,
        },
    }

    return learning_rule


def make_reinforce_baseline(log_lik_trial, reward_key='reward',
                            baseline_key='baseline'):
    """REINFORCE with a per-trial baseline subtracted from the reward.

    The update direction is:

        v̂_t  =  ∇_θ log p(y_t | x_t, θ_t) · (r_t − b_t)

    where b_t is a baseline read from ``dat_trial['inputs'][baseline_key]``.
    This reduces the variance of the gradient estimate without introducing
    bias (Williams 1992, §4).

    If the baseline is not present in the data, consider using
    :func:`make_reinforce` instead and letting the Empirical Bayes
    optimisation absorb any constant baseline into the learning rates.

    Parameters
    ----------
    log_lik_trial : callable
        Per-trial log-likelihood (JAX-traceable).
    reward_key : str
        Key for the reward signal in ``dat_trial['inputs']``.
    baseline_key : str
        Key for the baseline signal in ``dat_trial['inputs']``.

    Returns
    -------
    learning_rule : callable
    """
    grad_fn = jax.grad(log_lik_trial, argnums=0)

    def learning_rule(params, dat_trial):
        score = grad_fn(params, dat_trial)
        advantage = dat_trial['inputs'][reward_key] - dat_trial['inputs'][baseline_key]
        return score * advantage

    learning_rule.required_data_keys = {
        reward_key: {
            'description': 'Reward signal (1 = rewarded, 0 = unrewarded)',
            'required': True,
        },
        baseline_key: {
            'description': 'Baseline signal subtracted from reward to reduce variance',
            'required': True,
        },
    }

    return learning_rule

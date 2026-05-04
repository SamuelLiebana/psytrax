"""Learning rules for psytrax decision models.

A learning rule is a JAX-traceable function with signature:

    learning_rule(params, dat_trial) -> (K,) array

where ``params`` is the (K,) parameter vector at trial *t* and ``dat_trial``
is the data dict for trial *t* (same format as the per-trial dict received by
``log_lik_trial``).  The returned array is the *unnormalized* update direction
v̂_t.  psytrax scales this by the per-parameter learning rate α_k:

    v_t = diag(α₁, …, α_K) · v̂_t

so that the Gaussian random-walk transition becomes:

    w_{t+1} − w_t  ∼  N(v_t, diag(σ₁², …, σ_K²))

The learning rates {α_k} are optimised as hyperparameters alongside the
volatilities {σ_k} in the Empirical Bayes outer loop.
"""

import numpy as np

from psytrax.learning_rules.reinforce import make_reinforce, make_reinforce_baseline


def simulate_with_learning_rule(
    sample_trial, learning_rule, params_0, inputs, *,
    alpha, sigma_walk, rng, model_hyper=None, reward_fn=None,
):
    """Forward-simulate parameter trajectories under a learning rule + Gaussian random walk.

    This is the inverse of :func:`psytrax.fit` with a learning rule: the rule
    here drives the *true* trajectory rather than serving as a fit prior.
    Per trial t (starting at t=0 with ``params_0``):

        y_t           = sample_trial(params_t, dat_trial)            # simulator output
        r_t           = reward_fn(c_t, y_t)                          # default below
        v̂_t           = learning_rule(params_t, dat_trial_with_reward)
        params_{t+1}  = params_t + α ⊙ v̂_t + 𝒩(0, σ_walk²)

    The default reward function rewards "correct" choices on signed contrasts:
    ``r_t = 1`` when ``sign(c_t) == sign(2·response − 1)`` (ignoring c=0,
    which gets a 50/50 random reward — easy to override).

    Args:
        sample_trial : per-trial sampler from a psytrax model.
        learning_rule: e.g. ``make_reinforce(log_lik_trial)``.
        params_0     : (K,) initial parameter vector.
        inputs       : dict of length-N arrays — must contain the model's
                       required input keys (e.g. 'c').
        alpha        : (K,) per-parameter learning rate.
        sigma_walk   : scalar or (K,) random-walk noise standard deviation.
        rng          : numpy.random.Generator.
        model_hyper  : optional model-level hyperparameters.
        reward_fn    : optional ``(c_t, response_dict) -> float``.  When None,
                       the default signed-contrast reward (above) is used.

    Returns:
        Tuple ``(params_traj, sim_data)`` — ``params_traj`` has shape (K, N)
        and ``sim_data`` is a psytrax-style data dict ready to feed into
        :func:`psytrax.fit`.
    """
    import jax.numpy as jnp

    K = len(params_0)
    inputs_arrays = {k: np.asarray(v) for k, v in inputs.items()}
    if not inputs_arrays:
        raise ValueError("inputs must be a non-empty dict")
    N = next(iter(inputs_arrays.values())).shape[0]
    alpha_arr = np.broadcast_to(np.asarray(alpha, dtype=float), (K,)).copy()
    sigma_arr = np.broadcast_to(np.asarray(sigma_walk, dtype=float), (K,)).copy()

    params_traj = np.zeros((K, N), dtype=float)
    params_traj[:, 0] = np.asarray(params_0, dtype=float)

    if reward_fn is None:
        def reward_fn(c, out):
            r = out.get('r', 0.0)
            if c > 0:
                return 1.0 if r >= 0.5 else 0.0
            if c < 0:
                return 1.0 if r < 0.5 else 0.0
            return 1.0 if rng.uniform() < 0.5 else 0.0

    responses = np.empty(N, dtype=float)
    times = None
    other_fields: dict[str, np.ndarray] = {}
    rewards = np.empty(N, dtype=float)

    c_arr = inputs_arrays.get('c')
    if c_arr is None:
        c_arr = np.zeros(N)

    for t in range(N):
        params_t = params_traj[:, t]
        dat_trial = {'inputs': {k: v[t] for k, v in inputs_arrays.items()}}

        out = sample_trial(jnp.asarray(params_t), dat_trial, rng,
                           model_hyper if model_hyper is not None else {})
        if not isinstance(out, dict) or 'r' not in out:
            raise ValueError("sample_trial must return a dict containing 'r'")

        r_t = float(reward_fn(float(c_arr[t]), out))
        rewards[t] = r_t

        # Build the dat_trial seen by the learning rule — mirrors what
        # psytrax.fit would assemble per trial (response + reward in inputs).
        lr_inputs = dict(dat_trial['inputs'])
        lr_inputs['reward'] = r_t
        lr_dat = {
            'inputs': lr_inputs,
            'r': float(out['r']),
        }
        if 'T' in out:
            lr_dat['T'] = float(out['T'])

        v_hat = np.asarray(
            learning_rule(jnp.asarray(params_t), lr_dat,
                          model_hyper if model_hyper is not None else {}),
            dtype=float,
        )

        if t + 1 < N:
            noise = rng.normal(0.0, sigma_arr, size=K)
            params_traj[:, t + 1] = params_t + alpha_arr * v_hat + noise

        responses[t] = float(out['r'])
        if 'T' in out:
            if times is None:
                times = np.empty(N, dtype=float)
            times[t] = float(out['T'])
        for key, value in out.items():
            if key in ('r', 'T'):
                continue
            other_fields.setdefault(key, np.empty(N, dtype=float))[t] = float(value)

    sim_data: dict = {
        'inputs': {**inputs_arrays, 'reward': rewards},
        'responses': responses,
    }
    if times is not None:
        sim_data['times'] = times
    sim_data.update(other_fields)
    return params_traj, sim_data


def get_required_data_keys(learning_rule):
    """Return the ``required_data_keys`` dict from a learning rule, or ``{}``
    if the rule does not expose one (e.g. a user-supplied custom function).

    Parameters
    ----------
    learning_rule : callable
        A learning rule function, typically produced by one of the factories
        in this module.

    Returns
    -------
    dict
        Mapping of ``data['inputs']`` key → ``{'description': ..., 'required': True}``.
    """
    return getattr(learning_rule, 'required_data_keys', {})


def augment_data_spec(data_spec, learning_rule):
    """Return a copy of *data_spec* extended with the learning rule's required
    input keys.

    This is useful for driving interactive column-mapping UIs (e.g. Streamlit)
    that need to know *all* required inputs — both those from the model and
    those from the learning rule.

    Parameters
    ----------
    data_spec : dict
        A model's ``DATA_SPEC`` dictionary.
    learning_rule : callable
        A learning rule function with a ``required_data_keys`` attribute
        (as produced by :func:`make_reinforce` or :func:`make_reinforce_baseline`).

    Returns
    -------
    dict
        A shallow copy of *data_spec* whose ``'inputs'`` dict is extended with
        any keys declared by the learning rule that are not already present.
    """
    lr_keys = get_required_data_keys(learning_rule)
    if not lr_keys:
        return data_spec

    merged = dict(data_spec)
    merged['inputs'] = dict(data_spec.get('inputs', {}))
    for key, info in lr_keys.items():
        if key not in merged['inputs']:
            merged['inputs'][key] = info
    return merged

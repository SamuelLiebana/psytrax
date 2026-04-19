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

from psytrax.learning_rules.reinforce import make_reinforce, make_reinforce_baseline


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

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

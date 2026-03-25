"""
Utilities for wrapping a per-trial log-likelihood function into the batched
forms that psytrax's MAP estimation requires.

Writing a JAX-compatible likelihood
------------------------------------
The simplest path is to write your per-trial function using `jax.numpy` (jnp)
instead of `numpy` (np).  Most numpy operations have a jnp equivalent:

    import jax.numpy as jnp
    from jax.scipy.stats.norm import logcdf  # scipy.stats functions exist in jax too

A few common gotchas when porting numpy → jax:
  - No in-place mutation (x[i] = v). Use jnp.where or functional updates instead.
  - Python control flow (if/for) on traced values must use jax.lax primitives
    (lax.cond, lax.scan, lax.fori_loop) or be written so branches depend only
    on static (non-array) values.
  - Integer dtypes: jax defaults to int32; use jnp.int32 / jnp.float64 explicitly
    and enable x64 with jax.config.update("jax_enable_x64", True) at import time.
  - jax.scipy covers a large subset of scipy.special and scipy.stats.

If your model is truly non-differentiable (e.g. a simulator), finite-difference
gradients are possible but will be slow.  Consider whether you can replace
the simulator with a JAX-compatible approximation (normalizing flows, etc.).
"""

import jax
import jax.numpy as jnp
from jax import hessian, jit, value_and_grad, vmap

jax.config.update("jax_enable_x64", True)


def _make_vmap_axes(x, N):
    """Recursively build JAX vmap in_axes for a pytree.

    Arrays whose first dimension equals N get axis 0 (mapped over trials).
    Everything else — scalars, None, arrays of other shapes — gets None
    (broadcast as a constant to every trial).
    """
    if x is None:
        return None
    if isinstance(x, dict):
        return {k: _make_vmap_axes(v, N) for k, v in x.items()}
    arr = jnp.asarray(x) if not hasattr(x, 'shape') else x
    if arr.ndim > 0 and arr.shape[0] == N:
        return 0
    return None


def make_likelihood_fns(log_lik_trial):
    """Wrap a per-trial log-likelihood into batched forms for MAP estimation.

    Args:
        log_lik_trial: callable with signature
            log_lik_trial(params_k, dat_trial) -> scalar

            params_k  : jnp array of shape (K,) — parameters for one trial
            dat_trial : dict — same keys as dat, but each array-valued field
                        contains a single trial's worth of data (scalar or
                        reduced leading dimension).  Fields that are not
                        trial-indexed (e.g. dayLength) are passed through
                        unchanged.

            The function must be JAX-traceable (written with jax.numpy).

    Returns:
        log_likelihood_fn : callable(E, dat) -> scalar
            E   : (K, N) parameter matrix
            dat : dict with arrays of shape (N, ...)
        likelihood_terms_fn : callable(E, dat) -> tuple
            Returns (logli, dlogli, hessian_blocks) where:
              - logli is a scalar
              - dlogli has shape (K, N)
              - hessian_blocks has shape (N, K, K)
    """

    trial_value_grad_fn = value_and_grad(log_lik_trial, argnums=0)
    trial_hessian_fn = hessian(log_lik_trial, argnums=0)

    @jit
    def log_likelihood_fn(E, dat):
        N = E.shape[1]
        in_axes = (1, _make_vmap_axes(dat, N))
        return jnp.sum(vmap(log_lik_trial, in_axes)(E, dat))

    @jit
    def likelihood_terms_fn(E, dat):
        N = E.shape[1]
        in_axes = (1, _make_vmap_axes(dat, N))
        values, grads = vmap(trial_value_grad_fn, in_axes)(E, dat)
        hessians = vmap(trial_hessian_fn, in_axes)(E, dat)
        return jnp.sum(values), jnp.swapaxes(grads, 0, 1), hessians

    return log_likelihood_fn, likelihood_terms_fn

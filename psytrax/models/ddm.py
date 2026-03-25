"""Drift Diffusion Model (DDM) with reaction times.

A single accumulator drifts toward ±z (right/left boundary) with drift rate
  d = w · contrast + b
and diffusion variance σ².  The first-passage-time density to each boundary
is an inverse Gaussian (Wald distribution).  Compared to the race model, the
DDM constrains the two response options to share the same weight, bias, and
boundary — reflecting the classic assumption that a single evidence variable
drives both speed and accuracy.

Parameters (K=4)
----------------
w  : contrast weight (drift rate per unit contrast)
b  : drift bias (baseline rightward drift)
z  : decision boundary (threshold)
σ  : diffusion noise standard deviation
"""

import jax
import jax.numpy as jnp
from jax import jit, lax
from jax.scipy.stats.norm import logcdf as jax_logcdf
from jax.scipy.stats.norm import cdf  as jax_cdf
import numpy as np

# Noise is fixed to 1 (standard DDM convention; absorbed into the scale of w).
# This gives a clean 3-parameter model that is well-identified.
N_PARAMS = 3
PARAM_NAMES = ['w', 'b', 'z']
_SIG = 1.0
_INVALID_LOG_LIK = -1e12


def log_lik_trial(params, dat_trial):
    """Per-trial log-likelihood for the DDM.

    Args:
        params    : (3,) array [w, b, z]
        dat_trial : dict with scalar fields
                    - inputs['c'] : signed contrast
                    - r           : response (1=right, 0=left)
                    - T           : reaction time
    """
    w, b, z = params
    T = dat_trial['T']
    valid = (
        jnp.isfinite(z) &
        jnp.isfinite(T) &
        (z > 0.0) &
        (T > 0.0)
    )
    return lax.cond(valid, lambda _: _log_lik_trial_valid(params, dat_trial),
                    lambda _: jnp.array(_INVALID_LOG_LIK), operand=None)


def _log_lik_trial_valid(params, dat_trial):
    """Per-trial log-likelihood assuming positive threshold and RT."""
    w, b, z = params
    c = dat_trial['inputs']['c']
    r = dat_trial['r']
    T = dat_trial['T']

    v = w * c + b                   # net drift toward right boundary
    d_chosen   =  (2 * r - 1) * v  # drift toward the chosen boundary
    d_unchosen = -(2 * r - 1) * v  # drift toward the unchosen boundary

    v2 = _SIG ** 2
    ll  = _log_inv_gauss_pdf(z, d_chosen,   v2, T)
    ll2 = _log_survival_from_cdf(_inv_gauss_cdf(z, d_unchosen, v2, T))
    return ll + ll2


def default_hyper(n_params=N_PARAMS, shared_sigma=False):
    return {
        'sigma':   float(2 ** -3) if shared_sigma else np.full(n_params, 2 ** -3),
        'sigInit': np.full(n_params, 2 **  4),
        'sigDay':  None,
    }


def default_E0(N, n_params=N_PARAMS):
    return np.array([
        np.linspace(0.5, 2.0, N),   # w
        np.zeros(N),                 # b
        np.ones(N),                  # z
    ])


# ---------------------------------------------------------------------------
# Inverse-Gaussian helpers (identical to race model)
# ---------------------------------------------------------------------------

@jit
def _log_inv_gauss_pdf(thr, drift, v, t):
    A = jnp.log(thr / jnp.sqrt(2 * jnp.pi * v * t ** 3))
    return A - (thr - drift * t) ** 2 / (2 * v * t)


@jit
def _inv_gauss_cdf(thr, drift, v, t):
    A    = jax_cdf((drift * t - thr) / jnp.sqrt(v * t))
    logB = 2.0 * thr * (drift / v) + jax_logcdf(-(drift * t + thr) / jnp.sqrt(v * t))
    return A + jnp.exp(logB)


@jit
def _log_survival_from_cdf(cdf):
    survival = 1.0 - jnp.clip(cdf, 0.0, 1.0)
    return jnp.log(jnp.maximum(survival, jnp.finfo(survival.dtype).tiny))

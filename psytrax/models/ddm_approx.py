"""Drift diffusion model — inverse-Gaussian (one-barrier) approximation.

A single accumulator drifts toward the chosen boundary with drift rate
    v = w·c + b
and the unchosen accumulator has not yet reached threshold.  This uses the
inverse-Gaussian first-passage-time density (same as one accumulator of the
race model), which is exact for a semi-infinite domain (one absorbing barrier).

Compared to ddm_exact.py, this model:
  - Is faster to evaluate (closed-form, no series)
  - Ignores the lower barrier (approximation that is accurate when z is large
    relative to diffusion noise, or equivalently when error rates are low)
  - Has 3 parameters instead of 4 (boundary separation and starting-point bias
    are not separately identifiable under this approximation)

Parameters (K = 3)
------------------
w   : contrast weight  (drift = w·c + b)
b   : drift bias       (baseline rightward drift)
z   : decision threshold / boundary (> 0)

See ddm_exact.py for the full two-barrier Wiener FPT solution
(Navarro & Fuss 2009 / Bogacz et al. 2006).
"""

import numpy as np
import jax.numpy as jnp
from jax import jit, lax
from jax.scipy.stats.norm import logcdf as jax_logcdf
from jax.scipy.stats.norm import cdf   as jax_cdf

# Within-trial noise is fixed at 1 (unidentifiable against z).
N_PARAMS    = 3
PARAM_NAMES = ['w', 'b', 'z']
_SIG        = 1.0
_INVALID_LOG_LIK = -1e12


def log_lik_trial(params, dat_trial):
    """Per-trial log-likelihood of the one-barrier DDM.

    Args:
        params    : (3,) array [w, b, z]
        dat_trial : dict with scalar fields
                    - inputs['c'] : signed contrast (positive = rightward)
                    - r           : response (1 = right, 0 = left)
                    - T           : reaction time (non-decision time already removed)

    Returns:
        scalar log-likelihood
    """
    w, b, z = params
    T = dat_trial['T']
    valid = (
        jnp.isfinite(z) & jnp.isfinite(T) &
        (z > 0.0) & (T > 0.0)
    )
    return lax.cond(
        valid,
        lambda _: _log_lik_valid(params, dat_trial),
        lambda _: jnp.array(_INVALID_LOG_LIK, dtype=params.dtype),
        operand=None,
    )


def _log_lik_valid(params, dat_trial):
    w, b, z = params
    c = dat_trial['inputs']['c']
    r = dat_trial['r']
    T = dat_trial['T']

    v = w * c + b                    # signed drift toward right boundary
    d_chosen   =  (2 * r - 1) * v   # drift toward the chosen boundary
    d_unchosen = -(2 * r - 1) * v   # drift toward the unchosen boundary

    v2  = _SIG ** 2
    ll  = _log_inv_gauss_pdf(z, d_chosen,   v2, T)
    ll2 = _log_survival_from_cdf(_inv_gauss_cdf(z, d_unchosen, v2, T))
    return ll + ll2


# ---------------------------------------------------------------------------
# Initialisation helpers
# ---------------------------------------------------------------------------

def default_hyper(n_params=N_PARAMS, shared_sigma=False):
    return {
        'sigma':   float(2 ** -3) if shared_sigma else np.full(n_params, 2 ** -3),
        'sigInit': np.full(n_params, 2 ** 4),
        'sigDay':  None,
    }


def default_E0(N, n_params=N_PARAMS):
    return np.array([
        np.linspace(0.5, 2.0, N),   # w
        np.zeros(N),                 # b
        np.ones(N),                  # z
    ])


# ---------------------------------------------------------------------------
# Inverse-Gaussian helpers (shared with race model)
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

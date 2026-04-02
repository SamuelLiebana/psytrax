"""Built-in race model (inverse-Gaussian race-to-threshold).

This module also serves as a template for writing your own model.
The only requirement is a JAX-compatible per-trial log-likelihood function
with the signature:

    log_lik_trial(params_k, dat_trial) -> scalar

params_k  : jnp array of shape (K,)
dat_trial : dict — same keys as your dat dict, but each trial-indexed field
            is a scalar (psytrax vmaps over trials automatically).

See psytrax/_likelihood.py for JAX porting tips.
"""

import jax.numpy as jnp
from jax import jit, lax
from jax.scipy.stats.norm import logcdf as jax_logcdf
from jax.scipy.stats.norm import cdf as jax_cdf
import numpy as np

# -----------------------------------------------------------------------
# Model specification
# -----------------------------------------------------------------------

N_PARAMS = 6
PARAM_NAMES = ['wr', 'wl', 'br', 'bl', 'z', 'sig_i']

# Fixed observation noise (not fitted)
_SIG_O = 1.0
_INVALID_LOG_LIK = -1e12


def log_lik_trial(params, dat_trial):
    """Per-trial log-likelihood of the race model.

    The model assumes two accumulators (right / left) with inverse-Gaussian
    first-passage-time distributions.  The chosen option's accumulator hits
    threshold z first; the unchosen accumulator has not yet hit threshold.

    Args:
        params    : (6,) array [wr, wl, br, bl, z, sig_i]
        dat_trial : dict with scalar fields
                    - inputs['c'] : signed contrast (positive = rightward)
                    - r           : response (1 = right, 0 = left)
                    - T           : reaction time

    Returns:
        scalar log-likelihood for this trial
    """
    wr, wl, br, bl, z, sig_i = params
    T = dat_trial['T']
    valid = (
        jnp.isfinite(z) &
        jnp.isfinite(sig_i) &
        jnp.isfinite(T) &
        (z > 0.0) &
        (sig_i >= 0.0) &
        (T > 0.0)
    )
    return lax.cond(valid, lambda _: _log_lik_trial_valid(params, dat_trial),
                    lambda _: jnp.array(_INVALID_LOG_LIK, dtype=params.dtype), operand=None)


def _log_lik_trial_valid(params, dat_trial):
    """Per-trial log-likelihood assuming positive threshold and RT."""
    wr, wl, br, bl, z, sig_i = params
    c = dat_trial['inputs']['c']
    r = dat_trial['r']
    T = dat_trial['T']

    # Drift rates
    drift1 = wr * jnp.maximum(c, 0.0) + br   # right accumulator
    drift2 = wl * jnp.maximum(-c, 0.0) + bl  # left accumulator

    # Diffusion variances
    v1 = wr ** 2 * sig_i ** 2 + _SIG_O ** 2
    v2 = wl ** 2 * sig_i ** 2 + _SIG_O ** 2

    # Chosen / unchosen accumulators
    drift_k    = r * drift1    + (1 - r) * drift2
    v_k        = r * v1        + (1 - r) * v2
    drift_kbar = (1 - r) * drift1 + r * drift2
    v_kbar     = (1 - r) * v1     + r * v2

    ll  = _log_inv_gauss_pdf(z, drift_k, v_k, T)
    ll2 = _log_survival_from_cdf(_inv_gauss_cdf(z, drift_kbar, v_kbar, T))
    return ll + ll2


# -----------------------------------------------------------------------
# Initialisation helpers
# -----------------------------------------------------------------------

def default_hyper(n_params=N_PARAMS, shared_sigma=False):
    """Reasonable starting hyperparameters for the race model."""
    if shared_sigma:
        sigma = float(2 ** -3)
    else:
        sigma = np.array([2 ** -3] * (n_params - 1) + [2 ** -10])  # tiny variance for sig_i
    return {
        'sigma': sigma,
        'sigInit': np.full(n_params, 2 ** 4),
        'sigDay': None,
    }


def default_E0(N, n_params=N_PARAMS):
    """Heuristic initial parameter matrix (K, N) for the race model."""
    E0 = np.array([
        np.linspace(0.05, 2.0,  N),  # wr
        np.linspace(0.05, 2.0,  N),  # wl
        np.linspace(0.4,  0.7,  N),  # br
        np.linspace(0.4,  0.7,  N),  # bl
        np.ones(N),                   # z
        np.full(N, 0.1),              # sig_i
    ])
    return E0


# -----------------------------------------------------------------------
# JAX helpers (inverse-Gaussian distribution)
# -----------------------------------------------------------------------

@jit
def _log_inv_gauss_pdf(thr, drift, v, t):
    A = jnp.log(thr / jnp.sqrt(2 * jnp.pi * v * t ** 3))
    return A - (thr - drift * t) ** 2 / (2 * v * t)


@jit
def _inv_gauss_cdf(thr, drift, v, t):
    A = jax_cdf((drift * t - thr) / jnp.sqrt(v * t))
    logB = 2.0 * thr * (drift / v) + jax_logcdf(-(drift * t + thr) / jnp.sqrt(v * t))
    return A + jnp.exp(logB)


@jit
def _log_survival_from_cdf(cdf):
    survival = 1.0 - jnp.clip(cdf, 0.0, 1.0)
    return jnp.log(jnp.maximum(survival, jnp.finfo(survival.dtype).tiny))

"""Drift diffusion model (DDM) — Navarro & Fuss (2009) / Bogacz et al. (2006) likelihood.

A Wiener process with drift v = w·c + b drifts between two absorbing barriers
(upper at a, lower at 0) starting from z·a.  The per-trial log-likelihood uses
the Navarro & Fuss (2009) hybrid series, which switches between two
complementary series depending on the standardised time τ = t/a²:

  Large-τ series (Ratcliff 1978 / Bogacz et al. 2006, good for τ ≳ 0.18):

      f_T(τ|w) = π × Σ_{k=1}^N  k · sin(kπw) · exp(−k²π²τ/2)

  Small-τ series (method of images, good for τ ≲ 0.18):

      f_T(τ|w) = (2πτ³)^{−1/2} × Σ_{k=−K}^{K}  (w+2k) · exp(−(w+2k)²/(2τ))

The full density is then recovered as:

      f+(t|v,a,z) = (1/a²) · exp(v·z − v²t/2) · f_T(t/a² | z/a)

The lower barrier follows by negating v and reflecting z → a−z.

Parameters (K = 4)
------------------
w   : contrast weight  (drift = w·c + b)
b   : drift bias       (baseline rightward drift)
a   : boundary separation (> 0)
z   : relative starting point in (0, 1);  absolute start = z·a  (0.5 = unbiased)

Note: within-trial noise σ is fixed at 1 (standard DDM convention; scale is absorbed
into a and w).  Subtract non-decision time from RTs before calling psytrax.fit().

References
----------
Ratcliff, R. (1978). Psychological Review, 85(2), 59–108.
Bogacz, R. et al. (2006). Psychological Review, 113(4), 700–765.
Navarro, D. J., & Fuss, I. G. (2009). Journal of Mathematical Psychology, 53, 222–230.
"""

import numpy as np
import jax.numpy as jnp
from jax import jit, lax

# ---------------------------------------------------------------------------
# Model specification
# ---------------------------------------------------------------------------

N_PARAMS    = 4
PARAM_NAMES = ['w', 'b', 'a', 'z']

_INVALID_LOG_LIK = -1e12
_TAU_CRIT        = 0.18   # switch threshold between large-t and small-t series
_N_LARGE         = 20     # terms in Ratcliff (large-τ) series
_K_SMALL         = 20     # half-width of image-charge (small-τ) series

# Static arrays computed once at import time
_K_VEC     = jnp.arange(1, _N_LARGE + 1,   dtype=jnp.float64)           # 1..N
_IMG_VEC   = jnp.arange(-_K_SMALL, _K_SMALL + 1, dtype=jnp.float64)    # −K..K


def log_lik_trial(params, dat_trial):
    """Per-trial log-likelihood of the DDM.

    Args:
        params    : (4,) array [w, b, a, z]
        dat_trial : dict with scalar fields
                    - inputs['c'] : signed contrast (positive = rightward)
                    - r           : response (1 = upper/right, 0 = lower/left)
                    - T           : RT in seconds (non-decision time already removed)

    Returns:
        scalar log-likelihood
    """
    w, b, a, z_rel = params
    T = dat_trial['T']
    valid = (
        jnp.isfinite(w) & jnp.isfinite(b) &
        jnp.isfinite(a) & jnp.isfinite(z_rel) &
        (a > 0.0) & (z_rel > 0.0) & (z_rel < 1.0) &
        (T > 0.0)
    )
    return lax.cond(
        valid,
        lambda _: _log_lik_valid(params, dat_trial),
        lambda _: jnp.array(_INVALID_LOG_LIK, dtype=params.dtype),
        operand=None,
    )


def _log_lik_valid(params, dat_trial):
    w, b, a, z_rel = params
    c = dat_trial['inputs']['c']
    r = dat_trial['r']
    T = dat_trial['T']

    v = w * c + b           # signed drift toward upper boundary
    z = z_rel * a           # absolute starting point

    # Lower-boundary response (r=0): negate drift and reflect starting point
    v_eff = jnp.where(r == 1.0, v, -v)
    z_eff = jnp.where(r == 1.0, z, a - z)

    return _log_fpt_upper(v_eff, a, z_eff, T)


@jit
def _log_fpt_upper(v, a, z, t):
    """Log of the upper-barrier FPT density via Navarro & Fuss (2009).

    Factored as  f+(t) = (1/a²) · exp(vz − v²t/2) · f_T(τ | w)
    where  τ = t/a²  and  w = z/a.
    """
    tau = t / (a ** 2)
    w   = z / a

    # --- Large-τ series (Ratcliff / Bogacz) ---
    sin_k  = jnp.sin(_K_VEC * jnp.pi * w)
    exp_k  = jnp.exp(-(_K_VEC ** 2) * (jnp.pi ** 2) * tau / 2.0)
    f_large = jnp.pi * jnp.sum(_K_VEC * sin_k * exp_k)

    # --- Small-τ series (method of images) ---
    nodes   = w + 2.0 * _IMG_VEC                              # w + 2k for k in −K..K
    exp_img = jnp.exp(-(nodes ** 2) / (2.0 * tau))
    f_small = jnp.sum(nodes * exp_img) / jnp.sqrt(2.0 * jnp.pi * tau ** 3)

    # Select series based on τ; both are always computed (JAX requires it under jit)
    f_T = jnp.where(tau >= _TAU_CRIT, f_large, f_small)

    log_prefactor = -2.0 * jnp.log(a) + v * z - 0.5 * v ** 2 * t
    return log_prefactor + jnp.log(jnp.maximum(f_T, jnp.finfo(jnp.float64).tiny))


# ---------------------------------------------------------------------------
# Initialisation helpers
# ---------------------------------------------------------------------------

def default_hyper(n_params=N_PARAMS, shared_sigma=False):
    """Reasonable starting hyperparameters for the DDM."""
    if shared_sigma:
        sigma = float(2 ** -3)
    else:
        # z lives in (0,1) so give it less process noise than w / b / a
        sigma = np.array([2 ** -3, 2 ** -3, 2 ** -4, 2 ** -6])
    return {
        'sigma':   sigma,
        'sigInit': np.full(n_params, 2 ** 4),
        'sigDay':  None,
    }


def default_E0(N, n_params=N_PARAMS):
    """Heuristic initial parameter matrix (K, N) for the DDM."""
    return np.tile(np.array([1.0, 0.0, 1.0, 0.5])[:, None], N)

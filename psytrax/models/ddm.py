"""Drift diffusion model (DDM) — Navarro & Fuss (2009) / Bogacz et al. (2006) likelihood.

A Wiener process with drift v = w·c + b diffuses between two absorbing barriers
(upper at a, lower at 0) starting from a/2 (unbiased start).  The per-trial
log-likelihood uses the Navarro & Fuss (2009) hybrid series, which switches
between two complementary series depending on the standardised time τ = t/a²:

  Large-τ series (Ratcliff 1978 / Bogacz et al. 2006, good for τ ≳ 0.18):

      f_T(τ|w) = π × Σ_{k=1}^N  k · sin(kπw) · exp(−k²π²τ/2)

  Small-τ series (method of images, good for τ ≲ 0.18):

      f_T(τ|w) = (2πτ³)^{−1/2} × Σ_{k=−K}^{K}  (w+2k) · exp(−(w+2k)²/(2τ))

The full density is then recovered as:

      f+(t|v,a) = (1/a²) · exp(v·a/2 − v²t/2) · f_T(t/a² | 1/2)

The lower barrier follows by negating v and reflecting the start point.

Parameters (K = 3)
------------------
w   : contrast weight  (drift = w·c + b)
b   : drift bias       (baseline rightward drift)
a   : boundary separation (> 0)

The starting point is fixed at a/2 (unbiased).  This eliminates the well-known
b/z degeneracy in the DDM: a starting-point bias z and a drift bias b both
shift response probability toward the upper boundary, so fitting both as free
trial-varying parameters lets the inferred trajectories swap weight between
them and recover sign-flipped solutions.  Bias is captured exclusively by `b`
in this implementation — the standard convention in many DDM applications.

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

N_PARAMS    = 3
PARAM_NAMES = ['w', 'b', 'a']

# Relative starting point fixed at the midpoint between barriers (unbiased).
_Z_REL = 0.5

DATA_SPEC = {
    'inputs': {
        'c': {'description': 'Signed stimulus strength (e.g. contrast)', 'required': True},
    },
    'response': {
        'key': 'r',
        'description': 'Choice — discrete (1=upper/right, 0=lower/left) or continuous in [0, 1]',
        'required': True,
    },
    'rt': {
        'key': 'T',
        'description': 'Reaction time in seconds (non-decision time already removed)',
        'required': True,
    },
}

_INVALID_LOG_LIK = -1e12
_TAU_CRIT        = 0.18   # switch threshold between large-t and small-t series
_N_LARGE         = 20     # terms in Ratcliff (large-τ) series
_K_SMALL         = 20     # half-width of image-charge (small-τ) series

# Static arrays computed once at import time
_K_VEC     = np.arange(1, _N_LARGE + 1,         dtype=np.float64)       # 1..N  (numpy; cast by JAX inside JIT)
_IMG_VEC   = np.arange(-_K_SMALL, _K_SMALL + 1, dtype=np.float64)       # −K..K


def log_lik_trial(params, dat_trial, model_hyper=None):
    """Per-trial log-likelihood of the DDM.

    Args:
        params      : (3,) array [w, b, a]
        dat_trial   : dict with scalar fields
                      - inputs['c'] : signed contrast (positive = rightward)
                      - r           : response (1 = upper/right, 0 = lower/left)
                      - T           : RT in seconds (non-decision time already removed)
        model_hyper : unused (DDM has no model-level hyperparameters).

    Returns:
        scalar log-likelihood
    """
    w, b, a = params
    T = dat_trial['T']
    valid = (
        jnp.isfinite(w) & jnp.isfinite(b) & jnp.isfinite(a) &
        (a > 0.0) & (T > 0.0)
    )
    return lax.cond(
        valid,
        lambda _: _log_lik_valid(params, dat_trial),
        lambda _: jnp.array(_INVALID_LOG_LIK, dtype=params.dtype),
        operand=None,
    )


def _log_lik_valid(params, dat_trial):
    w, b, a = params
    c = dat_trial['inputs']['c']
    r = dat_trial['r']
    T = dat_trial['T']

    v = w * c + b           # signed drift toward upper boundary
    z = _Z_REL * a          # absolute starting point at a/2

    # Lower-boundary response (r=0): negate drift and reflect starting point.
    # With z = a/2 the reflection (a − z) is also a/2, so z_eff = a/2 always —
    # the only effective change between r=0 and r=1 is the sign of the drift.
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
    """Reasonable starting hyperparameters for the DDM.

    Initial sigma is scaled to give EB room to escape the constant-trajectory
    local mode of the marginal-likelihood surface.  EB optimises sigma anyway,
    so the starting value mostly determines which local mode the outer loop
    settles into.
    """
    if shared_sigma:
        sigma = float(2 ** -1)
    else:
        # `a` is constrained > 0 so it gets a tighter starting prior than w/b.
        sigma = np.array([2 ** -1, 2 ** -1, 2 ** -2])
    return {
        'sigma':   sigma,
        'sigInit': np.full(n_params, 2 ** 4),
        'sigDay':  None,
    }


def default_E0(N, n_params=N_PARAMS):
    """Heuristic initial parameter matrix (K, N) for the DDM."""
    return np.tile(np.array([1.0, 0.0, 1.0])[:, None], N)


def sample_trial(params, dat_trial, rng, model_hyper=None, dt=1e-3, t_max=10.0):
    """Sample one trial from the two-barrier DDM via Euler-Maruyama.

    A Wiener process with drift ``v = w·c + b`` and unit variance per unit
    time is integrated forward from the unbiased start point ``a/2`` until
    it hits 0 (left/lower) or ``a`` (right/upper).  This is the most general
    sampler available for the exact DDM — closed-form sampling does not
    have a simple form.

    Args:
        params    : (3,) array [w, b, a]
        dat_trial : dict with scalar field ``dat_trial['inputs']['c']``.
        rng       : numpy.random.Generator
        dt        : integration step size in seconds (default 1 ms).
        t_max     : maximum simulated time before declaring a non-decision.

    Returns:
        dict with keys ``'r'`` (1=upper/right, 0=lower/left) and ``'T'`` (RT).
        If neither boundary is hit before ``t_max`` the trial is forced to
        the closer boundary at time ``t_max``.
    """
    w, b, a = (float(p) for p in params)
    c = float(dat_trial['inputs']['c'])

    if not (a > 0.0):
        # Slider-driven trajectories may briefly leave the valid region.
        return {'r': float(rng.integers(0, 2)), 'T': float(t_max)}

    v = w * c + b
    x = _Z_REL * a              # unbiased start at a/2
    sqrt_dt = np.sqrt(dt)
    n_steps = int(np.ceil(t_max / dt))
    noise = rng.standard_normal(n_steps)

    for i in range(n_steps):
        x += v * dt + sqrt_dt * noise[i]
        if x >= a:
            return {'r': 1.0, 'T': float((i + 1) * dt)}
        if x <= 0.0:
            return {'r': 0.0, 'T': float((i + 1) * dt)}
    # Forced choice at t_max
    return {'r': 1.0 if x > a / 2.0 else 0.0, 'T': float(t_max)}


def default_learning_rule(reward_key='reward'):
    """Return a REINFORCE learning rule for the DDM.

    The update direction at trial t is the score function
    ∇_θ log p(y_t, RT_t | x_t, θ) scaled by the reward signal.  Because
    the DDM likelihood is fully differentiable in JAX, the gradient is
    computed automatically via ``jax.grad``.

    The data dict must contain ``data['inputs']['reward']``, typically
    1 for correct and 0 otherwise.

    Returns
    -------
    learning_rule : callable
    """
    from psytrax.learning_rules import make_reinforce
    return make_reinforce(log_lik_trial, reward_key=reward_key)

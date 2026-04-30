"""Built-in race model (inverse-Gaussian race-to-threshold).

This module also serves as a template for writing your own model.
A psytrax model exposes:

    log_lik_trial(params, dat_trial, model_hyper) -> scalar
    sample_trial (params, dat_trial, rng, model_hyper) -> dict   # for simulation
    default_model_hyper() -> dict                                 # optional

``params`` are the *trial-varying* parameters (one (K,) vector per trial),
``model_hyper`` carries *constants* shared across all trials and is jointly
optimised by Empirical Bayes alongside ``sigma``.

The race model has K = 5 trial-varying parameters (wr, wl, br, bl, z) and one
model-level scalar (``sig_i`` — within-trial accumulator noise).

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

N_PARAMS = 5
PARAM_NAMES = ['wr', 'wl', 'br', 'bl', 'z']

DATA_SPEC = {
    'inputs': {
        'c': {'description': 'Signed stimulus strength (e.g. contrast)', 'required': True},
    },
    'response': {
        'key': 'r',
        'description': 'Choice — discrete (1=right, 0=left) or continuous in [0, 1]',
        'required': True,
    },
    'rt': {
        'key': 'T',
        'description': 'Reaction time in seconds',
        'required': True,
    },
}

# Fixed observation noise (not fitted)
_SIG_O = 1.0
_INVALID_LOG_LIK = -1e12

# Default initial value for the within-trial accumulator noise (used as the
# starting point for the EB outer loop unless the caller overrides it).
DEFAULT_SIG_I = 0.1


def default_model_hyper():
    """Initial values for the model-level scalar hyperparameters.

    The race model has one: ``sig_i`` — the within-trial accumulator noise.
    Empirical Bayes optimises this alongside ``sigma`` by maximising the log
    evidence, so the value below is just a starting point.
    """
    return {'sig_i': float(DEFAULT_SIG_I)}


def log_lik_trial(params, dat_trial, model_hyper):
    """Per-trial log-likelihood of the race model.

    The model assumes two accumulators (right / left) with inverse-Gaussian
    first-passage-time distributions.  The chosen option's accumulator hits
    threshold z first; the unchosen accumulator has not yet hit threshold.

    Args:
        params      : (5,) array [wr, wl, br, bl, z]
        dat_trial   : dict with scalar fields
                      - inputs['c'] : signed contrast (positive = rightward)
                      - r           : response (1 = right, 0 = left)
                      - T           : reaction time
        model_hyper : dict with key 'sig_i' (within-trial accumulator noise)

    Returns:
        scalar log-likelihood for this trial
    """
    wr, wl, br, bl, z = params
    sig_i = model_hyper['sig_i']
    T = dat_trial['T']
    valid = (
        jnp.isfinite(z) &
        jnp.isfinite(sig_i) &
        jnp.isfinite(T) &
        (z > 0.0) &
        (sig_i >= 0.0) &
        (T > 0.0)
    )
    return lax.cond(
        valid,
        lambda _: _log_lik_trial_valid(params, dat_trial, sig_i),
        lambda _: jnp.array(_INVALID_LOG_LIK, dtype=params.dtype),
        operand=None,
    )


def _log_lik_trial_valid(params, dat_trial, sig_i):
    """Per-trial log-likelihood assuming positive threshold and RT."""
    wr, wl, br, bl, z = params
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
        sigma = np.array([2 ** -3] * n_params)
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
    ])
    return E0


# -----------------------------------------------------------------------
# Forward simulation
# -----------------------------------------------------------------------

def sample_trial(params, dat_trial, rng, model_hyper):
    """Sample one trial from the race model.

    Two independent inverse-Gaussian first-passage times are drawn (one per
    accumulator); the winner determines the choice and the elapsed time.

    Args:
        params      : (5,) array [wr, wl, br, bl, z]
        dat_trial   : dict with scalar fields — must contain
                      ``dat_trial['inputs']['c']`` (signed contrast).
        rng         : numpy.random.Generator
        model_hyper : dict with key 'sig_i'.

    Returns:
        dict with keys ``'r'`` (1=right, 0=left) and ``'T'`` (RT in seconds).
    """
    wr, wl, br, bl, z = (float(p) for p in params)
    sig_i = float(model_hyper['sig_i'])
    c = float(dat_trial['inputs']['c'])

    # Drift rates and diffusion variances for the two accumulators.
    drift_r = wr * max(c, 0.0) + br
    drift_l = wl * max(-c, 0.0) + bl
    v_r = wr ** 2 * sig_i ** 2 + _SIG_O ** 2
    v_l = wl ** 2 * sig_i ** 2 + _SIG_O ** 2

    t_r = _sample_inv_gauss_fpt(z, drift_r, v_r, rng)
    t_l = _sample_inv_gauss_fpt(z, drift_l, v_l, rng)

    if t_r < t_l:
        return {'r': 1.0, 'T': float(t_r)}
    return {'r': 0.0, 'T': float(t_l)}


def _sample_inv_gauss_fpt(threshold, drift, variance, rng):
    """Sample an inverse-Gaussian first-passage time.

    Returns +inf if the drift is non-positive (the accumulator never
    deterministically reaches threshold under a one-shot Wald draw — the race
    model treats this as "never wins").  This keeps the sampler well-defined
    when slider-driven trajectories briefly enter that region.
    """
    if not np.isfinite(drift) or drift <= 0.0 or threshold <= 0.0 or variance <= 0.0:
        return np.inf
    # Time to threshold z under a Brownian motion with drift `drift` and
    # variance per unit time `variance` is IG with mean=z/drift, shape=z²/variance.
    mean = threshold / drift
    shape = threshold ** 2 / variance
    return float(rng.wald(mean, shape))


def default_learning_rule(reward_key='reward'):
    """Return a REINFORCE learning rule for the race model.

    The update direction at trial t is the score function
    ∇_θ log p(y_t, RT_t | x_t, θ) scaled by the reward signal.  Because
    the race model likelihood is fully differentiable in JAX, the gradient
    is computed automatically via ``jax.grad``.

    The data dict must contain ``data['inputs']['reward']``, typically
    1 for correct and 0 otherwise.
    """
    from psytrax.learning_rules import make_reinforce
    return make_reinforce(log_lik_trial, reward_key=reward_key)


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

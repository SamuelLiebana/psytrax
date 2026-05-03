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

import jax
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
    'dopamine': {
        'key': 'dopamine',
        'description': (
            'Per-trial peak dopamine response (e.g. peak of MA(window=10) of '
            'the stimulus-aligned signal in 0.2-0.35s). Optional. NaN marks a '
            'missing trial. Activated by setting model_hyper["sig_DA"].'
        ),
        'required': False,
    },
}

# Fixed observation noise (not fitted)
_SIG_O = 1.0
_INVALID_LOG_LIK = -1e12

# Default initial value for the within-trial accumulator noise (used as the
# starting point for the EB outer loop unless the caller overrides it).
DEFAULT_SIG_I = 0.1
# Default initial value for the dopamine observation noise (only used when
# fitting the optional dopamine likelihood term).
DEFAULT_SIG_DA = 0.2

# Default starting values for the dopamine sigmoid response function:
#     pred = σ(da_beta * (w_eff · c / z − da_offset))
# Both `da_beta` (inverse temperature) and `da_offset` (centre on the
# linear-prediction axis) live in ``model_hyper`` and are optimised
# jointly with ``sig_DA`` and ``sig_i`` by the EB outer loop.  The defaults
# (β = 6, centre = 0.5) place the steep part of the curve around the
# midpoint of a 0–1 normalised dopamine signal, but EB will move them as
# needed to match the data.
DEFAULT_DA_BETA   = 6.0
DEFAULT_DA_OFFSET = 0.5


def default_model_hyper():
    """Initial values for the model-level scalar hyperparameters.

    The race model has one: ``sig_i`` — the within-trial accumulator noise.
    Empirical Bayes optimises this alongside ``sigma`` by maximising the log
    evidence, so the value below is just a starting point.
    """
    return {'sig_i': float(DEFAULT_SIG_I)}


def default_model_hyper_with_dopamine():
    """Model-level hyperparameters when the dopamine term is enabled.

    The dopamine peak is modelled as
        ``N(σ(da_beta · (w_eff · c / z − da_offset)), sig_DA²)``
    where ``w_eff = wr`` if ``c >= 0`` else ``wl``.  All four scalars are
    optimised jointly by Empirical Bayes:

      * ``sig_i``     — within-trial accumulator noise (existing race scalar)
      * ``sig_DA``    — Gaussian std on the dopamine peak
      * ``da_beta``   — sigmoid inverse temperature (slope)
      * ``da_offset`` — sigmoid centre on the linear-prediction axis

    Pass this dict explicitly to ``psytrax.fit(..., model_hyper=...)`` when
    fitting the joint choice + RT + dopamine likelihood.
    """
    return {
        'sig_i':     float(DEFAULT_SIG_I),
        'sig_DA':    float(DEFAULT_SIG_DA),
        'da_beta':   float(DEFAULT_DA_BETA),
        'da_offset': float(DEFAULT_DA_OFFSET),
    }


def log_lik_trial(params, dat_trial, model_hyper):
    """Per-trial log-likelihood of the race model.

    The model assumes two accumulators (right / left) with inverse-Gaussian
    first-passage-time distributions.  The chosen option's accumulator hits
    threshold z first; the unchosen accumulator has not yet hit threshold.

    When ``dat_trial`` contains a ``'dopamine'`` field and ``model_hyper``
    contains a ``'sig_DA'`` scalar, an extra Gaussian likelihood term is
    added:  the per-trial dopamine peak is modelled as
    ``N(σ(da_beta · (w_eff · c / z − da_offset)), sig_DA²)`` where
    ``w_eff = wr if c >= 0 else wl``.  ``da_beta`` and ``da_offset`` are
    also pulled from ``model_hyper`` (see
    :func:`default_model_hyper_with_dopamine`).
    Trials whose dopamine value is NaN contribute zero to this term, so
    per-trial missing data is allowed.

    Args:
        params      : (5,) array [wr, wl, br, bl, z]
        dat_trial   : dict with scalar fields
                      - inputs['c'] : signed contrast (positive = rightward)
                      - r           : response (1 = right, 0 = left)
                      - T           : reaction time
                      - dopamine    : optional per-trial scalar (NaN allowed)
        model_hyper : dict with key 'sig_i' (within-trial accumulator noise)
                      and optional 'sig_DA' (Gaussian std on dopamine peak).

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
    base = lax.cond(
        valid,
        lambda _: _log_lik_trial_valid(params, dat_trial, sig_i),
        lambda _: jnp.array(_INVALID_LOG_LIK, dtype=params.dtype),
        operand=None,
    )

    # Dopamine term — Python-level conditional on dat_trial / model_hyper
    # structure, evaluated once at trace time, so this stays JAX-safe under
    # vmap.
    if 'dopamine' in dat_trial and 'sig_DA' in model_hyper:
        base = base + _log_lik_dopamine(params, dat_trial, model_hyper)
    return base


def _log_lik_dopamine(params, dat_trial, model_hyper):
    """Gaussian log-likelihood of the per-trial dopamine peak.

    NaN dopamine values are masked out (they contribute 0).  When ``z`` is
    non-positive the prediction is undefined, so the term collapses to 0
    rather than producing -inf — the choice/RT branch already imposes the
    ``z > 0`` validity check on the rest of the trial likelihood.

    The NaN dopamine value is replaced with a finite dummy *before* the
    squared-error computation so that JAX's grad-through-``jnp.where``
    doesn't propagate NaN gradients on missing trials.
    """
    wr, wl, br, bl, z = params
    c   = dat_trial['inputs']['c']
    da  = dat_trial['dopamine']
    sig_DA    = model_hyper['sig_DA']
    da_beta   = model_hyper.get('da_beta',   DEFAULT_DA_BETA)
    da_offset = model_hyper.get('da_offset', DEFAULT_DA_OFFSET)
    w_eff = jnp.where(c >= 0.0, wr, wl)
    z_safe = jnp.where(z > 0.0, z, 1.0)
    pred = jax.nn.sigmoid(da_beta * (w_eff * c / z_safe - da_offset))
    valid = (
        jnp.isfinite(da)
        & jnp.isfinite(sig_DA)
        & (sig_DA > 0.0)
        & (z > 0.0)
    )
    # Mask NaN/inf dopamine to a finite dummy so neither value nor gradient
    # picks up a NaN from the masked-out branch.
    da_safe = jnp.where(jnp.isfinite(da), da, 0.0)
    log_norm = -jnp.log(sig_DA) - 0.5 * jnp.log(2.0 * jnp.pi)
    sq       = -0.5 * ((da_safe - pred) / sig_DA) ** 2
    return jnp.where(valid, log_norm + sq, jnp.array(0.0, dtype=params.dtype))


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
    """Reasonable starting hyperparameters for the race model.

    Initial sigma is set loose enough that EB can escape the
    constant-trajectory local mode of the marginal-likelihood surface.
    EB optimises sigma anyway, so the starting value mostly determines
    which local mode the outer loop settles into.
    """
    if shared_sigma:
        sigma = float(2 ** -1)
    else:
        sigma = np.array([2 ** -1] * n_params)
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
        out = {'r': 1.0, 'T': float(t_r)}
    else:
        out = {'r': 0.0, 'T': float(t_l)}

    # Optional dopamine sample — emitted only when sig_DA is in model_hyper
    # so existing simulations keep their (r, T)-only output.
    if 'sig_DA' in model_hyper:
        sig_DA = float(model_hyper['sig_DA'])
        if z > 0.0 and sig_DA > 0.0:
            da_beta   = float(model_hyper.get('da_beta',   DEFAULT_DA_BETA))
            da_offset = float(model_hyper.get('da_offset', DEFAULT_DA_OFFSET))
            w_eff = wr if c >= 0.0 else wl
            pred = 1.0 / (1.0 + np.exp(-da_beta * (w_eff * c / z - da_offset)))
            out['dopamine'] = float(rng.normal(pred, sig_DA))
        else:
            out['dopamine'] = float('nan')
    return out


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

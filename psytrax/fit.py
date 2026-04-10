import os
import warnings
import numpy as np
from datetime import datetime

import jax
import jax.numpy as jnp

from psytrax._likelihood import make_likelihood_fns
from psytrax._hyper_opt import hyperOpt
from psytrax._helper.helperFunctions import trim
from psytrax._device import setup_device
import psytrax._map as _map_module


def fit(data, log_lik_trial, n_params,
        param_names=None,
        hyper=None,
        shared_sigma=False,
        session_boundaries=False,
        E0=None,
        n_trials=None,
        hess_calc='weights',
        device='auto',
        precision='float64',
        optimizer='jax',
        map_tol=1e-6,
        subject_name=None,
        save=False,
        verbose=False):
    """Fit a decision model to behavioural data using Empirical Bayes + Laplace.

    Parameters
    ----------
    data : dict or str
        Either a data dict or a path to a .npy file containing one.
        Required keys:
          - 'inputs'   : dict of input arrays, each of shape (N, ...)
          - 'responses': integer array of shape (N,)  [also accepts 'r']
        Optional keys:
          - 'times'           : RT array shape (N,)   [also accepts 'T']
          - 'session_lengths' : array of per-session trial counts [also 'dayLength']
    log_lik_trial : callable
        Per-trial log-likelihood with signature
            log_lik_trial(params_k, dat_trial) -> scalar
        Must be JAX-traceable (written with jax.numpy).
        See psytrax/_likelihood.py for porting tips and psytrax/models/race.py
        for a complete example.
    n_params : int
        Number of parameters per trial (K).
    param_names : list[str], optional
        Human-readable names for the K parameters.
    hyper : dict, optional
        Initial hyperparameters.  Must contain 'sigma' (positive scalar or
        array of length K).
        May also contain 'sigInit' and 'sigDay'.
        Defaults to sigma=2^{-3} for all params.
    shared_sigma : bool
        If True and `hyper` is not provided, use a single scalar process-noise
        sigma shared across all parameters.
    session_boundaries : bool
        If True, fit a larger process noise ('sigDay') at session boundaries.
        Requires 'session_lengths' in data.
    E0 : np.ndarray, optional
        Initial parameter matrix of shape (K, N).  Defaults to 0.01 everywhere.
        For the built-in race model, use psytrax.models.race.default_E0(N).
    n_trials : int, optional
        Use only the first n_trials trials.
    hess_calc : str
        Which credible intervals to compute: 'weights', 'hyper', 'All', or None.
    map_tol : float
        Convergence tolerance for the inner MAP optimization. Larger values
        stop earlier and usually run faster.
    subject_name : str, optional
        Used as the stem of the save filename.
    save : bool
        If True, save results to disk and return the path.
        If False (default), return the results dict directly.
    verbose : bool
        Print optimisation progress.

    Returns
    -------
    dict (save=False) or str path (save=True)
        Keys: 'params' (K×N), 'param_names', 'hyper', 'log_evidence',
              'hess_info', 'data', 'duration'.
    """

    # ------------------------------------------------------------------
    # Device + precision selection
    # ------------------------------------------------------------------
    setup_device(device, verbose=verbose)

    if precision not in ('float32', 'float64'):
        raise ValueError(f"precision must be 'float32' or 'float64', got '{precision}'")
    _dtype = jnp.float32 if precision == 'float32' else jnp.float64
    _map_module._JAX_DTYPE = _dtype
    if verbose:
        print(f'psytrax: JAX precision {precision}')

    if optimizer not in ('jax', 'scipy'):
        raise ValueError(f"optimizer must be 'jax' or 'scipy', got '{optimizer}'")
    if optimizer == 'jax':
        try:
            import optax as _optax  # noqa: F401 — check availability before use
            from psytrax._jax_map import getMAP_jax
            _map_fn = getMAP_jax
            _optimizer_used = 'jax (optax L-BFGS)'
            if verbose:
                print('psytrax: optimizer jax (optax L-BFGS)')
        except ImportError:
            warnings.warn(
                "optax not installed — falling back to scipy trust-NCG optimizer. "
                "Install with: pip install optax",
                UserWarning, stacklevel=2,
            )
            _map_fn = None
            _optimizer_used = 'scipy trust-NCG (optax unavailable)'
            if verbose:
                print('psytrax: optimizer scipy (optax unavailable)')
    else:
        _map_fn = None   # hyperOpt defaults to getMAP (scipy trust-ncg)
        _optimizer_used = 'scipy trust-NCG'
        if verbose:
            print('psytrax: optimizer scipy (trust-NCG)')

    # ------------------------------------------------------------------
    # Load / normalise data
    # ------------------------------------------------------------------
    if isinstance(data, (str, os.PathLike)):
        raw = np.load(os.fspath(data), allow_pickle=True).item()
    elif isinstance(data, dict):
        raw = data
    else:
        raise TypeError(f"data must be a dict or file path, not {type(data)}")

    dat = _normalise_dat(raw)
    _validate_dat(dat)

    N_total = len(dat['r'])
    if n_trials is not None:
        if not isinstance(n_trials, (int, np.integer)):
            raise TypeError(f"n_trials must be an integer, not {type(n_trials)}")
        if n_trials <= 0:
            raise ValueError("n_trials must be positive")
    N = n_trials if n_trials is not None and n_trials < N_total else N_total

    if N != N_total:
        dat = trim(dat, END=N)
        _validate_dat(dat)

    # ------------------------------------------------------------------
    # Hyperparameters
    # ------------------------------------------------------------------
    K = n_params

    if param_names is not None and len(param_names) != K:
        raise ValueError(f"param_names must have length {K}, got {len(param_names)}")

    if hyper is None:
        hyper = {
            'sigma': float(2 ** -3) if shared_sigma else np.full(K, 2 ** -3),
            'sigInit': np.full(K, 2 ** 4),
            'sigDay': None,
        }

    opt_list = ['sigma']

    if session_boundaries:
        if 'dayLength' not in dat or dat['dayLength'] is None:
            raise ValueError("session_boundaries=True requires 'session_lengths' in data")
        if hyper.get('sigDay') is None:
            hyper['sigDay'] = float(2 ** -2) if shared_sigma else np.full(K, 2 ** -2)
        opt_list = ['sigma', 'sigDay']

    _validate_hyper(hyper, K)

    # ------------------------------------------------------------------
    # Build batched likelihood functions
    # ------------------------------------------------------------------
    log_lik_fns = make_likelihood_fns(log_lik_trial)

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------
    if E0 is None:
        if verbose:
            print('Finding warm-start initialisation (constant-parameter fit)...')
        const_params = _warm_start_constant(dat, log_lik_fns, K, N, verbose=verbose, dtype=_dtype)
        E0 = np.tile(const_params[:, np.newaxis], N)
    else:
        _check_E0_validity(E0, dat, log_lik_fns, K, N)

    # ------------------------------------------------------------------
    # Run hyperparameter optimisation
    # ------------------------------------------------------------------
    start = datetime.now()
    best_hyper, log_evd, eMode, hess_info = hyperOpt(
        dat=dat,
        hyper=hyper,
        n_params=K,
        log_lik_fns=log_lik_fns,
        optList=opt_list,
        E0=E0,
        method=None,
        showOpt=int(verbose),
        hess_calc=hess_calc,
        show_progress=verbose,
        map_tol=map_tol,
        map_fn=_map_fn,
    )
    duration = datetime.now() - start

    params = np.reshape(eMode, (K, N), order='C')

    results = {
        'params': params,
        'param_names': param_names or [str(i) for i in range(K)],
        'hyper': best_hyper,
        'log_evidence': log_evd,
        'hess_info': hess_info,
        'data': dat,
        'n_trials': N,
        'duration': duration,
        'optimizer': _optimizer_used,
    }

    if not save:
        return results

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    if subject_name is None:
        raise ValueError("subject_name required when save=True")

    os.makedirs('fits', exist_ok=True)
    suffix = f'_N{N}' if n_trials is not None else ''
    path = os.path.join('fits', f'{subject_name}{suffix}_fit.npy')
    np.save(path, results)
    return path


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

def _normalise_dat(raw):
    """Translate user-facing key names to the internal convention."""
    dat = {}

    if 'inputs' not in raw:
        raise KeyError("data must contain an 'inputs' dict")
    dat['inputs'] = raw['inputs']

    # responses
    if 'responses' in raw:
        dat['r'] = np.asarray(raw['responses'])
    elif 'r' in raw:
        dat['r'] = np.asarray(raw['r'])
    else:
        raise KeyError("data must contain 'responses' or 'r'")

    # reaction times (optional)
    if 'times' in raw:
        dat['T'] = np.asarray(raw['times'])
    elif 'T' in raw:
        dat['T'] = np.asarray(raw['T'])

    # session lengths (optional)
    if 'session_lengths' in raw:
        dat['dayLength'] = np.asarray(raw['session_lengths'])
    elif 'dayLength' in raw:
        dat['dayLength'] = np.asarray(raw['dayLength'])

    return dat


def _validate_dat(dat):
    """Validate trial-aligned arrays before entering JAX/optimizer code."""
    if not isinstance(dat['inputs'], dict) or not dat['inputs']:
        raise ValueError("data['inputs'] must be a non-empty dict of trial-aligned arrays")

    r = np.asarray(dat['r'])
    if r.ndim != 1:
        raise ValueError(f"responses must be one-dimensional, got shape {r.shape}")
    if r.size == 0:
        raise ValueError("responses must contain at least one trial")
    if not np.all(np.isfinite(r)):
        raise ValueError("responses must be finite")
    dat['r'] = r

    N = r.shape[0]
    for key, value in dat['inputs'].items():
        arr = np.asarray(value)
        if arr.ndim == 0:
            raise ValueError(f"input '{key}' must include a trial dimension of length {N}")
        if arr.shape[0] != N:
            raise ValueError(
                f"input '{key}' has {arr.shape[0]} trials but responses have {N}"
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"input '{key}' must be finite")
        dat['inputs'][key] = arr

    if 'T' in dat:
        T = np.asarray(dat['T'])
        if T.ndim != 1:
            raise ValueError(f"times must be one-dimensional, got shape {T.shape}")
        if T.shape[0] != N:
            raise ValueError(f"times have {T.shape[0]} trials but responses have {N}")
        if not np.all(np.isfinite(T)):
            raise ValueError("times must be finite")
        if np.any(T <= 0):
            raise ValueError("times must be strictly positive")
        dat['T'] = T

    if 'dayLength' in dat:
        day_length = np.asarray(dat['dayLength'])
        if day_length.ndim != 1:
            raise ValueError(
                f"session_lengths must be one-dimensional, got shape {day_length.shape}"
            )
        if day_length.size == 0:
            raise ValueError("session_lengths must contain at least one session")
        if not np.all(np.isfinite(day_length)):
            raise ValueError("session_lengths must be finite")
        if not np.all(np.equal(day_length, np.floor(day_length))):
            raise ValueError("session_lengths must be integers")
        day_length = day_length.astype(int, copy=False)
        if np.any(day_length <= 0):
            raise ValueError("session_lengths must be positive")
        if np.sum(day_length) != N:
            raise ValueError(
                f"session_lengths sum to {np.sum(day_length)} but responses have {N} trials"
            )
        dat['dayLength'] = day_length


def _validate_hyper(hyper, n_params):
    """Validate prior hyperparameters before optimization starts."""
    if not isinstance(hyper, dict):
        raise TypeError(f"hyper must be a dict, not {type(hyper)}")

    for key in ('sigma', 'sigInit', 'sigDay'):
        if key not in hyper or hyper[key] is None:
            if key == 'sigma':
                raise KeyError("hyper must contain 'sigma'")
            continue
        _validate_hyper_value(key, hyper[key], n_params)


def _validate_hyper_value(name, value, n_params):
    """Ensure hyperparameters are positive scalars or length-K vectors."""
    if np.isscalar(value):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"hyper['{name}'] must be positive and finite")
        return

    arr = np.asarray(value, dtype=float)
    if arr.shape != (n_params,):
        raise ValueError(
            f"hyper['{name}'] must have shape ({n_params},), got {arr.shape}"
        )
    if not np.all(np.isfinite(arr)) or np.any(arr <= 0):
        raise ValueError(f"hyper['{name}'] must contain only positive finite values")


def _to_jax(dat, dtype=None):
    """Recursively convert numpy arrays in a data dict to jax arrays."""
    result = {}
    for k, v in dat.items():
        if v is None:
            result[k] = v
        elif isinstance(v, dict):
            result[k] = _to_jax(v, dtype=dtype)
        elif isinstance(v, np.ndarray):
            if dtype is not None and np.issubdtype(v.dtype, np.floating):
                result[k] = jnp.asarray(v, dtype=dtype)
            else:
                result[k] = jnp.asarray(v)
        else:
            result[k] = v
    return result


def _warm_start_constant(dat, log_lik_fns, K, N, verbose=False,
                         dtype=None):
    """Find constant MAP parameters by maximising total log-likelihood.

    Optimises a single (K,) parameter vector shared across all N trials,
    using L-BFGS-B with JAX-computed gradients.  The result provides a
    data-informed E0 that avoids the sentinel-value region.

    Args:
        dat         : normalised data dict
        log_lik_fns : (log_likelihood_fn, likelihood_terms_fn) from make_likelihood_fns
        K           : number of parameters
        N           : number of trials
        verbose     : print optimisation result

    Returns:
        params : (K,) numpy array of best-fit constant parameters
    """
    from scipy.optimize import minimize as _minimize

    if dtype is None:
        dtype = _map_module._JAX_DTYPE
    log_lik_fn = log_lik_fns[0]
    dat_jax = _to_jax(dat, dtype=dtype)

    # Define the objective once so JAX compiles it only on the first call and
    # reuses the cached compiled version for every subsequent iteration.
    # (Defining a new lambda inside the loop would force re-tracing each call.)
    @jax.jit
    def _ll_of_params(params):
        return log_lik_fn(jnp.tile(params[:, jnp.newaxis], (1, N)), dat_jax)

    _ll_and_grad = jax.jit(jax.value_and_grad(_ll_of_params))

    def neg_ll_and_grad(params_flat):
        params = jnp.array(params_flat, dtype=dtype)
        val, grad = _ll_and_grad(params)
        val_f = float(-val)
        grad_f = np.array(-grad, dtype=np.float64)
        if not np.isfinite(val_f):
            val_f = 1e20
        if not np.all(np.isfinite(grad_f)):
            grad_f = np.zeros(K, dtype=np.float64)
        return val_f, grad_f

    # Try two starting points: small positive (avoids z=0 sentinel region in
    # constrained models like the race model) and a unit vector.
    starts = [np.full(K, 0.5), np.full(K, 1.0)]
    best_result = None
    for x0 in starts:
        r = _minimize(
            neg_ll_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 200, 'ftol': 1e-8},
        )
        if best_result is None or r.fun < best_result.fun:
            best_result = r
    if verbose:
        print(f'  Warm-start log-likelihood: {-best_result.fun:.4f}  ({best_result.message})')
    return np.array(best_result.x)


def _check_E0_validity(E0, dat, log_lik_fns, K, N):
    """Warn if E0 causes many sentinel (invalid) log-likelihoods.

    The race model (and similar) returns a large negative sentinel (-1e12)
    when parameter constraints are violated (e.g. z <= 0, T <= 0).  Even a
    handful of sentinel trials will corrupt the MAP objective.  This check
    computes the total log-likelihood at E0 and warns when it is implausibly
    low.

    Args:
        E0          : (K, N) initial parameter matrix
        dat         : normalised data dict
        log_lik_fns : (log_likelihood_fn, likelihood_terms_fn)
        K, N        : parameter count and trial count
    """
    log_lik_fn = log_lik_fns[0]
    E0_jax = jnp.asarray(E0, dtype=jnp.float64)
    dat_jax = _to_jax(dat)

    total_ll = float(log_lik_fn(E0_jax, dat_jax))

    # A realistic per-trial log-likelihood for a well-specified model is O(-1).
    # Sentinel values are -1e12, so even one sentinel completely dominates.
    if not np.isfinite(total_ll) or total_ll < -N * 100:
        n_sentinel_est = int(min(N, max(1, round(-total_ll / 1e12))))
        warnings.warn(
            f"The provided E0 yields a very low log-likelihood ({total_ll:.3e}), "
            f"suggesting ~{n_sentinel_est} trial(s) have invalid parameter values "
            "(e.g. z<=0, T<=0, or other model-specific constraints). "
            "Consider passing E0=None to use the automatic warm-start, or adjusting "
            "your initial parameter values.",
            UserWarning,
            stacklevel=4,
        )

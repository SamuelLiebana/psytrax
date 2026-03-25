import os
import numpy as np
from datetime import datetime

from psytrax._likelihood import make_likelihood_fns
from psytrax._hyper_opt import hyperOpt
from psytrax._helper.helperFunctions import trim
from psytrax._device import setup_device


def fit(data, log_lik_trial, n_params,
        param_names=None,
        hyper=None,
        session_boundaries=False,
        E0=None,
        n_trials=None,
        hess_calc='weights',
        device='auto',
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
        Initial hyperparameters.  Must contain 'sigma' (array of length K).
        May also contain 'sigInit' and 'sigDay'.
        Defaults to sigma=2^{-3} for all params.
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
    # Device selection
    # ------------------------------------------------------------------
    setup_device(device, verbose=True)

    # ------------------------------------------------------------------
    # Load / normalise data
    # ------------------------------------------------------------------
    if isinstance(data, str):
        raw = np.load(data, allow_pickle=True).item()
    elif isinstance(data, dict):
        raw = data
    else:
        raise TypeError(f"data must be a dict or file path, not {type(data)}")

    dat = _normalise_dat(raw)

    N_total = len(dat['r'])
    N = n_trials if n_trials is not None and n_trials < N_total else N_total

    if N != N_total:
        dat = trim(dat, END=N)

    # ------------------------------------------------------------------
    # Hyperparameters
    # ------------------------------------------------------------------
    K = n_params

    if hyper is None:
        hyper = {
            'sigma': np.full(K, 2 ** -3),
            'sigInit': np.full(K, 2 ** 4),
            'sigDay': None,
        }

    opt_list = ['sigma']

    if session_boundaries:
        if 'dayLength' not in dat or dat['dayLength'] is None:
            raise ValueError("session_boundaries=True requires 'session_lengths' in data")
        if hyper.get('sigDay') is None:
            hyper['sigDay'] = np.full(K, 2 ** -2)
        opt_list = ['sigma', 'sigDay']

    # ------------------------------------------------------------------
    # Build batched likelihood functions
    # ------------------------------------------------------------------
    log_lik_fns = make_likelihood_fns(log_lik_trial)

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
        show_progress=True,
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

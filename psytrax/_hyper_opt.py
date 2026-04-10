import numpy as np
from scipy.optimize import minimize
from scipy.sparse import linalg
from tqdm.auto import tqdm

from psytrax._map import getPosteriorTerms
from psytrax._helper.invBlkTriDiag import getCredibleInterval
from psytrax._helper.jacHessCheck import compHess
from psytrax._helper.helperFunctions import DT_X_D, make_invSigma, sparse_logdet


def hyperOpt(dat, hyper, n_params, log_lik_fns, optList, E0=None,
             method=None, showOpt=0, jump=2, hess_calc='weights', show_progress=True,
             map_tol=1e-6):
    """Optimise hyperparameters and return MAP weights.

    Uses the decoupled Laplace approximation to find the hyperparameter values
    (process noise sigmas) that maximise the marginal likelihood of the data.

    Args:
        dat         : data dict
        hyper       : dict of hyperparameters with initial values
        n_params    : int, number of parameters per trial (K)
        log_lik_fns : (log_likelihood_fn, likelihood_terms_fn)
        optList     : list of hyper keys to optimise (e.g. ['sigma', 'sigDay'])
        E0          : initial parameter array shape (K, N); defaults to 0.01
        method      : None | '_constant' | '_days'
        showOpt     : 0 silent | 1 verbose
        jump        : patience — how many consecutive worse steps before stopping
        hess_calc   : 'weights' | 'hyper' | 'All' | None
        map_tol     : convergence tolerance for each inner MAP solve

    Returns:
        best_hyper  : dict, optimised hyperparameters
        best_logEvd : float
        best_eMode  : array, MAP parameter estimates (K*N,)
        hess_info   : dict with credible intervals / Hessian info
    """
    from psytrax._jax_map import getMAP_jax as map_fn

    K = n_params
    N = len(dat['r'])

    for val in optList:
        if val not in hyper or hyper[val] is None:
            raise Exception(f"Cannot optimise '{val}': not in hyper or is None")

    if E0 is None:
        E0 = 0.01 * np.ones((K, N))

    current_hyper = hyper.copy()
    best_logEvd = None
    current_jump = jump
    first_iter = True

    pbar = tqdm(desc='Fitting', unit='cycle') if show_progress else None

    opt_keywords = {
        'dat': dat,
        'hyper': hyper,
        'n_params': n_params,
        'log_lik_fns': log_lik_fns,
        'optList': optList,
        'method': method,
    }

    while True:
        current_E0 = E0 if first_iter else llstruct['eMode']  # noqa: F821
        first_iter = False

        Hess, logEvd, llstruct = map_fn(
            dat, current_hyper, K, log_lik_fns,
            E0=current_E0, method=method, showOpt=0, pbar=pbar, map_tol=map_tol,
        )
        if pbar is not None:
            pbar.update(1)
            pbar.set_postfix({
                'log_evd': f'{logEvd:.3f}',
                'best': f'{best_logEvd:.3f}' if best_logEvd is not None else '—',
            })

        if best_logEvd is None or logEvd >= best_logEvd:
            current_jump = jump
            best_hyper = current_hyper.copy()
            best_logEvd = logEvd
            best_Hess = Hess
            best_eMode = llstruct['eMode']
            best_llstruct = llstruct.copy()
        else:
            current_jump -= 1
            for val in optList:
                current_hyper[val] = (current_hyper[val] + best_hyper[val]) / 2

        if showOpt:
            print(f'\nLog-evidence: {np.round(logEvd, 5)}')
            for val in optList:
                print(val, np.round(np.log2(current_hyper[val]), 4))

        if not current_jump:
            eMode = best_llstruct['eMode']
            H = best_llstruct['lT']['ddlogli']['H']
            ddlogprior = best_llstruct['pT']['ddlogprior']
            LL_v = -(H + ddlogprior) @ eMode
            opt_keywords.update({
                'hyper': best_hyper,
                'LL_terms': best_llstruct['lT']['ddlogli'],
                'LL_v': LL_v,
                'eMode': eMode,
            })
            if showOpt:
                print('Stopping: no improvement in evidence.')
            break

        # --- Decouple prior/likelihood for BFGS ---
        eMode = llstruct['eMode']
        H = llstruct['lT']['ddlogli']['H']
        ddlogprior = llstruct['pT']['ddlogprior']
        LL_v = -(H + ddlogprior) @ eMode
        opt_keywords.update({
            'hyper': current_hyper,
            'LL_terms': llstruct['lT']['ddlogli'],
            'LL_v': LL_v,
            'eMode': eMode,
        })

        optVals = _pack_optvals(current_hyper, optList, K)

        if showOpt:
            print('\nOptimising hyperparameters...')
            opts = {'disp': True, 'maxiter': 15}
            callback = lambda x: print(x)
        else:
            opts = {'disp': False, 'maxiter': 15}
            callback = None

        # L-BFGS-B with bounds keeps the line search away from degenerate sigma
        # values (log2 in [-15, 5] ↔ sigma in [~3e-5, 32]) that make the
        # log-evidence Hessian singular.
        n_hyper_vals = len(optVals)
        bounds = [(-15, 5)] * n_hyper_vals
        result = minimize(
            _hyperOpt_lossfun,
            optVals,
            args=opt_keywords,
            method='L-BFGS-B',
            bounds=bounds,
            options=opts,
            callback=callback,
        )

        diff = np.linalg.norm((optVals - np.array(result.x)) / np.maximum(np.abs(optVals), 1e-8))
        if showOpt:
            print(f'Recovered hypers: {np.array(result.x)}')
            print(f'Log-evidence:     {np.round(-result.fun, 5)}')
            print(f'Hyper change:     {np.round(diff, 4)}')

        if diff > 0.1:
            _unpack_optvals(result.x, current_hyper, optList, K)
        else:
            break

    if pbar is not None:
        pbar.close()

    # --- Credible intervals ---
    hess_info = {'hess': best_Hess}
    if hess_calc in ['weights', 'All']:
        hess_info['W_std'] = getCredibleInterval(best_Hess)
    if hess_calc in ['hyper', 'All']:
        optVals = _pack_optvals(best_hyper, optList, K)
        num_hess, _ = compHess(
            fun=_hyperOpt_lossfun,
            x0=np.array(optVals),
            dx=0.01,
            kwargs={'keywords': opt_keywords},
        )
        hess_info['hyp_std'] = np.sqrt(np.diag(np.linalg.inv(num_hess)))

    return best_hyper, best_logEvd, best_eMode, hess_info


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pack_optvals(hyper, optList, K):
    """Flatten hyperparameters into a log2-scaled vector for optimisation."""
    vals = []
    for val in optList:
        if np.isscalar(hyper[val]):
            vals.append(np.log2(hyper[val]))
        else:
            vals.extend(np.log2(hyper[val]).tolist())
    return vals


def _unpack_optvals(result_x, hyper, optList, K):
    """Write optimised log2 values back into the hyper dict in-place."""
    count = 0
    for val in optList:
        if np.isscalar(hyper[val]):
            hyper[val] = float(2 ** result_x[count])
            count += 1
        else:
            hyper[val] = 2 ** np.array(result_x[count:count + K])
            count += K


def _hyperOpt_lossfun(optVals, keywords):
    """Negative log-evidence for a given set of hyperparameter values.

    Uses the decoupled Laplace approximation: re-estimates w_MAP cheaply by
    solving a linear system rather than re-running the full MAP optimisation.
    Returns a large positive sentinel (1e20) when numerical issues arise so
    that the outer L-BFGS-B optimiser backs off to a safer region.
    """
    N = keywords['dat']['r'].shape[0]
    K = keywords['LL_terms']['K']
    method = keywords['method']
    dat = keywords['dat']
    log_lik_fns = keywords['log_lik_fns']

    hyper = keywords['hyper'].copy()
    _unpack_optvals(optVals, hyper, keywords['optList'], K)

    if method is None:
        w_N = N
        days_arr = np.cumsum(dat['dayLength'], dtype=int)[:-1]
        missing_trials = dat.get('missing_trials')
    elif method == '_constant':
        w_N = 1
        days_arr = np.array([], dtype=int)
        missing_trials = None
    elif method == '_days':
        w_N = len(dat['dayLength'])
        days_arr = np.arange(1, w_N, dtype=int)
        missing_trials = None
    else:
        raise Exception(f"method '{method}' not supported")

    try:
        invSigma = make_invSigma(hyper, days_arr, missing_trials, w_N, K)
        ddlogprior = -DT_X_D(invSigma, K)
        H = keywords['LL_terms']['H']
        LL_v = keywords['LL_v']
        Lambda = -H - ddlogprior
        E_flat = linalg.spsolve(Lambda, LL_v)

        pT, lT = getPosteriorTerms(E_flat, dat, hyper, log_lik_fns, method)
        logterm_post = 0.5 * sparse_logdet(-ddlogprior - lT['ddlogli']['H'])
        evd = lT['logli'] + pT['logprior'] - logterm_post
        if not np.isfinite(evd):
            return 1e20
        return -evd
    except (RuntimeError, np.linalg.LinAlgError, ValueError):
        return 1e20

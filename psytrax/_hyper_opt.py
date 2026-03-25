import numpy as np
from scipy.sparse import linalg
from tqdm.auto import tqdm

from psytrax._map import getMAP, getPosteriorTerms
from psytrax._helper.invBlkTriDiag import getCredibleInterval, invBlkTriDiag
from psytrax._helper.jacHessCheck import compHess
from psytrax._helper.helperFunctions import DT_X_D, make_invSigma, sparse_logdet


def hyperOpt(dat, hyper, n_params, log_lik_fns, optList, E0=None,
             method=None, showOpt=0, jump=2, hess_calc='weights', show_progress=True,
             map_tol=1e-6):
    """Optimise hyperparameters and return MAP weights.

    Uses the decoupled Laplace approximation: after each MAP solve, hyperparameters
    are updated via a closed-form EM M-step rather than iterative BFGS.

    For the Gaussian random-walk prior the M-step is:
        sigma_k^2 = mean_t [ (w_k[t] - w_k[t-1])^2 + Var_post(w_k[t] - w_k[t-1]) ]
    where the posterior increment variance comes from the block-tridiagonal inverse
    of the posterior precision matrix.

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

    # opt_keywords is used by _hyperOpt_lossfun (needed for hess_calc == 'hyper')
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

        Hess, logEvd, llstruct = getMAP(
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
            break

        # --- EM M-step: closed-form hyperparameter update ---
        old_optVals = np.array(_pack_optvals(current_hyper, optList, K), dtype=float)
        _em_update_hyper(current_hyper, Hess, llstruct['eMode'], optList, K, dat)
        new_optVals = np.array(_pack_optvals(current_hyper, optList, K), dtype=float)

        diff = np.linalg.norm(
            (old_optVals - new_optVals) / np.abs(old_optVals).clip(1e-8)
        )
        if showOpt:
            print(f'EM hyper change: {diff:.4f}')
            for val in optList:
                print(f'  {val}: {np.round(np.log2(current_hyper[val]), 4)}')

        if diff <= 0.1:
            break

    if pbar is not None:
        pbar.close()

    # Build opt_keywords from best result (used by hess_calc == 'hyper')
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
# EM M-step helpers
# ---------------------------------------------------------------------------

def _get_posterior_cov_blocks(Hess):
    """Return diagonal and below-diagonal blocks of the posterior covariance.

    The posterior precision Λ_MAP is reordered from parameter-major to
    trial-major (K×K blocks per trial) before inversion, matching the
    convention of invBlkTriDiag.

    Returns:
        MinvBlocks          : (K, K, N)   — diagonal K×K blocks
        MinvBelowDiagBlocks : (K, K, N-1) — MinvBelowDiagBlocks[:,:,t] = Σ[t+1, t] block
    """
    center = -(Hess['ddlogprior'] + Hess['H'])
    K = Hess['K']
    N = int(Hess['ddlogprior'].shape[0] / K)
    # Reorder: parameter-major (NK) → trial-major (N blocks of size K)
    ii = (np.reshape(np.arange(K * N), (N, -1), order='F').T).flatten(order='F')
    M = center[ii][:, ii]
    _, MinvBlocks, MinvBelowDiagBlocks = invBlkTriDiag(M, K)
    return MinvBlocks, MinvBelowDiagBlocks


def _em_sigma(w, K, transitions, MinvBlocks, MinvBelowDiagBlocks, is_scalar, current_val):
    """EM M-step for a sigma value over the given set of transition indices t.

    sigma^2 = mean_t [ (w_k[t] - w_k[t-1])^2 + Var_post(w_k[t] - w_k[t-1]) ]

    where Var_post(Δw_k[t]) = Σ_kk[t,t] + Σ_kk[t-1,t-1] - 2*Σ_kk[t,t-1]
    and Σ_kk[t,t-1] = MinvBelowDiagBlocks[k, k, t-1].
    """
    if not transitions:
        return current_val
    transitions = sorted(transitions)

    if is_scalar:
        total, n = 0.0, K * len(transitions)
        for k in range(K):
            incr_sq = np.array([(w[k, t] - w[k, t - 1]) ** 2 for t in transitions])
            post_var = np.array([
                MinvBlocks[k, k, t] + MinvBlocks[k, k, t - 1]
                - 2.0 * MinvBelowDiagBlocks[k, k, t - 1]
                for t in transitions
            ])
            total += float(np.sum(np.maximum(incr_sq + post_var, 0.0)))
        return float(np.sqrt(max(total / n, 1e-24)))
    else:
        new_vals = np.zeros(K)
        for k in range(K):
            incr_sq = np.array([(w[k, t] - w[k, t - 1]) ** 2 for t in transitions])
            post_var = np.array([
                MinvBlocks[k, k, t] + MinvBlocks[k, k, t - 1]
                - 2.0 * MinvBelowDiagBlocks[k, k, t - 1]
                for t in transitions
            ])
            new_vals[k] = np.sqrt(max(float(np.mean(np.maximum(incr_sq + post_var, 0.0))), 1e-24))
        return new_vals


def _em_update_hyper(hyper, Hess, eMode, optList, K, dat):
    """EM M-step: update each optimised hyperparameter to its closed-form optimum.

    For Gaussian random-walk priors the marginal likelihood gradient w.r.t.
    log sigma_k is zero exactly when sigma_k equals the RMS of (MAP increment +
    posterior increment variance).  Computing this directly replaces iterative BFGS.
    """
    N = int(len(eMode) / K)
    w = eMode.reshape(K, N)  # (K, N) parameter-major

    day_lengths = dat.get('dayLength', np.array([], dtype=int))
    days = set(np.cumsum(day_lengths, dtype=int)[:-1].tolist()) if len(day_lengths) else set()

    MinvBlocks, MinvBelowDiagBlocks = _get_posterior_cov_blocks(Hess)

    for hyp_name in optList:
        is_scalar = np.isscalar(hyper[hyp_name])

        if hyp_name == 'sigma':
            interior = {t for t in range(1, N) if t not in days}
            hyper['sigma'] = _em_sigma(
                w, K, interior, MinvBlocks, MinvBelowDiagBlocks, is_scalar, hyper['sigma']
            )

        elif hyp_name == 'sigInit':
            # Prior: w_k[0] ~ N(0, sigInit_k^2)
            # M-step: sigInit_k^2 = w_k[0]^2 + Σ_kk[0,0]
            if is_scalar:
                hyper['sigInit'] = float(np.sqrt(
                    max(np.mean([w[k, 0] ** 2 + MinvBlocks[k, k, 0] for k in range(K)]), 1e-24)
                ))
            else:
                hyper['sigInit'] = np.array([
                    np.sqrt(max(w[k, 0] ** 2 + MinvBlocks[k, k, 0], 1e-24))
                    for k in range(K)
                ])

        elif hyp_name == 'sigDay':
            if not days:
                continue
            hyper['sigDay'] = _em_sigma(
                w, K, days, MinvBlocks, MinvBelowDiagBlocks, is_scalar, hyper['sigDay']
            )


# ---------------------------------------------------------------------------
# Helpers for hess_calc == 'hyper' (numerical Hessian of evidence w.r.t. hypers)
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
    Retained for computing the numerical Hessian of evidence w.r.t. hypers
    (hess_calc == 'hyper').
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
        days = np.cumsum(dat['dayLength'], dtype=int)[:-1]
        missing_trials = dat.get('missing_trials')
    elif method == '_constant':
        w_N = 1
        days = np.array([], dtype=int)
        missing_trials = None
    elif method == '_days':
        w_N = len(dat['dayLength'])
        days = np.arange(1, w_N, dtype=int)
        missing_trials = None
    else:
        raise Exception(f"method '{method}' not supported")

    invSigma = make_invSigma(hyper, days, missing_trials, w_N, K)
    ddlogprior = -DT_X_D(invSigma, K)

    H = keywords['LL_terms']['H']
    LL_v = keywords['LL_v']
    Lambda_MAP_inv = -H - ddlogprior
    E_flat = linalg.spsolve(Lambda_MAP_inv, LL_v)

    pT, lT = getPosteriorTerms(E_flat, dat, hyper, log_lik_fns, method)

    logterm_post = 0.5 * sparse_logdet(-ddlogprior - lT['ddlogli']['H'])
    evd = lT['logli'] + pT['logprior'] - logterm_post
    return -evd

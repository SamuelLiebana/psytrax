import numpy as np
from scipy.optimize import minimize
from scipy.sparse import linalg
from tqdm.auto import tqdm

from psytrax._map import getPosteriorTerms
from psytrax._helper.invBlkTriDiag import getCredibleInterval
from psytrax._helper.jacHessCheck import compHess
from psytrax._helper.helperFunctions import (
    DT_X_D, make_invSigma, sparse_logdet,
    build_v_mean_flat, compute_invC_u, correct_logprior_for_learning_rule,
)


def hyperOpt(dat, hyper, n_params, log_lik_fns, optList, E0=None,
             method=None, showOpt=0, jump=2, hess_calc='weights', show_progress=True,
             map_tol=1e-6, execution_plan=None, status_callback=None,
             learning_rule=None, model_hyper=None, model_hyper_optList=None):
    """Optimise hyperparameters and return MAP weights.

    Uses the decoupled Laplace approximation to find the hyperparameter values
    (process noise sigmas, and optionally learning rates) that maximise the
    marginal likelihood of the data.

    Args:
        dat         : data dict
        hyper       : dict of hyperparameters with initial values
        n_params    : int, number of parameters per trial (K)
        log_lik_fns : (log_likelihood_fn, likelihood_terms_fn)
        optList     : list of hyper keys to optimise (e.g. ['sigma', 'sigDay', 'alpha'])
        E0          : initial parameter array shape (K, N); defaults to 0.01
        method      : None | '_constant' | '_days'
        showOpt     : 0 silent | 1 verbose
        jump        : patience — how many consecutive worse steps before stopping
        hess_calc   : 'weights' | 'hyper' | 'All' | None
        map_tol     : convergence tolerance for each inner MAP solve
        learning_rule : callable or None.  When provided, the random-walk prior
                        gets a non-zero mean v_t = diag(α) · learning_rule(w_t, data_t).

    Returns:
        best_hyper  : dict, optimised hyperparameters
        best_logEvd : float
        best_eMode  : array, MAP parameter estimates (K*N,)
        hess_info   : dict with credible intervals / Hessian info
    """
    from psytrax._jax_map import getMAP_jax as map_fn

    K = n_params
    N = len(dat['r'])
    has_lr = learning_rule is not None

    for val in optList:
        if val not in hyper or hyper[val] is None:
            raise Exception(f"Cannot optimise '{val}': not in hyper or is None")

    # ---- model_hyper handling ----
    if model_hyper is None:
        model_hyper = {}
    if model_hyper_optList is None:
        model_hyper_optList = list(model_hyper.keys())
    for key in model_hyper_optList:
        if key not in model_hyper or model_hyper[key] is None:
            raise Exception(f"Cannot optimise model_hyper '{key}': not in model_hyper or is None")

    if E0 is None:
        E0 = 0.01 * np.ones((K, N))

    current_hyper = hyper.copy()
    current_model_hyper = dict(model_hyper)
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
        'learning_rule': learning_rule,
        'model_hyper': current_model_hyper,
        'model_hyper_optList': list(model_hyper_optList),
    }

    while True:
        current_E0 = E0 if first_iter else llstruct['eMode']  # noqa: F821
        first_iter = False

        try:
            _emit_status(status_callback, "Re-running MAP at current hyperparameters…", stage="map")
            Hess, logEvd, llstruct = map_fn(
                dat, current_hyper, K, log_lik_fns,
                E0=current_E0, method=method, showOpt=0, pbar=pbar, map_tol=map_tol,
                execution_plan=execution_plan,
                status_callback=status_callback,
                learning_rule=learning_rule,
                model_hyper=current_model_hyper,
            )
        except RuntimeError as exc:
            msg = str(exc).lower()
            if ("invalid parameter region" not in msg and
                    "non-finite log-evidence" not in msg and
                    "sentinel" not in msg and
                    "singular" not in msg and
                    "hessian" not in msg):
                raise
            if best_logEvd is None:
                raise
            current_jump -= 1
            for val in optList:
                current_hyper[val] = (current_hyper[val] + best_hyper[val]) / 2
            for key in model_hyper_optList:
                current_model_hyper[key] = (
                    current_model_hyper[key] + best_model_hyper[key]
                ) / 2
            _emit_status(status_callback, f"MAP step invalid, backing off hyperparameters: {exc}",
                         stage="retry")
            if showOpt:
                print(f'\nMAP step became invalid at current hyperparameters, backing off: {exc}')
            if not current_jump:
                eMode = best_llstruct['eMode']
                H = best_llstruct['lT']['ddlogli']['H']
                ddlogprior = best_llstruct['pT']['ddlogprior']
                LL_v = _compute_LL_v_pure(
                    eMode, H, ddlogprior, best_llstruct, best_hyper,
                    dat, has_lr, K, N,
                )
                opt_keywords.update({
                    'hyper': best_hyper,
                    'LL_terms': best_llstruct['lT']['ddlogli'],
                    'LL_v': LL_v,
                    'eMode': eMode,
                    'lr_hat': best_llstruct.get('lr_hat'),
                })
                break
            continue
        if pbar is not None:
            pbar.update(1)
            pbar.set_postfix({
                'log_evd': f'{logEvd:.3f}',
                'best': f'{best_logEvd:.3f}' if best_logEvd is not None else '—',
            })

        if best_logEvd is None or logEvd >= best_logEvd:
            current_jump = jump
            best_hyper = current_hyper.copy()
            best_model_hyper = dict(current_model_hyper)
            best_logEvd = logEvd
            best_Hess = Hess
            best_eMode = llstruct['eMode']
            best_llstruct = llstruct.copy()
        else:
            current_jump -= 1
            for val in optList:
                current_hyper[val] = (current_hyper[val] + best_hyper[val]) / 2
            for key in model_hyper_optList:
                current_model_hyper[key] = (
                    current_model_hyper[key] + best_model_hyper[key]
                ) / 2

        if showOpt:
            print(f'\nLog-evidence: {np.round(logEvd, 5)}')
            for val in optList:
                print(val, np.round(np.log2(current_hyper[val]), 4))
            for key in model_hyper_optList:
                print(f'model_hyper[{key}]', np.round(np.log2(current_model_hyper[key]), 4))
        _emit_status(status_callback, f"Evaluated current hyperparameters (log-evidence {logEvd:.3f}).",
                     stage="cycle", log_evidence=float(logEvd))

        if not current_jump:
            eMode = best_llstruct['eMode']
            H = best_llstruct['lT']['ddlogli']['H']
            ddlogprior = best_llstruct['pT']['ddlogprior']
            LL_v = _compute_LL_v_pure(
                eMode, H, ddlogprior, best_llstruct, best_hyper,
                dat, has_lr, K, N,
            )
            opt_keywords.update({
                'hyper': best_hyper,
                'model_hyper': best_model_hyper,
                'LL_terms': best_llstruct['lT']['ddlogli'],
                'LL_v': LL_v,
                'eMode': eMode,
                'lr_hat': best_llstruct.get('lr_hat'),
            })
            if showOpt:
                print('Stopping: no improvement in evidence.')
            break

        # --- Decouple prior/likelihood for BFGS ---
        eMode = llstruct['eMode']
        H = llstruct['lT']['ddlogli']['H']
        ddlogprior = llstruct['pT']['ddlogprior']
        LL_v = _compute_LL_v_pure(
            eMode, H, ddlogprior, llstruct, current_hyper,
            dat, has_lr, K, N,
        )
        opt_keywords.update({
            'hyper': current_hyper,
            'model_hyper': current_model_hyper,
            'LL_terms': llstruct['lT']['ddlogli'],
            'LL_v': LL_v,
            'eMode': eMode,
            'lr_hat': llstruct.get('lr_hat'),
        })

        optVals = _pack_optvals(current_hyper, optList, K,
                                current_model_hyper, model_hyper_optList)

        if showOpt:
            print('\nOptimising hyperparameters...')
            opts = {'disp': True, 'maxiter': 15}
            callback = lambda x: print(x)
        else:
            opts = {'disp': False, 'maxiter': 15}
            callback = None

        _emit_status(status_callback, "Updating hyperparameters for the next cycle…", stage="hyper")
        # L-BFGS-B with bounds keeps the line search away from degenerate sigma
        # values (log2 in [-15, 5] ↔ sigma in [~3e-5, 32]) that make the
        # log-evidence Hessian singular.  The same bounds are applied to
        # model_hyper entries (positivity is enforced via the log2 mapping).
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
        _emit_status(status_callback, f"Hyperparameter update size {diff:.4f}.", stage="hyper")

        if diff > 0.1:
            _unpack_optvals(result.x, current_hyper, optList, K,
                            current_model_hyper, model_hyper_optList)
        else:
            break

    if pbar is not None:
        pbar.close()

    # --- Credible intervals ---
    hess_info = {'hess': best_Hess}
    if hess_calc in ['weights', 'All']:
        hess_info['W_std'] = getCredibleInterval(best_Hess)
    if hess_calc in ['hyper', 'All']:
        optVals = _pack_optvals(best_hyper, optList, K,
                                best_model_hyper, model_hyper_optList)
        num_hess, _ = compHess(
            fun=_hyperOpt_lossfun,
            x0=np.array(optVals),
            dx=0.01,
            kwargs={'keywords': opt_keywords},
        )
        hess_info['hyp_std'] = np.sqrt(np.diag(np.linalg.inv(num_hess)))

    return best_hyper, best_logEvd, best_eMode, hess_info, best_model_hyper


def _emit_status(callback, message, stage=None, **extra):
    if callback is None:
        return
    payload = {"message": message}
    if stage is not None:
        payload["stage"] = stage
    payload.update(extra)
    callback(payload)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_LL_v_pure(eMode, H, ddlogprior, llstruct, hyper, dat, has_lr, K, N):
    """Compute the "pure likelihood" vector for the decoupled Laplace step.

    Without a learning rule (zero prior mean):
        LL_v = -(H + ddlogprior) @ eMode   (= dlogli - H @ eMode)

    With a learning rule (non-zero prior mean u):
        LL_v_pure = -(H + ddlogprior) @ eMode - invC @ u
                  = dlogli - H @ eMode

    The subtraction of invC @ u ensures that LL_v encodes only the likelihood
    information.  The prior mean contribution is re-added in the decoupled loss
    function for the new hyperparameters.
    """
    LL_v = -(H + ddlogprior) @ eMode

    if has_lr and llstruct.get('lr_hat') is not None:
        alpha = hyper.get('alpha')
        if alpha is not None:
            dat.setdefault('dayLength', np.array([], dtype=int))
            dat.setdefault('missing_trials', None)
            day_lengths = dat.get('dayLength', np.array([], dtype=int))
            if len(day_lengths) > 0:
                days_arr = np.cumsum(day_lengths, dtype=int)[:-1]
            else:
                days_arr = np.array([], dtype=int)
            invSigma = make_invSigma(hyper, days_arr, dat.get('missing_trials'), N, K)
            v_mean_flat = build_v_mean_flat(llstruct['lr_hat'], alpha, K, N)
            invC_u = compute_invC_u(v_mean_flat, invSigma, K, N)
            LL_v = LL_v - invC_u

    return LL_v


def _pack_optvals(hyper, optList, K, model_hyper=None, model_hyper_optList=None):
    """Flatten hyperparameters into a log2-scaled vector for optimisation.

    Layout: ``[hyper entries…, model_hyper entries…]``.  Each ``hyper`` entry
    contributes either 1 (scalar) or K (length-K vector) values; each
    ``model_hyper`` entry currently contributes a single scalar (vector
    model_hyper is not yet supported — extend the helper if needed).
    """
    vals = []
    for val in optList:
        if np.isscalar(hyper[val]):
            vals.append(np.log2(hyper[val]))
        else:
            vals.extend(np.log2(hyper[val]).tolist())
    if model_hyper is not None and model_hyper_optList:
        for key in model_hyper_optList:
            v = model_hyper[key]
            if not np.isscalar(v):
                raise NotImplementedError(
                    f"model_hyper['{key}'] must be scalar; vector model_hyper is not yet supported"
                )
            vals.append(float(np.log2(v)))
    return vals


def _unpack_optvals(result_x, hyper, optList, K,
                    model_hyper=None, model_hyper_optList=None):
    """Write optimised log2 values back into hyper / model_hyper in-place."""
    count = 0
    for val in optList:
        if np.isscalar(hyper[val]):
            hyper[val] = float(2 ** result_x[count])
            count += 1
        else:
            hyper[val] = 2 ** np.array(result_x[count:count + K])
            count += K
    if model_hyper is not None and model_hyper_optList:
        for key in model_hyper_optList:
            model_hyper[key] = float(2 ** result_x[count])
            count += 1


def _hyperOpt_lossfun(optVals, keywords):
    """Negative log-evidence for a given set of hyperparameter values.

    Uses the decoupled Laplace approximation: re-estimates w_MAP cheaply by
    solving a linear system rather than re-running the full MAP optimisation.
    Returns a large positive sentinel (1e20) when numerical issues arise so
    that the outer L-BFGS-B optimiser backs off to a safer region.

    When a learning rule is present, the prior mean is non-zero and the RHS
    of the decoupled system includes an additional  invC_new @ u_new  term.
    The log-prior in the evidence also gets a correction for the mean shift.

    Model-level hyperparameters (``keywords['model_hyper']``) re-enter the
    likelihood at the new value via ``getPosteriorTerms``; the linear system
    above is solved with the previous H, so this is a quasi-Newton step that
    converges over multiple outer cycles.
    """
    N = keywords['dat']['r'].shape[0]
    K = keywords['LL_terms']['K']
    method = keywords['method']
    dat = keywords['dat']
    log_lik_fns = keywords['log_lik_fns']
    learning_rule = keywords.get('learning_rule')
    lr_hat = keywords.get('lr_hat')   # (K, N-1) or None
    has_lr = learning_rule is not None and lr_hat is not None

    hyper = keywords['hyper'].copy()
    model_hyper = dict(keywords.get('model_hyper') or {})
    model_hyper_optList = keywords.get('model_hyper_optList') or []
    _unpack_optvals(optVals, hyper, keywords['optList'], K,
                    model_hyper, model_hyper_optList)

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
        LL_v = keywords['LL_v']          # "pure likelihood" vector
        Lambda = -H - ddlogprior

        # Build the RHS of the decoupled system
        if has_lr:
            alpha_new = hyper.get('alpha')
            if alpha_new is not None:
                v_mean_flat = build_v_mean_flat(lr_hat, alpha_new, K, N)
                invC_u_new = compute_invC_u(v_mean_flat, invSigma, K, N)
                rhs = LL_v + invC_u_new
            else:
                rhs = LL_v
                v_mean_flat = None
        else:
            rhs = LL_v
            v_mean_flat = None

        E_flat = linalg.spsolve(Lambda, rhs)

        pT, lT = getPosteriorTerms(E_flat, dat, hyper, log_lik_fns, method,
                                   model_hyper=model_hyper)

        # Correct the log-prior for the learning-rule mean shift
        if has_lr and v_mean_flat is not None:
            pT['logprior'] = correct_logprior_for_learning_rule(
                pT['logprior'], E_flat, v_mean_flat, invSigma, K, N,
            )

        logterm_post = 0.5 * sparse_logdet(-ddlogprior - lT['ddlogli']['H'])
        evd = lT['logli'] + pT['logprior'] - logterm_post
        if not np.isfinite(evd):
            return 1e20
        return -evd
    except (RuntimeError, np.linalg.LinAlgError, ValueError):
        return 1e20

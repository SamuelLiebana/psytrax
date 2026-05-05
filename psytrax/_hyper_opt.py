import os
import numpy as np
from scipy.optimize import minimize, OptimizeResult
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
        _emit_status(
            status_callback,
            f"Evaluated current hyperparameters (log-evidence {logEvd:.3f}).",
            stage="cycle",
            log_evidence=float(logEvd),
        )

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
        result = _minimize_hyperparameters(
            optVals,
            opt_keywords,
            bounds,
            opts,
            callback=callback,
            showOpt=showOpt,
            status_callback=status_callback,
        )

        result = _limit_hyperparameter_step(result, optVals, opt_keywords, showOpt=showOpt)
        diff = _hyperparameter_update_size(optVals, result.x)
        if showOpt:
            method_used = getattr(result, 'psytrax_method', result.get('method', 'unknown'))
            print(f'Hyper optimiser: {method_used}')
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
        try:
            num_hess, _ = compHess(
                fun=_hyperOpt_lossfun,
                x0=np.array(optVals),
                dx=0.01,
                kwargs={'keywords': opt_keywords},
            )
            inv_hess = np.linalg.inv(num_hess)
            diag = np.diag(inv_hess)
            # Negative or non-finite diag entries indicate a degenerate Hessian
            # (e.g. a hyperparameter sat at its bound).  Mask those rather than
            # producing nonsense sqrt-of-negative numbers.
            with np.errstate(invalid='ignore'):
                hyp_std = np.where(np.isfinite(diag) & (diag > 0),
                                   np.sqrt(np.abs(diag)),
                                   np.nan)
            hess_info['hyp_std'] = hyp_std
            hess_info['hyp_optList'] = list(optList)
            hess_info['hyp_model_hyper_optList'] = list(model_hyper_optList)
        except (np.linalg.LinAlgError, RuntimeError, ValueError) as exc:
            # Singular numerical Hessian — happens when a hyperparameter sits
            # at a bound or two hypers are highly confounded.  Skip the CIs
            # rather than crash the whole fit.
            if showOpt:
                print(f'WARNING: hyperparameter Hessian inversion failed ({exc}); '
                      'skipping hyperparameter credible intervals.')
            hess_info['hyp_std'] = None
            hess_info['hyp_std_error'] = str(exc)

    return best_hyper, best_logEvd, best_eMode, hess_info, best_model_hyper


def _minimize_hyperparameters(optVals, opt_keywords, bounds, opts, callback=None,
                              showOpt=0, status_callback=None):
    """Run the outer hyperparameter minimiser.

    The default path first tries a JAX/Optax L-BFGS hyperparameter update for
    the standard trial-wise EB case.  It uses a cached, jitted block-tridiagonal
    solve/logdet runner so repeated EB cycles do not rebuild the compiled
    optimiser.  Unsupported cases fall through to the mature SciPy objective.
    ``L-BFGS-B`` can occasionally report success with ``x`` unchanged on that
    finite-difference objective, so the SciPy path has a bounded Powell escape
    hatch for nonstationary stalls.
    """
    optVals = np.array(optVals, dtype=float)

    jax_result = _minimize_hyperparameters_jax(
        optVals, opt_keywords, bounds, opts,
        showOpt=showOpt, status_callback=status_callback,
    )
    if jax_result is not None:
        return jax_result

    result = minimize(
        _hyperOpt_lossfun,
        optVals,
        args=opt_keywords,
        method='L-BFGS-B',
        bounds=bounds,
        options=opts,
        callback=callback,
    )
    result.psytrax_method = 'L-BFGS-B'

    if not _should_retry_hyper_minimize(result, optVals, bounds):
        return result

    _emit_status(
        status_callback,
        "L-BFGS-B did not move despite a non-zero hyperparameter gradient; retrying with Powell…",
        stage="hyper",
    )
    if showOpt:
        print("L-BFGS-B hyper step stalled; retrying with Powell...")

    n_vals = len(optVals)
    powell_options = {
        'disp': False,
        'maxiter': max(20, 4 * n_vals),
        'maxfev': max(300, 45 * n_vals),
        'xtol': 1e-3,
        'ftol': 1e-3,
    }
    powell_result = minimize(
        _hyperOpt_lossfun,
        optVals,
        args=opt_keywords,
        method='Powell',
        bounds=bounds,
        options=powell_options,
    )
    powell_result.psytrax_method = 'Powell'

    # Keep the original L-BFGS-B answer if Powell failed to improve the same
    # raw objective; this preserves the fast path on genuinely flat surfaces.
    improvement_tol = max(1e-8, 1e-8 * abs(float(result.fun)))
    if (np.isfinite(powell_result.fun) and
            float(powell_result.fun) < float(result.fun) - improvement_tol):
        return powell_result
    return result


def _minimize_hyperparameters_jax(optVals, opt_keywords, bounds, opts,
                                  showOpt=0, status_callback=None):
    """JAX-native hyperparameter update for the standard trial-wise EB case."""
    disabled = os.environ.get('PSYTRAX_DISABLE_JAX_HYPER', '').lower()
    if disabled in {'1', 'true', 'yes'}:
        return None
    if opt_keywords.get('method') is not None:
        return None
    if opt_keywords.get('learning_rule') is not None:
        # The learning-rule objective has an extra prior-mean correction; keep
        # the mature SciPy path until the JAX version covers that case too.
        return None

    try:
        import jax
        import jax.numpy as jnp
        import optax
    except Exception:
        return None

    dat = opt_keywords['dat']
    hyper = opt_keywords['hyper']
    model_hyper = dict(opt_keywords.get('model_hyper') or {})
    optList = list(opt_keywords['optList'])
    model_hyper_optList = list(opt_keywords.get('model_hyper_optList') or [])
    ll_terms = opt_keywords['LL_terms']
    H_sparse = ll_terms.get('H')
    if H_sparse is None:
        return None

    K = int(ll_terms['K'])
    N = int(dat['r'].shape[0])
    dtype = jnp.float64
    H_prev_blocks = _sparse_trial_blocks(H_sparse, K, N)
    LL_v_time = np.asarray(opt_keywords['LL_v'], dtype=float).reshape(K, N).T

    try:
        runner = _get_jax_hyper_runner(
            opt_keywords, bounds, opts, jax, jnp, optax, dtype,
            showOpt=showOpt, status_callback=status_callback,
        )
        if runner is None:
            return None
        model_hyper_jax = {
            k: jnp.asarray(v, dtype=dtype)
            for k, v in model_hyper.items()
        }
        x_final, f0, f_final, grad_norm, n_iter = runner(
            jnp.asarray(optVals, dtype=dtype),
            jnp.asarray(H_prev_blocks, dtype=dtype),
            jnp.asarray(LL_v_time, dtype=dtype),
            model_hyper_jax,
        )
        x_final = np.asarray(x_final, dtype=float)
        f0 = float(f0)
        f_final = float(f_final)
        grad_norm = float(grad_norm)
        n_iter = int(n_iter)
    except Exception as exc:
        if showOpt:
            print(f'JAX hyper optimiser unavailable ({exc}); falling back to SciPy.')
        return None

    improvement_tol = max(1e-8, 1e-8 * abs(f0))
    if not np.isfinite(f_final) or f_final >= f0 - improvement_tol:
        if showOpt:
            print(f'JAX hyper optimiser did not improve objective '
                  f'({f0:.6g} -> {f_final:.6g}); falling back to SciPy.')
        return None

    result = OptimizeResult(
        x=x_final,
        fun=f_final,
        success=True,
        status=0,
        message='JAX L-BFGS hyperparameter update',
        nit=n_iter,
        nfev=n_iter,
        jac=None,
    )
    result.psytrax_method = 'JAX L-BFGS'
    if showOpt:
        print(f'JAX hyper grad norm: {grad_norm:.3e}')
    return result


def _get_jax_hyper_runner(opt_keywords, bounds, opts, jax, jnp, optax, dtype,
                          showOpt=0, status_callback=None):
    """Return a cached compiled JAX hyperparameter optimisation runner."""
    dat = opt_keywords['dat']
    hyper = opt_keywords['hyper']
    optList = list(opt_keywords['optList'])
    model_hyper = dict(opt_keywords.get('model_hyper') or {})
    model_hyper_optList = list(opt_keywords.get('model_hyper_optList') or [])
    likelihood_terms_fn = opt_keywords['log_lik_fns'][1]
    ll_terms = opt_keywords['LL_terms']
    K = int(ll_terms['K'])
    N = int(dat['r'].shape[0])
    max_iter = max(10, int((opts or {}).get('maxiter', 15)))
    grad_tol = float((opts or {}).get('gtol', 1e-3))

    hyper_specs = []
    count = 0
    for key in optList:
        if key not in hyper or hyper[key] is None:
            return None
        n = 1 if np.isscalar(hyper[key]) else K
        hyper_specs.append((key, count, n))
        count += n
    model_specs = []
    for key in model_hyper_optList:
        model_specs.append((key, count))
        count += 1
    if count != len(bounds):
        return None

    dat_signature = _jax_tree_signature(dat)
    hyper_signature = tuple(
        (key, 1 if np.isscalar(hyper[key]) else K)
        for key in optList
    )
    model_signature = tuple(model_hyper.keys())
    cache_key = (
        id(likelihood_terms_fn),
        K,
        N,
        tuple(optList),
        hyper_signature,
        tuple(model_hyper_optList),
        model_signature,
        tuple(tuple(bound) for bound in bounds),
        max_iter,
        grad_tol,
        dat_signature,
    )
    cache = opt_keywords.setdefault('_jax_hyper_cache', {})
    if cache_key in cache:
        return cache[cache_key]

    _emit_status(status_callback, "Compiling JAX hyperparameter optimiser…", stage="hyper")

    dat_jax = _to_jax_for_hyper(dat, jnp, dtype)
    lo = jnp.asarray([b[0] for b in bounds], dtype=dtype)
    hi = jnp.asarray([b[1] for b in bounds], dtype=dtype)

    day_lengths = np.asarray(dat.get('dayLength', np.array([], dtype=int)), dtype=int)
    days = np.cumsum(day_lengths, dtype=int)[:-1] if day_lengths.size else np.array([], dtype=int)
    days_jax = jnp.asarray(days, dtype=jnp.int32)
    missing = dat.get('missing_trials')
    missing_jax = None if missing is None else jnp.asarray(missing, dtype=dtype)
    fixed_hyper = {
        k: jnp.asarray(v, dtype=dtype)
        for k, v in hyper.items()
        if k not in optList and v is not None
    }

    def clip_x(x):
        return jnp.clip(x, lo, hi)

    def unpack_x(x, base_model_hyper):
        values = {}
        for key, start, n in hyper_specs:
            raw = 2.0 ** x[start:start + n]
            values[key] = raw[0] if n == 1 else raw
        mh = dict(base_model_hyper)
        for key, start in model_specs:
            mh[key] = 2.0 ** x[start]
        return values, mh

    def hyper_value(values, key, default=None):
        if key in values:
            return values[key]
        if key in fixed_hyper:
            return fixed_hyper[key]
        if default is not None:
            return default
        return None

    def broadcast_k(value):
        return jnp.broadcast_to(jnp.asarray(value, dtype=dtype), (K,))

    def inv_var_from_values(values):
        sigma_k = broadcast_k(hyper_value(values, 'sigma'))
        sig_init = hyper_value(values, 'sigInit', sigma_k)
        sig_init_k = broadcast_k(sig_init)
        sig_day = hyper_value(values, 'sigDay', sigma_k)
        sig_day_k = broadcast_k(sig_day)

        var = jnp.broadcast_to((sigma_k ** 2)[:, None], (K, N))
        if days.size:
            var = var.at[:, days_jax].set(sig_day_k[:, None] ** 2)
        var = var.at[:, 0].set(sig_init_k ** 2)
        if missing_jax is not None:
            var = var + missing_jax[None, :] * (sigma_k[:, None] ** 2)
        return 1.0 / var

    def precision_parts(inv_var):
        inv_t = inv_var.T
        q_diag = inv_t
        if N > 1:
            q_diag = q_diag.at[:-1, :].add(inv_t[1:, :])
            q_off = -inv_t[1:, :]
        else:
            q_off = jnp.zeros((0, K), dtype=dtype)
        return q_diag, q_off

    def prior_logprob(E_time, inv_var):
        E = E_time.T
        dE = jnp.zeros((K, N), dtype=dtype)
        dE = dE.at[:, 0].set(E[:, 0])
        if N > 1:
            dE = dE.at[:, 1:].set(E[:, 1:] - E[:, :-1])
        quad = jnp.sum(dE * dE * inv_var)
        logdet_inv = jnp.sum(jnp.log(inv_var))
        return 0.5 * (logdet_inv - quad)

    def loss_x(x, H_prev_blocks, LL_v_time, base_model_hyper):
        values, mh = unpack_x(x, base_model_hyper)
        inv_var = inv_var_from_values(values)
        q_diag, q_off = precision_parts(inv_var)
        A_prev = _blocks_from_prior_and_likelihood(jnp, q_diag, H_prev_blocks)
        E_time, _ = _block_tridiag_solve_logdet_jax(jnp, A_prev, q_off, LL_v_time)
        E = E_time.T

        logli, _, H_new = likelihood_terms_fn(E, dat_jax, mh)
        A_new = _blocks_from_prior_and_likelihood(jnp, q_diag, H_new)
        _, logdet_center = _block_tridiag_solve_logdet_jax(
            jnp, A_new, q_off, jnp.zeros((N, K), dtype=dtype),
            solve_rhs=False,
        )
        evd = logli + prior_logprob(E_time, inv_var) - 0.5 * logdet_center
        return jnp.nan_to_num(-evd, nan=1e20, posinf=1e20, neginf=1e20)

    def bounded_loss_x(x, H_prev_blocks, LL_v_time, base_model_hyper):
        return loss_x(clip_x(x), H_prev_blocks, LL_v_time, base_model_hyper)

    solver = optax.lbfgs(memory_size=10, scale_init_precond=True)
    value_and_grad = optax.value_and_grad_from_state(bounded_loss_x)

    @jax.jit
    def run(opt_vals, H_prev_blocks, LL_v_time, base_model_hyper):
        x0 = clip_x(opt_vals)
        state0 = solver.init(x0)
        f0 = bounded_loss_x(x0, H_prev_blocks, LL_v_time, base_model_hyper)
        init = (jnp.asarray(0, dtype=jnp.int32),
                x0,
                state0,
                f0,
                jnp.asarray(jnp.inf, dtype=dtype),
                jnp.asarray(False))

        def do_step(carry):
            n_iter, x, opt_state, _value, _grad_norm, _done = carry
            value, grad = value_and_grad(
                x, H_prev_blocks, LL_v_time, base_model_hyper,
                state=opt_state,
            )
            updates, new_state = solver.update(
                grad, opt_state, x,
                value=value,
                grad=grad,
                value_fn=bounded_loss_x,
                H_prev_blocks=H_prev_blocks,
                LL_v_time=LL_v_time,
                base_model_hyper=base_model_hyper,
            )
            x_new = clip_x(optax.apply_updates(x, updates))
            grad_norm = jnp.max(jnp.abs(grad))
            done = grad_norm < grad_tol
            return n_iter + 1, x_new, new_state, value, grad_norm, done

        def body(_, carry):
            return jax.lax.cond(carry[-1], lambda c: c, do_step, carry)

        n_iter, x_final, _state, _value, grad_norm, _done = jax.lax.fori_loop(
            0, max_iter, body, init,
        )
        f_final = bounded_loss_x(x_final, H_prev_blocks, LL_v_time, base_model_hyper)
        return x_final, f0, f_final, grad_norm, n_iter

    cache[cache_key] = run
    return run


def _jax_tree_signature(x):
    if isinstance(x, dict):
        return tuple((k, _jax_tree_signature(v)) for k, v in sorted(x.items()))
    if isinstance(x, np.ndarray):
        return (x.shape, str(x.dtype))
    if np.isscalar(x):
        return ('scalar', type(x).__name__)
    if x is None:
        return None
    return type(x).__name__


def _to_jax_for_hyper(dat, jnp, dtype):
    if isinstance(dat, dict):
        return {k: _to_jax_for_hyper(v, jnp, dtype) for k, v in dat.items()}
    if isinstance(dat, np.ndarray):
        if np.issubdtype(dat.dtype, np.floating):
            return jnp.asarray(dat, dtype=dtype)
        return jnp.asarray(dat)
    return dat


def _sparse_trial_blocks(H, K, N):
    """Recover ``(N, K, K)`` trial Hessian blocks from ``myblk_diags`` output."""
    blocks = np.empty((N, K, K), dtype=float)
    for i in range(K):
        for j in range(K):
            offset = (j - i) * N
            start = min(i, j) * N
            blocks[:, i, j] = H.diagonal(offset)[start:start + N]
    return blocks


def _blocks_from_prior_and_likelihood(jnp, q_diag, H_blocks):
    eye = jnp.eye(H_blocks.shape[1], dtype=H_blocks.dtype)
    return q_diag[:, :, None] * eye[None, :, :] - H_blocks


def _block_tridiag_solve_logdet_jax(jnp, A, q_off, rhs, solve_rhs=True):
    """Solve/logdet for symmetric block-tridiagonal matrices with diagonal off-blocks."""
    N, K = rhs.shape
    dtype = A.dtype

    def diag_batch(v):
        eye = jnp.eye(K, dtype=dtype)
        return v[:, :, None] * eye[None, :, :]

    C = diag_batch(q_off) if N > 1 else jnp.zeros((0, K, K), dtype=dtype)
    C_pad = jnp.concatenate([C, jnp.zeros((1, K, K), dtype=dtype)], axis=0)

    _, logdet0 = jnp.linalg.slogdet(A[0])
    cp0 = jnp.linalg.solve(A[0], C_pad[0])
    dp0 = jnp.linalg.solve(A[0], rhs[0]) if solve_rhs else jnp.zeros((K,), dtype=dtype)

    def forward(carry, x):
        cp_prev, dp_prev, logdet_prev = carry
        A_t, C_lower, C_upper, rhs_t = x
        denom = A_t - C_lower @ cp_prev
        _, logdet = jnp.linalg.slogdet(denom)
        cp = jnp.linalg.solve(denom, C_upper)
        rhs_eff = rhs_t - C_lower @ dp_prev
        dp = jnp.linalg.solve(denom, rhs_eff) if solve_rhs else jnp.zeros((K,), dtype=dtype)
        return (cp, dp, logdet_prev + logdet), (cp, dp)

    if N > 1:
        xs = (A[1:], C, C_pad[1:], rhs[1:])
        (_, _, logdet), (cp_tail, dp_tail) = jax_lax_scan(
            jnp, forward, (cp0, dp0, logdet0), xs,
        )
        cps = jnp.concatenate([cp0[None, :, :], cp_tail], axis=0)
        dps = jnp.concatenate([dp0[None, :], dp_tail], axis=0)
    else:
        logdet = logdet0
        cps = cp0[None, :, :]
        dps = dp0[None, :]

    if not solve_rhs:
        return rhs, logdet

    def backward(x_next, x):
        cp, dp = x
        x_cur = dp - cp @ x_next
        return x_cur, x_cur

    if N > 1:
        _, rev = jax_lax_scan(jnp, backward, dps[-1], (cps[:-1][::-1], dps[:-1][::-1]))
        sol = jnp.concatenate([rev[::-1], dps[-1][None, :]], axis=0)
    else:
        sol = dps
    return sol, logdet


def jax_lax_scan(jnp, fn, init, xs):
    # Keep the import local so importing psytrax._hyper_opt does not initialize
    # JAX unless the JAX hyper optimiser is actually used.
    import jax
    return jax.lax.scan(fn, init, xs)


def _should_retry_hyper_minimize(result, optVals, bounds,
                                 step_tol=1e-4, grad_tol=1e-2):
    """Return True when L-BFGS-B appears to have stalled at a nonstationary point."""
    x = np.asarray(result.x, dtype=float)
    optVals = np.asarray(optVals, dtype=float)
    rel_step = np.linalg.norm((x - optVals) / np.maximum(np.abs(optVals), 1e-8))
    if rel_step > step_tol:
        return False

    jac = getattr(result, 'jac', None)
    if jac is None:
        return False
    jac = np.asarray(jac, dtype=float)
    if jac.shape != optVals.shape or not np.any(np.isfinite(jac)):
        return False

    free = np.ones_like(optVals, dtype=bool)
    if bounds is not None:
        for i, bound in enumerate(bounds):
            if bound is None:
                continue
            lo, hi = bound
            if lo is not None and optVals[i] <= lo + 1e-8 and jac[i] > 0:
                free[i] = False
            if hi is not None and optVals[i] >= hi - 1e-8 and jac[i] < 0:
                free[i] = False

    if not np.any(free):
        return False
    return np.nanmax(np.abs(jac[free])) > grad_tol


def _hyperparameter_update_size(old_x, new_x):
    """Return the absolute Euclidean step size in log2-hyperparameter space."""
    old_x = np.asarray(old_x, dtype=float)
    new_x = np.asarray(new_x, dtype=float)
    return float(np.linalg.norm(new_x - old_x))


def _limit_hyperparameter_step(result, optVals, opt_keywords,
                               max_log2_step=2.0, showOpt=0):
    """Damp very large EB hyperparameter moves in log2 space.

    The outer optimiser works in log2 coordinates, so an absolute displacement
    is already multiplicative: one unit is a 2x change, two units is a 4x
    change.  Limiting the largest coordinate move keeps the decoupled Laplace
    proposal close enough that the next full MAP/evidence cycle is less likely
    to jump into a poor region.
    """
    optVals = np.asarray(optVals, dtype=float)
    target = np.asarray(result.x, dtype=float)
    delta = target - optVals
    max_abs = float(np.nanmax(np.abs(delta))) if delta.size else 0.0
    if not np.isfinite(max_abs) or max_abs <= max_log2_step:
        return result

    base_fun = _hyperOpt_lossfun(optVals, opt_keywords)
    if not np.isfinite(base_fun):
        return result

    initial_scale = max_log2_step / max_abs
    best_x = optVals
    best_fun = base_fun
    scale = initial_scale
    improvement_tol = max(1e-8, 1e-8 * abs(float(base_fun)))

    for _ in range(10):
        candidate = optVals + scale * delta
        candidate_fun = _hyperOpt_lossfun(candidate, opt_keywords)
        if (np.isfinite(candidate_fun) and
                candidate_fun < best_fun - improvement_tol):
            best_x = candidate
            best_fun = candidate_fun
            break
        scale *= 0.5

    limited = OptimizeResult(result.copy())
    limited.x = best_x
    limited.fun = best_fun
    limited.psytrax_method = getattr(result, 'psytrax_method', 'unknown')
    if np.array_equal(best_x, optVals):
        limited.psytrax_method += ' (step rejected)'
    else:
        limited.psytrax_method += f' (step limited {scale:.3f}x)'
    if showOpt and not np.array_equal(best_x, target):
        print(
            f'Limited hyper step: max |Δlog2| {max_abs:.3f} -> '
            f'{np.nanmax(np.abs(best_x - optVals)):.3f}'
        )
    return limited


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

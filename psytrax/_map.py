import numpy as onp
import jax
import jax.numpy as jnp
from scipy.optimize import minimize
from tqdm.auto import tqdm

from psytrax._helper.memoize import memoize
from psytrax._helper.jacHessCheck import jacHessCheck, jacEltsCheck
from psytrax._helper.helperFunctions import (
    DT_X_D,
    sparse_logdet,
    make_invSigma,
    myblk_diags,
)

jax.config.update("jax_enable_x64", True)

# Dtype used for JAX vmap calls (likelihood / gradient / hessian).
# Set via psytrax.fit(..., precision='float32') — never change directly.
_JAX_DTYPE = jnp.float64


def getMAP(dat, hyper, n_params, log_lik_fns, method=None, E0=None, showOpt=0,
           pbar=None, map_tol=1e-6):
    """Estimate MAP parameters under a Gaussian random-walk prior.

    Args:
        dat         : dict with at least 'inputs' (dict) and 'r' (responses)
        hyper       : dict with at least 'sigma'; optionally 'sigInit', 'sigDay'
        n_params    : int, number of parameters per trial (K)
        log_lik_fns : tuple (log_likelihood_fn, likelihood_terms_fn) as returned
                      by make_likelihood_fns()
        method      : None (trial-by-trial) | '_constant' | '_days'
        E0          : initial parameter array, shape (w_N*K,) or (K, w_N)
        showOpt     : 0 silent | 1 verbose | 2+ also runs derivative checks
        map_tol     : trust-region convergence tolerance

    Returns:
        hess    : dict of sparse matrices for the Laplace approximation
        logEvd  : float, log marginal likelihood
        llstruct: dict with likelihood/prior terms and eMode
    """
    if 'inputs' not in dat or 'r' not in dat or not isinstance(dat['inputs'], dict):
        raise Exception("dat must contain 'inputs' (dict) and 'r'")

    N = len(dat['r'])
    K = n_params

    if method is None:
        w_N = N
    elif method == '_constant':
        w_N = 1
    elif method == '_days':
        w_N = len(dat['dayLength'])
    else:
        raise Exception(f"method '{method}' not supported")

    if E0 is not None:
        if not isinstance(E0, onp.ndarray):
            raise Exception(f'E0 must be a numpy array, not {type(E0)}')
        if E0.shape == (w_N * K,):
            eInit = E0.copy()
        elif E0.shape == (K, w_N):
            eInit = E0.flatten()
        else:
            raise Exception(f'E0 must be shape ({w_N*K},) or ({K}, {w_N}), not {E0.shape}')
    else:
        eInit = 0.01 * onp.ones(w_N * K)

    if 'sigma' not in hyper:
        raise Exception("'sigma' not found in hyper dict")
    if method == '_constant' and hyper.get('sigInit') is None:
        print(f'WARNING: sigInit being set to sigma for method {method}')
    if method == '_days' and hyper.get('sigDay') is None:
        print(f'WARNING: sigDay being set to sigma for method {method}')
    if 'dayLength' not in dat and hyper.get('sigDay') is not None:
        print('WARNING: sigDay has no effect, dayLength not in dat')
        dat['dayLength'] = onp.array([], dtype=int)

    dat.setdefault('missing_trials', None)

    # --- MAP optimisation ---
    prior_cache = make_prior_cache(dat, hyper, K, method)
    lossfun = memoize(negLogPost)
    my_args = (dat, prior_cache, log_lik_fns, method)

    _map_pbar = tqdm(desc='  MAP', unit='iter', leave=False, disable=pbar is None)

    def _callback(xk):
        _map_pbar.update(1)
        val = lossfun(xk, *my_args)
        _map_pbar.set_postfix({'loss': f'{val:.3f}'})
        if pbar is not None:
            pbar.set_postfix({'MAP loss': f'{val:.3f}'})

    # maxiter: trust-ncg default is 200*len(x), which is huge for large N*K.
    # Cap at 2000 — the optimizer typically converges in < 100 iterations.
    max_iter = 2000
    if showOpt:
        opts = {'disp': True,  'maxiter': max_iter}
        callback = lambda x: (print(x), _callback(x))
    else:
        opts = {'disp': False, 'maxiter': max_iter}
        callback = _callback

    if showOpt:
        print('Obtaining MAP estimate...')

    result = minimize(
        lossfun,
        eInit,
        jac=lossfun.jacobian,
        hessp=lossfun.hessian_prod,
        method='trust-ncg',
        tol=map_tol,
        args=my_args,
        options=opts,
        callback=callback,
    )

    _map_pbar.close()
    eMode = result.x

    if showOpt and not result.success:
        print('WARNING — MAP estimate: minimize() did not converge\n', result.message)

    if showOpt >= 2:
        print('** Jacobian and Hessian Check **')
        for check in range(showOpt - 1):
            print(f'\nCheck {check + 1}:')
            jacHessCheck(lossfun, eMode, *my_args)
        print('** Jacobian element check **')
        for check in range(showOpt - 1):
            print(f'\nCheck {check + 1}:')
            jacEltsCheck(lossfun, 2, eMode, *my_args)

    # --- Evidence (Laplace approximation) ---
    if showOpt:
        print('Calculating evidence...')

    pT, lT = getPosteriorTerms(eMode, *my_args)
    hess = {'H': lT['ddlogli']['H'], 'K': lT['ddlogli']['K'], 'ddlogprior': pT['ddlogprior']}

    center = -pT['ddlogprior'] - lT['ddlogli']['H']
    logterm_post = 0.5 * sparse_logdet(center)
    logEvd = lT['logli'] + pT['logprior'] - logterm_post

    if showOpt:
        print('Evidence:', logEvd)

    llstruct = {'lT': lT, 'pT': pT, 'eMode': eMode}
    return hess, logEvd, llstruct


def negLogPost(E_flat, dat, prior_cache, log_lik_fns, method=None):
    """Return (neg log-posterior, its gradient, its Hessian dict) at E_flat."""
    priorTerms, liTerms = getPosteriorTerms(
        E_flat, dat, prior_cache, log_lik_fns, method
    )
    negPost = -priorTerms['logprior'] - liTerms['logli']
    negdPost = -priorTerms['dlogprior'] - liTerms['dlogli']
    negddPost = {
        'negddlogprior': -priorTerms['ddlogprior'],
        'negH': -liTerms['ddlogli']['H'],
        'K': liTerms['ddlogli']['K'],
    }
    return negPost, negdPost, negddPost


def getPosteriorTerms(E_flat, dat, hyper_or_prior, log_lik_fns, method=None):
    """Compute prior and likelihood terms (with derivatives) at E_flat.

    Args:
        E_flat      : (w_N * K,) flat parameter vector
        dat         : data dict
        hyper_or_prior : hyperparameter dict or cached prior terms
        log_lik_fns : (log_likelihood_fn, likelihood_terms_fn)
        method      : None | '_constant' | '_days'

    Returns:
        priorTerms : dict with 'logprior', 'dlogprior', 'ddlogprior'
        liTerms    : dict with 'logli', 'dlogli', 'ddlogli'
    """
    if method in ['_days', '_constant']:
        raise NotImplementedError("'_days' and '_constant' methods are not yet supported")

    dat.setdefault('dayLength', onp.array([], dtype=int))
    dat.setdefault('missing_trials', None)

    prior_cache = (
        hyper_or_prior
        if isinstance(hyper_or_prior, dict) and 'invC' in hyper_or_prior
        else make_prior_cache(
            dat, hyper_or_prior, int(len(E_flat) / len(dat['r'])), method
        )
    )
    K = prior_cache['K']
    w_N = prior_cache['w_N']

    if E_flat.shape != (w_N * K,):
        raise Exception(f'E_flat shape {E_flat.shape} != ({w_N * K},)')

    # --- Prior ---
    invC = prior_cache['invC']
    logdet_C = prior_cache['logdet_C']
    logprior = 0.5 * (-logdet_C - E_flat @ invC @ E_flat)
    dlogprior = -invC @ E_flat
    ddlogprior = prior_cache['ddlogprior']

    priorTerms = {'logprior': logprior, 'dlogprior': dlogprior, 'ddlogprior': ddlogprior}

    # --- Likelihood ---
    # Cast E and floating dat arrays to _JAX_DTYPE for the vmap call so that
    # float32 precision (and GPU) can be used without changing scipy's optimizer.
    _, likelihood_terms_fn = log_lik_fns
    E = onp.reshape(E_flat, (K, w_N), order='C')
    E_jax = jnp.asarray(E, dtype=_JAX_DTYPE)

    def _cast(x):
        if isinstance(x, (onp.ndarray, jnp.ndarray)) and jnp.issubdtype(
                jnp.asarray(x).dtype, jnp.floating):
            return jnp.asarray(x, dtype=_JAX_DTYPE)
        return x
    dat_jax = jax.tree_util.tree_map(_cast, dat)

    logli, dlogli_matrix, HlliList = likelihood_terms_fn(E_jax, dat_jax)
    # Cast back to float64 so scipy's trust-ncg optimizer stays numerically stable
    dlogli = onp.asarray(dlogli_matrix, dtype=onp.float64).flatten()
    HlliList = onp.asarray(HlliList, dtype=onp.float64)
    ddlogli = {'H': myblk_diags(HlliList), 'K': K}

    liTerms = {'logli': logli, 'dlogli': dlogli, 'ddlogli': ddlogli}
    return priorTerms, liTerms


def make_prior_cache(dat, hyper, n_params, method=None):
    """Precompute prior quantities that stay fixed within a MAP solve."""
    dat.setdefault('dayLength', onp.array([], dtype=int))
    dat.setdefault('missing_trials', None)

    N = len(dat['r'])
    K = n_params

    if method is None:
        w_N = N
        days = onp.cumsum(dat['dayLength'], dtype=int)[:-1]
        missing_trials = dat['missing_trials']
    elif method == '_constant':
        w_N = 1
        days = onp.array([], dtype=int)
        missing_trials = None
    elif method == '_days':
        w_N = len(dat['dayLength'])
        days = onp.arange(1, w_N, dtype=int)
        missing_trials = None
    else:
        raise Exception(f"method '{method}' not supported")

    invSigma = make_invSigma(hyper, days, missing_trials, w_N, K)
    invC = DT_X_D(invSigma, K)

    return {
        'K': K,
        'w_N': w_N,
        'invC': invC,
        'ddlogprior': -invC,
        'logdet_C': -onp.sum(jnp.log(invSigma.diagonal())),
    }

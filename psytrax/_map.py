import numpy as onp
import jax
import jax.numpy as jnp
from jax import grad
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


def getMAP(dat, hyper, n_params, log_lik_fns, method=None, E0=None, showOpt=0, pbar=None):
    """Estimate MAP parameters under a Gaussian random-walk prior.

    Args:
        dat         : dict with at least 'inputs' (dict) and 'r' (responses)
        hyper       : dict with at least 'sigma'; optionally 'sigInit', 'sigDay'
        n_params    : int, number of parameters per trial (K)
        log_lik_fns : tuple (log_likelihood_fn, ll_hessian_blks_fn) as returned
                      by make_likelihood_fns()
        method      : None (trial-by-trial) | '_constant' | '_days'
        E0          : initial parameter array, shape (w_N*K,) or (K, w_N)
        showOpt     : 0 silent | 1 verbose | 2+ also runs derivative checks

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
    lossfun = memoize(negLogPost)
    my_args = (dat, hyper, log_lik_fns, method)

    _map_pbar = tqdm(desc='  MAP', unit='iter', leave=False, disable=pbar is None)

    def _callback(xk):
        _map_pbar.update(1)
        val = lossfun(xk, *my_args)
        _map_pbar.set_postfix({'loss': f'{val:.3f}'})
        if pbar is not None:
            pbar.set_postfix({'MAP loss': f'{val:.3f}'})

    if showOpt:
        opts = {'disp': True}
        callback = lambda x: (print(x), _callback(x))
    else:
        opts = {'disp': False}
        callback = _callback

    if showOpt:
        print('Obtaining MAP estimate...')

    result = minimize(
        lossfun,
        eInit,
        jac=lossfun.jacobian,
        hessp=lossfun.hessian_prod,
        method='trust-ncg',
        tol=1e-9,
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


def negLogPost(E_flat, dat, hyper, log_lik_fns, method=None):
    """Return (neg log-posterior, its gradient, its Hessian dict) at E_flat."""
    priorTerms, liTerms = getPosteriorTerms(E_flat, dat, hyper, log_lik_fns, method)
    negPost = -priorTerms['logprior'] - liTerms['logli']
    negdPost = -priorTerms['dlogprior'] - liTerms['dlogli']
    negddPost = {
        'negddlogprior': -priorTerms['ddlogprior'],
        'negH': -liTerms['ddlogli']['H'],
        'K': liTerms['ddlogli']['K'],
    }
    return negPost, negdPost, negddPost


def getPosteriorTerms(E_flat, dat, hyper, log_lik_fns, method=None):
    """Compute prior and likelihood terms (with derivatives) at E_flat.

    Args:
        E_flat      : (w_N * K,) flat parameter vector
        dat         : data dict
        hyper       : hyperparameter dict
        log_lik_fns : (log_likelihood_fn, ll_hessian_blks_fn)
        method      : None | '_constant' | '_days'

    Returns:
        priorTerms : dict with 'logprior', 'dlogprior', 'ddlogprior'
        liTerms    : dict with 'logli', 'dlogli', 'ddlogli'
    """
    if method in ['_days', '_constant']:
        raise NotImplementedError("'_days' and '_constant' methods are not yet supported")

    dat.setdefault('dayLength', onp.array([], dtype=int))
    dat.setdefault('missing_trials', None)

    N = len(dat['r'])
    K = int(len(E_flat) / N)

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

    if E_flat.shape != (w_N * K,):
        raise Exception(f'E_flat shape {E_flat.shape} != ({w_N * K},)')

    # --- Prior ---
    invSigma = make_invSigma(hyper, days, missing_trials, w_N, K)
    invC = DT_X_D(invSigma, K)

    logdet_C = -onp.sum(jnp.log(invSigma.diagonal()))
    logprior = 0.5 * (-logdet_C - E_flat @ invC @ E_flat)
    dlogprior = -invC @ E_flat
    ddlogprior = -invC

    priorTerms = {'logprior': logprior, 'dlogprior': dlogprior, 'ddlogprior': ddlogprior}

    # --- Likelihood ---
    log_likelihood_fn, ll_hessian_blks_fn = log_lik_fns
    E = onp.reshape(E_flat, (K, w_N), order='C')

    HlliList = ll_hessian_blks_fn(E, dat)
    logli = log_likelihood_fn(E, dat)
    dlogli = grad(log_likelihood_fn)(E, dat).flatten()
    ddlogli = {'H': myblk_diags(HlliList), 'K': K}

    liTerms = {'logli': logli, 'dlogli': dlogli, 'ddlogli': ddlogli}
    return priorTerms, liTerms

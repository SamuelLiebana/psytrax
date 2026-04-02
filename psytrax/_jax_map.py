"""JAX-native MAP optimisation for the psytrax random-walk model.

Replaces the scipy trust-NCG inner loop in getMAP with a fully JAX-traceable
L-BFGS optimisation.  The prior and likelihood are both computed in JAX so
the entire inner loop can run on GPU (Metal, CUDA) with no CPU round-trips.

After L-BFGS converges the Hessian and Laplace evidence are computed once
using the existing numpy/scipy code (cheap relative to the optimisation).
"""

import numpy as np
import jax
import jax.numpy as jnp
from tqdm.auto import tqdm

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# JAX-traceable Gaussian random-walk log-prior
# ---------------------------------------------------------------------------

def _log_prior_jax(E_flat, K, N, sigma_k, sigInit_k, is_boundary, sigDay_k):
    """Gaussian random-walk log-prior, fully JAX-traceable.

    Args:
        E_flat      : (K*N,) parameter vector (C order, so E[k,t] = E_flat[k*N+t])
        K, N        : static Python ints
        sigma_k     : (K,) process noise std
        sigInit_k   : (K,) initial uncertainty std
        is_boundary : (N-1,) bool mask — True where session boundary occurs,
                      or None if no session boundaries
        sigDay_k    : (K,) session-boundary noise std (ignored if is_boundary is None)
    """
    E = jnp.reshape(E_flat, (K, N))

    # Initial term: E[:, 0] ~ N(0, sigInit_k^2)
    lp = jnp.sum(-0.5 * (E[:, 0] / sigInit_k) ** 2 - jnp.log(sigInit_k))

    if N == 1:
        return lp

    dE = E[:, 1:] - E[:, :-1]                          # (K, N-1)

    if is_boundary is None:
        # Uniform process noise: sigma_k broadcast over all transitions
        sig_t = jnp.broadcast_to(sigma_k[:, None], (K, N - 1))
    else:
        # At session boundaries use sigDay_k, elsewhere sigma_k
        sig_t = jnp.where(is_boundary[None, :], sigDay_k[:, None], sigma_k[:, None])

    lp += jnp.sum(-0.5 * (dE / sig_t) ** 2 - jnp.log(sig_t))
    return lp


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def getMAP_jax(dat, hyper, n_params, log_lik_fns,
               E0=None, method=None, showOpt=0, pbar=None, map_tol=1e-6):
    """MAP estimation using JAX L-BFGS — GPU-compatible drop-in for getMAP.

    The inner optimisation loop runs entirely in JAX (GPU-native).
    The Hessian and log-evidence (needed for the Laplace approximation) are
    computed once after convergence using the existing numpy/scipy code.

    Args: same as psytrax._map.getMAP
    Returns: (hess, logEvd, llstruct)  — same structure as getMAP
    """
    from psytrax._map import getPosteriorTerms, make_prior_cache, _JAX_DTYPE
    from psytrax._helper.helperFunctions import sparse_logdet

    K = n_params
    N = len(dat['r'])
    dtype = _JAX_DTYPE

    # ---- build prior arrays ----
    sigma   = hyper['sigma']
    sigInit = hyper.get('sigInit', np.full(K, 2 ** 4))
    sigDay  = hyper.get('sigDay', None)

    day_lengths = dat.get('dayLength', np.array([], dtype=int))
    if len(day_lengths) > 0:
        session_starts = np.cumsum(day_lengths, dtype=int)[:-1]   # first trial of each new session
    else:
        session_starts = np.array([], dtype=int)

    sigma_k   = jnp.broadcast_to(jnp.asarray(sigma,   dtype=dtype), (K,))
    sigInit_k = jnp.broadcast_to(jnp.asarray(sigInit, dtype=dtype), (K,))

    if sigDay is not None and len(session_starts) > 0:
        sigDay_k = jnp.broadcast_to(jnp.asarray(sigDay, dtype=dtype), (K,))
        # dE[:, t] = E[:, t+1] - E[:, t]; the boundary at session_start d
        # uses sigDay for the transition d-1 → d, i.e., index d-1 in dE
        bd = np.clip(session_starts - 1, 0, N - 2)
        is_boundary = jnp.zeros(N - 1, dtype=jnp.bool_).at[bd].set(True)
    else:
        sigDay_k    = sigma_k          # placeholder, unused when is_boundary is None
        is_boundary = None

    # ---- convert dat to JAX arrays ----
    def _cast(x):
        if isinstance(x, np.ndarray) and np.issubdtype(x.dtype, np.floating):
            return jnp.asarray(x, dtype=dtype)
        elif isinstance(x, np.ndarray):
            return jnp.asarray(x)
        return x

    dat_jax = jax.tree_util.tree_map(_cast, dat)
    log_likelihood_fn = log_lik_fns[0]

    # ---- JIT-compiled objective (traced once per unique is_boundary shape) ----
    @jax.jit
    def neg_log_post(E_flat):
        E_flat_t = E_flat.astype(dtype)
        E = jnp.reshape(E_flat_t, (K, N))
        logli = log_likelihood_fn(E, dat_jax)
        lp = _log_prior_jax(E_flat_t, K, N,
                             sigma_k, sigInit_k, is_boundary, sigDay_k)
        return -(logli + lp)

    # ---- initial parameters ----
    if E0 is None:
        E0_flat = jnp.full(K * N, 0.01, dtype=dtype)
    elif isinstance(E0, np.ndarray):
        if E0.shape == (K, N):
            E0_flat = jnp.asarray(E0.flatten(order='C'), dtype=dtype)
        else:
            E0_flat = jnp.asarray(E0.flatten(), dtype=dtype)
    else:
        E0_flat = E0.flatten().astype(dtype)

    # ---- run L-BFGS ----
    result = jax.scipy.optimize.minimize(
        neg_log_post,
        E0_flat,
        method='l-bfgs-experimental-do-not-rely-on-this',
        options={'maxiter': 2000, 'gtol': float(map_tol)},
    )

    if showOpt and not bool(result.success):
        print(f'  WARNING — JAX L-BFGS did not converge: {result.status}')

    eMode = np.array(result.x, dtype=np.float64)

    # ---- Hessian + Laplace evidence (numpy/scipy, cheap one-time cost) ----
    pT, lT = getPosteriorTerms(eMode, dat, hyper, log_lik_fns, method=None)

    hess = {
        'H':          lT['ddlogli']['H'],
        'K':          lT['ddlogli']['K'],
        'ddlogprior': pT['ddlogprior'],
    }
    center      = -pT['ddlogprior'] - lT['ddlogli']['H']
    logterm_post = 0.5 * sparse_logdet(center)
    logEvd      = float(lT['logli']) + float(pT['logprior']) - logterm_post

    llstruct = {'lT': lT, 'pT': pT, 'eMode': eMode}
    return hess, logEvd, llstruct

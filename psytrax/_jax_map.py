"""JAX-native MAP optimisation for the psytrax random-walk model.

Replaces the scipy trust-NCG inner loop in getMAP with a fully JAX-traceable
L-BFGS optimisation running in the **prior-whitened** parameter space.

Prior whitening
---------------
The Gaussian random-walk prior precision Q is block-tridiagonal — K independent
N×N tridiagonal matrices, one per parameter dimension.  We compute the Cholesky
factor L analytically (lower bidiagonal, O(KN) via lax.scan) and optimise in
z = L^T e space.  In z-space the prior contributes the identity matrix to the
Hessian, so L-BFGS starts with an accurate Hessian approximation and avoids
the large early steps that can jump into model-sentinel territory.

Sentinel barrier
----------------
Model log-likelihoods use lax.cond to return the constant -1e12 for invalid
parameter combinations.  That constant has zero gradient, which traps L-BFGS.
A smooth repulsive L2 barrier, active only when the mean per-trial log-likelihood
falls below -50 nats (never in valid territory), provides a non-zero gradient
pointing back toward the prior mean.

After L-BFGS converges the Hessian and Laplace evidence are computed once
using the existing numpy/scipy code (cheap relative to the optimisation).
"""

import numpy as np
import jax
import jax.numpy as jnp

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
# Prior-whitening: Cholesky of the tridiagonal prior precision
# ---------------------------------------------------------------------------

def _prior_chol(K, N, sigma_k, sigInit_k, is_boundary, sigDay_k):
    """Lower-bidiagonal Cholesky factor L of the prior precision Q = L L^T.

    Each parameter dimension k has an independent N×N tridiagonal precision
    matrix.  L is lower bidiagonal and is computed in O(KN) via a forward
    recursion (jax.lax.scan).

    Returns
    -------
    L_diag : (K, N)   — diagonal elements L[k, t, t]
    L_sub  : (K, N-1) — sub-diagonal elements L[k, t+1, t]
    """
    dtype = sigma_k.dtype

    if N == 1:
        # Precision is just diag(1/sigInit_k^2); Cholesky is diag(1/sigInit_k)
        return (1.0 / sigInit_k)[:, None], jnp.zeros((K, 0), dtype=dtype)

    # Transition stds: (K, N-1)
    if is_boundary is None:
        sig_t = jnp.broadcast_to(sigma_k[:, None], (K, N - 1))
    else:
        sig_t = jnp.where(is_boundary[None, :], sigDay_k[:, None], sigma_k[:, None])

    prec_t = 1.0 / (sig_t ** 2)  # (K, N-1)

    # Diagonal of Q
    q0    = 1.0 / (sigInit_k ** 2) + prec_t[:, 0]   # (K,)
    q_end = prec_t[:, -1]                             # (K,)
    if N == 2:
        q_diag = jnp.stack([q0, q_end], axis=1)      # (K, 2)
    else:
        q_mid  = prec_t[:, :-1] + prec_t[:, 1:]      # (K, N-2)
        q_diag = jnp.concatenate(
            [q0[:, None], q_mid, q_end[:, None]], axis=1)  # (K, N)

    # Sub-diagonal of Q: Q[k, t+1, t] = -prec_t[k, t]
    q_sub = -prec_t  # (K, N-1)

    # Forward Cholesky recursion: for t = 0 .. N-2
    #   L_sub[k, t]     = Q[k, t+1, t] / L_diag[k, t]
    #   L_diag[k, t+1]  = sqrt(Q[k, t+1, t+1] - L_sub[k, t]^2)
    def chol_step(l_diag_prev, xs_t):
        q_sub_t, q_diag_tp1 = xs_t                          # both (K,)
        l_sub_t    = q_sub_t / l_diag_prev
        l_diag_tp1 = jnp.sqrt(q_diag_tp1 - l_sub_t ** 2)
        return l_diag_tp1, (l_sub_t, l_diag_tp1)

    l0  = jnp.sqrt(q_diag[:, 0])                            # (K,)
    xs  = (q_sub.T, q_diag[:, 1:].T)                        # each (N-1, K)
    _, (l_sub_arr, l_diag_rest) = jax.lax.scan(chol_step, l0, xs)
    # l_sub_arr   : (N-1, K) — sub-diagonal elements
    # l_diag_rest : (N-1, K) — diagonal elements t = 1..N-1

    L_diag = jnp.concatenate([l0[None, :], l_diag_rest], axis=0).T  # (K, N)
    L_sub  = l_sub_arr.T                                              # (K, N-1)
    return L_diag, L_sub


def _whiten(E_flat, K, N, L_diag, L_sub):
    """Forward whitening transform z = L^T e.

    In z-space the random-walk prior is N(0, I), so L-BFGS begins with an
    accurate (identity) Hessian approximation for the prior's contribution.
    """
    E = jnp.reshape(E_flat, (K, N))
    if N == 1:
        return (L_diag[:, 0] * E[:, 0]).reshape(-1)
    # L^T is upper bidiagonal: z[k,t] = L[t,t]*e[t] + L[t+1,t]*e[t+1]  for t<N-1
    Z_body = L_diag[:, :-1] * E[:, :-1] + L_sub * E[:, 1:]  # (K, N-1)
    Z_tail = (L_diag[:, -1] * E[:, -1])[:, None]              # (K, 1)
    return jnp.concatenate([Z_body, Z_tail], axis=1).reshape(-1)


def _unwhiten(z_flat, K, N, L_diag, L_sub):
    """Inverse whitening transform e = L^{-T} z (back-substitution).

    Solves the upper-bidiagonal system L^T e = z from the last row upward.
    """
    Z = jnp.reshape(z_flat, (K, N))
    if N == 1:
        return (Z[:, 0] / L_diag[:, 0]).reshape(-1)

    e_last = Z[:, N - 1] / L_diag[:, N - 1]   # (K,)

    # Scan from t = N-2 down to t = 0
    def back_step(e_next, xs_t):
        z_t, l_sub_t, l_diag_t = xs_t          # all (K,)
        e_t = (z_t - l_sub_t * e_next) / l_diag_t
        return e_t, e_t

    xs_rev = (Z[:, N - 2::-1].T,       # (N-1, K)  z values t = N-2..0
              L_sub[:, N - 2::-1].T,   # (N-1, K)
              L_diag[:, N - 2::-1].T)  # (N-1, K)
    _, e_rev = jax.lax.scan(back_step, e_last, xs_rev)
    # e_rev: (N-1, K) — [e[:,N-2], e[:,N-3], ..., e[:,0]]

    E = jnp.concatenate([e_rev[::-1], e_last[None, :]], axis=0).T  # (K, N)
    return E.reshape(-1)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def getMAP_jax(dat, hyper, n_params, log_lik_fns,
               E0=None, method=None, showOpt=0, pbar=None, map_tol=1e-6):
    """MAP estimation using JAX L-BFGS in prior-whitened space.

    The inner optimisation loop runs entirely in JAX (GPU-native) in a
    coordinate system where the random-walk prior has identity covariance.
    This gives L-BFGS a well-conditioned initial Hessian approximation and
    prevents the large early steps that can send parameters into sentinel
    territory.  A smooth repulsive barrier provides a non-zero gradient in
    sentinel territory so that L-BFGS never gets permanently trapped there.

    Args / Returns: same as psytrax._map.getMAP
    """
    from psytrax._map import getPosteriorTerms, _JAX_DTYPE
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
        session_starts = np.cumsum(day_lengths, dtype=int)[:-1]
    else:
        session_starts = np.array([], dtype=int)

    sigma_k   = jnp.broadcast_to(jnp.asarray(sigma,   dtype=dtype), (K,))
    sigInit_k = jnp.broadcast_to(jnp.asarray(sigInit, dtype=dtype), (K,))

    if sigDay is not None and len(session_starts) > 0:
        sigDay_k = jnp.broadcast_to(jnp.asarray(sigDay, dtype=dtype), (K,))
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

    # ---- prior-whitening Cholesky (computed once, baked into JIT as constants) ----
    L_diag, L_sub = _prior_chol(K, N, sigma_k, sigInit_k, is_boundary, sigDay_k)

    # ---- JIT-compiled objective in whitened z-space ----
    @jax.jit
    def neg_log_post_z(z_flat):
        """Penalised neg-log-posterior in whitened z = L^T e space.

        Includes a smooth repulsive barrier that activates in sentinel territory
        (mean log-lik per trial < -50 nats) to give L-BFGS a non-zero gradient
        there.  The barrier is negligible in valid parameter territory.
        """
        E_flat = _unwhiten(z_flat, K, N, L_diag, L_sub).astype(dtype)
        E      = jnp.reshape(E_flat, (K, N))
        logli  = log_likelihood_fn(E, dat_jax)
        lp     = _log_prior_jax(E_flat, K, N, sigma_k, sigInit_k, is_boundary, sigDay_k)
        # Barrier: sigmoid activates near 0 when logli/N << -50 (sentinel territory),
        # and stays ≈0 in valid territory (logli/N typically > -5).
        sentinel_weight = jax.nn.sigmoid(-logli / N - 50.0)
        repulsive = sentinel_weight * jnp.sum(z_flat ** 2) * 1e-4
        return -(logli + lp) + repulsive

    @jax.jit
    def neg_log_post_exact(z_flat):
        """Unpenalised neg-log-posterior (used for validity check and final evaluation)."""
        E_flat = _unwhiten(z_flat, K, N, L_diag, L_sub).astype(dtype)
        E      = jnp.reshape(E_flat, (K, N))
        logli  = log_likelihood_fn(E, dat_jax)
        lp     = _log_prior_jax(E_flat, K, N, sigma_k, sigInit_k, is_boundary, sigDay_k)
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

    z0_flat = _whiten(E0_flat, K, N, L_diag, L_sub)

    # ---- run L-BFGS via optax ----
    try:
        import optax
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "psytrax.fit requires the optional optimizer dependency 'optax'. "
            "Install psytrax with `pip install -r requirements.txt` for the app, "
            "or `pip install -e .[dev]` after updating package metadata."
        ) from exc

    def _run_lbfgs(z0, tol, max_iter=2000):
        solver = optax.lbfgs(memory_size=20, scale_init_precond=True)
        value_and_grad_fn = optax.value_and_grad_from_state(neg_log_post_z)

        opt_state = solver.init(z0)
        z_cur = z0

        @jax.jit
        def step(z, state):
            value, grad = value_and_grad_fn(z, state=state)
            updates, new_state = solver.update(
                grad, state, z, value=value, grad=grad,
                value_fn=neg_log_post_z,
            )
            return optax.apply_updates(z, updates), new_state, value, grad

        grad_norm = float('inf')
        for _ in range(max_iter):
            z_cur, opt_state, val, grad = step(z_cur, opt_state)
            grad_norm = float(jnp.max(jnp.abs(grad)))
            if grad_norm < tol:
                break
        return z_cur, grad_norm

    z_current, grad_norm = _run_lbfgs(z0_flat, map_tol)

    # ---- validity check: retry if sentinel values still dominate ----
    total_ll = float(-neg_log_post_exact(z_current))
    if total_ll < -N * 100:
        if showOpt:
            print(f'  WARNING: L-BFGS result has sentinel values (ll={total_ll:.2e}), retrying...')
        z_current, grad_norm = _run_lbfgs(z0_flat, map_tol * 1e-2, max_iter=5000)

    if showOpt:
        print(f'  JAX L-BFGS: final grad norm = {grad_norm:.2e}')

    # Convert back to e-space for Hessian computation
    E_current = _unwhiten(z_current, K, N, L_diag, L_sub)
    eMode = np.array(E_current, dtype=np.float64)

    # ---- Hessian + Laplace evidence (numpy/scipy, cheap one-time cost) ----
    pT, lT = getPosteriorTerms(eMode, dat, hyper, log_lik_fns, method=None)

    hess = {
        'H':          lT['ddlogli']['H'],
        'K':          lT['ddlogli']['K'],
        'ddlogprior': pT['ddlogprior'],
    }
    center       = -pT['ddlogprior'] - lT['ddlogli']['H']
    logterm_post = 0.5 * sparse_logdet(center)
    logEvd       = float(lT['logli']) + float(pT['logprior']) - logterm_post

    llstruct = {'lT': lT, 'pT': pT, 'eMode': eMode}
    return hess, logEvd, llstruct

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
from contextlib import nullcontext
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

_INVALID_LL_THRESHOLD_PER_TRIAL = -100.0


# ---------------------------------------------------------------------------
# JAX-traceable Gaussian random-walk log-prior
# ---------------------------------------------------------------------------

def _log_prior_jax(E_flat, K, N, sigma_k, sigInit_k, is_boundary, sigDay_k,
                   v_mean=None):
    """Gaussian random-walk log-prior, fully JAX-traceable.

    Args:
        E_flat      : (K*N,) parameter vector (C order, so E[k,t] = E_flat[k*N+t])
        K, N        : static Python ints
        sigma_k     : (K,) process noise std
        sigInit_k   : (K,) initial uncertainty std
        is_boundary : (N-1,) bool mask — True where session boundary occurs,
                      or None if no session boundaries
        sigDay_k    : (K,) session-boundary noise std (ignored if is_boundary is None)
        v_mean      : (K, N-1) learning-rule mean shift for each transition,
                      or None for a zero-mean random walk.  When provided the
                      transition model becomes  w_{t+1} − w_t ∼ N(v_mean[:,t], σ²).
    """
    E = jnp.reshape(E_flat, (K, N))

    # Initial term: E[:, 0] ~ N(0, sigInit_k^2)
    lp = jnp.sum(-0.5 * (E[:, 0] / sigInit_k) ** 2 - jnp.log(sigInit_k))

    if N == 1:
        return lp

    dE = E[:, 1:] - E[:, :-1]                          # (K, N-1)

    # Subtract the deterministic learning-rule contribution
    if v_mean is not None:
        dE = dE - v_mean

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

def _make_vmap_axes(x, N):
    """Recursively build JAX vmap in_axes for a pytree (local copy)."""
    if x is None:
        return None
    if isinstance(x, dict):
        return {k: _make_vmap_axes(v, N) for k, v in x.items()}
    arr = jnp.asarray(x) if not hasattr(x, 'shape') else x
    if arr.ndim > 0 and arr.shape[0] == N:
        return 0
    return None


def _compute_lr_hat_numpy(eMode, dat, learning_rule, K, N, dtype=None):
    """Compute raw learning-rule outputs at a numpy MAP estimate.

    Returns lr_hat as a numpy (K, N-1) array.  Used for the evidence
    computation and the decoupled Laplace step (both in numpy land).
    """
    if dtype is None:
        dtype = jnp.float64
    if N <= 1:
        return np.zeros((K, 0))

    E_jax = jnp.asarray(eMode.reshape(K, N), dtype=dtype)

    def _cast(x):
        if isinstance(x, np.ndarray) and np.issubdtype(x.dtype, np.floating):
            return jnp.asarray(x, dtype=dtype)
        elif isinstance(x, np.ndarray):
            return jnp.asarray(x)
        return x

    dat_jax = jax.tree_util.tree_map(_cast, dat)

    def _slice(x):
        if isinstance(x, jnp.ndarray) and x.ndim > 0 and x.shape[0] == N:
            return x[:-1]
        return x

    dat_lr = jax.tree_util.tree_map(_slice, dat_jax)
    N_lr = N - 1
    in_axes_lr = (1, _make_vmap_axes(dat_lr, N_lr))
    lr_out = jax.vmap(learning_rule, in_axes=in_axes_lr)(E_jax[:, :-1], dat_lr)  # (N-1, K)
    return np.asarray(lr_out.T, dtype=np.float64)  # (K, N-1)


def _compute_v_mean_jax(E, learning_rule, dat_jax, alpha_k, K, N):
    """Compute the learning-rule prior-mean shift v_mean (K, N-1).

    v_mean[:, t] = alpha_k * learning_rule(E[:, t], dat_trial_t)
    for t = 0, …, N-2.  This is the mean for transition t → t+1, i.e. it is
    subtracted from dE[:, t] = E[:, t+1] − E[:, t] in the log-prior.
    """
    if N <= 1:
        return jnp.zeros((K, 0), dtype=E.dtype)

    # Slice dat to first N-1 trials (trials that generate an update)
    def _slice(x):
        if isinstance(x, jnp.ndarray) and x.ndim > 0 and x.shape[0] == N:
            return x[:-1]
        return x

    dat_lr = jax.tree_util.tree_map(_slice, dat_jax)
    N_lr = N - 1
    in_axes_lr = (1, _make_vmap_axes(dat_lr, N_lr))
    lr_out = jax.vmap(learning_rule, in_axes=in_axes_lr)(E[:, :-1], dat_lr)  # (N-1, K)
    return (alpha_k[None, :] * lr_out).T   # (K, N-1)


def getMAP_jax(dat, hyper, n_params, log_lik_fns,
               E0=None, method=None, showOpt=0, pbar=None, map_tol=1e-6,
               execution_plan=None, status_callback=None,
               learning_rule=None):
    """MAP estimation using JAX L-BFGS in prior-whitened space.

    The inner optimisation loop runs entirely in JAX (GPU-native) in a
    coordinate system where the random-walk prior has identity covariance.
    This gives L-BFGS a well-conditioned initial Hessian approximation and
    prevents the large early steps that can send parameters into sentinel
    territory.  A smooth repulsive barrier provides a non-zero gradient in
    sentinel territory so that L-BFGS never gets permanently trapped there.

    When a ``learning_rule`` is provided, the Gaussian random-walk transition
    becomes  w_{t+1} − w_t ∼ N(v_t, diag(σ²))  where
    v_t = diag(α) · learning_rule(w_t, data_t).  The learning rates α_k
    are read from ``hyper['alpha']``.

    Args / Returns: same as psytrax._map.getMAP
    """
    from psytrax._map import getPosteriorTerms, _JAX_DTYPE
    import psytrax._map as _map_module
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

    # ---- learning rule setup ----
    has_lr = learning_rule is not None
    if has_lr:
        alpha = hyper.get('alpha')
        if alpha is None:
            raise ValueError("hyper must contain 'alpha' when a learning_rule is provided")
        alpha_k = jnp.broadcast_to(jnp.asarray(alpha, dtype=dtype), (K,))
    else:
        alpha_k = None

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

        # Compute learning-rule prior mean shift
        if has_lr:
            v_mean = _compute_v_mean_jax(E, learning_rule, dat_jax, alpha_k, K, N)
        else:
            v_mean = None

        lp = _log_prior_jax(E_flat, K, N, sigma_k, sigInit_k, is_boundary, sigDay_k,
                            v_mean=v_mean)
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

        if has_lr:
            v_mean = _compute_v_mean_jax(E, learning_rule, dat_jax, alpha_k, K, N)
        else:
            v_mean = None

        lp = _log_prior_jax(E_flat, K, N, sigma_k, sigInit_k, is_boundary, sigDay_k,
                            v_mean=v_mean)
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
        _emit_status(status_callback, "Compiling MAP objective…", stage="map")
        for step_idx in range(max_iter):
            z_cur, opt_state, val, grad = step(z_cur, opt_state)
            grad_norm = float(jnp.max(jnp.abs(grad)))
            if pbar is not None and (step_idx == 0 or (step_idx + 1) % 10 == 0):
                pbar.set_postfix({'MAP loss': f'{float(val):.3f}'})
            if step_idx == 0:
                _emit_status(status_callback, "Running MAP iterations…", stage="map")
            if grad_norm < tol:
                break
        return z_cur, grad_norm

    z_current, grad_norm = _run_lbfgs(z0_flat, map_tol)

    # ---- validity check: retry if sentinel values still dominate ----
    total_ll = float(-neg_log_post_exact(z_current))
    if total_ll < -N * 100:
        if showOpt:
            print(f'  WARNING: L-BFGS result has sentinel values (ll={total_ll:.2e}), retrying...')
        _emit_status(status_callback, "Retrying MAP with tighter tolerance…", stage="retry")
        z_current, grad_norm = _run_lbfgs(z0_flat, map_tol * 1e-2, max_iter=5000)
        total_ll = float(-neg_log_post_exact(z_current))

    _raise_if_invalid_solution(total_ll, N)

    if showOpt:
        print(f'  JAX L-BFGS: final grad norm = {grad_norm:.2e}')

    # Convert back to e-space for Hessian computation
    E_current = _unwhiten(z_current, K, N, L_diag, L_sub)
    eMode = np.array(E_current, dtype=np.float64)

    # ---- Hessian + Laplace evidence (numpy/scipy, cheap one-time cost) ----
    evidence_precision = getattr(execution_plan, "evidence_precision", "float64")
    evidence_backend = getattr(execution_plan, "evidence_backend", "cpu")
    evidence_dtype = jnp.float32 if evidence_precision == "float32" else jnp.float64
    prev_dtype = _map_module._JAX_DTYPE
    ctx = nullcontext()
    try:
        devs = jax.devices(evidence_backend)
        if devs:
            ctx = jax.default_device(devs[0])
    except Exception:
        pass

    try:
        _map_module._JAX_DTYPE = evidence_dtype
        _emit_status(status_callback, "Computing Hessian and Laplace evidence…", stage="evidence")
        with ctx:
            pT, lT = getPosteriorTerms(eMode, dat, hyper, log_lik_fns, method=None)
    finally:
        _map_module._JAX_DTYPE = prev_dtype

    # Correct the log-prior if a learning rule shifts the prior mean
    if has_lr:
        from psytrax._helper.helperFunctions import (
            build_v_mean_flat, correct_logprior_for_learning_rule, make_invSigma,
        )
        day_lengths_np = dat.get('dayLength', np.array([], dtype=int))
        if len(day_lengths_np) > 0:
            days_arr = np.cumsum(day_lengths_np, dtype=int)[:-1]
        else:
            days_arr = np.array([], dtype=int)
        invSigma = make_invSigma(hyper, days_arr, dat.get('missing_trials'), N, K)

        lr_hat = _compute_lr_hat_numpy(eMode, dat, learning_rule, K, N, dtype=evidence_dtype)
        v_mean_flat = build_v_mean_flat(lr_hat, np.asarray(hyper['alpha']), K, N)
        pT['logprior'] = correct_logprior_for_learning_rule(
            pT['logprior'], eMode, v_mean_flat, invSigma, K, N,
        )

    hess = {
        'H':          lT['ddlogli']['H'],
        'K':          lT['ddlogli']['K'],
        'ddlogprior': pT['ddlogprior'],
    }
    center       = -pT['ddlogprior'] - lT['ddlogli']['H']
    logterm_post = 0.5 * sparse_logdet(center)
    logEvd       = float(lT['logli']) + float(pT['logprior']) - logterm_post
    _raise_if_invalid_solution(float(lT['logli']) + float(pT['logprior']), N)
    if not np.isfinite(logEvd):
        raise RuntimeError(
            "psytrax.fit produced a non-finite log-evidence. "
            "This usually indicates numerical instability in the MAP/Hessian step."
        )

    llstruct = {'lT': lT, 'pT': pT, 'eMode': eMode}
    if has_lr:
        llstruct['lr_hat'] = lr_hat  # (K, N-1) raw learning-rule outputs at MAP
    return hess, logEvd, llstruct


def _raise_if_invalid_solution(total_ll, n_trials):
    """Raise when optimization lands in a model-sentinel region."""
    if np.isfinite(total_ll) and total_ll >= n_trials * _INVALID_LL_THRESHOLD_PER_TRIAL:
        return

    n_sentinel_est = "unknown"
    if np.isfinite(total_ll):
        n_sentinel_est = int(min(n_trials, max(1, round(-total_ll / 1e12))))
    raise RuntimeError(
        "psytrax.fit converged to an invalid parameter region and produced an implausibly "
        f"low objective ({total_ll:.3e}). This usually means sentinel log-likelihood values "
        f"are still dominating the fit (estimated invalid trials: {n_sentinel_est}). "
        "Try the automatic warm-start (E0=None), a better model-specific E0, or looser "
        "initial hyperparameters."
    )


def _emit_status(callback, message, stage=None, **extra):
    if callback is None:
        return
    payload = {"message": message}
    if stage is not None:
        payload["stage"] = stage
    payload.update(extra)
    callback(payload)

from scipy.sparse.linalg import splu
from scipy.sparse import isspmatrix_csc, diags, eye
import numpy as np


def myblk_diags(A):
    """Convert (N, K, K) array into a sparse (N*K, N*K) block-diagonal matrix."""
    N, K, _ = np.shape(A)
    d = np.zeros((2 * K - 1, N * K))
    offsets = np.hstack((np.arange(K), np.arange(-K + 1, 0))) * N
    for i in range(K):
        for j in range(K):
            m = np.min([i, j])
            d[j - i, m * N:(m + 1) * N] = A[:, i, j]
    return diags(d, offsets, shape=(N * K, N * K), format='csc')


def sparse_logdet(A):
    """Log determinant of a sparse CSC matrix via LU decomposition."""
    if not isspmatrix_csc(A):
        raise Exception('sparse_logdet: matrix must be in sparse CSC format')

    # Hyperparameter searches can briefly visit nearly singular curvature
    # matrices. Add progressively larger diagonal jitter before giving up.
    ridges = (0.0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2)
    last_error = None
    for ridge in ridges:
        try:
            aux = splu(A if ridge == 0.0 else A + ridge * eye(A.shape[0], format='csc'))
            break
        except RuntimeError as exc:
            last_error = exc
    else:
        raise RuntimeError(
            "Posterior Hessian remained exactly singular after adaptive ridge regularization."
        ) from last_error

    return np.sum(
        np.log(np.abs(aux.L.diagonal())) + np.log(np.abs(aux.U.diagonal())))


def make_invSigma(hyper, days, missing_trials, N, K):
    """Build the inverse prior covariance (random-walk) matrix.

    Args:
        hyper: dict with 'sigma', optionally 'sigInit' and 'sigDay'
        days: array of trial indices that start a new session
        missing_trials: boolean array of length N marking held-out trials, or None
        N: number of (effective) trials
        K: number of parameters

    Returns:
        sparse diagonal matrix (N*K, N*K)
    """
    sigma = hyper['sigma']
    sigInit = hyper['sigInit'] if hyper.get('sigInit') is not None else sigma
    sigDay = hyper['sigDay'] if hyper.get('sigDay') is not None else sigma
    sigma = _broadcast_hyper_vector(sigma, K, 'sigma')
    sigInit = _broadcast_hyper_vector(sigInit, K, 'sigInit')
    sigDay = _broadcast_hyper_vector(sigDay, K, 'sigDay')

    flat = np.zeros(N * K)
    for k in range(K):
        flat[k * N:(k + 1) * N] = sigma[k] ** 2
        flat[k * N + days] = sigDay[k] ** 2
        flat[k * N] = sigInit[k] ** 2
        if missing_trials is not None:
            flat[k * N:(k + 1) * N] += missing_trials * sigma[k] ** 2
    return diags(flat ** -1)


def _broadcast_hyper_vector(value, K, name):
    """Broadcast a scalar hyperparameter to length K or validate a vector."""
    if np.isscalar(value):
        return np.full(K, float(value))

    arr = np.asarray(value, dtype=float)
    if arr.shape != (K,):
        raise Exception(f'{name} must be scalar or have shape ({K},), got {arr.shape}')
    return arr


def trim(dat, START=0, END=0):
    """Slice a dataset to [START, END) trials, keeping session info intact."""
    if not START and not END:
        return dat

    N = len(dat['r'])
    if START < 0:
        START = N + START
    if END <= 0:
        END = N + END
    if END > N:
        END = N
    if START >= END:
        raise Exception(f'START >= END: {START}, {END}')

    new_dat = {}
    for k in dat.keys():
        if k == 'inputs':
            continue
        try:
            if dat[k] is not None and np.asarray(dat[k]).shape[0] == N:
                new_dat[k] = dat[k][START:END].copy()
            else:
                new_dat[k] = dat[k]
        except Exception:
            new_dat[k] = dat[k]

    new_dat['inputs'] = {i: dat['inputs'][i][START:END] for i in dat['inputs']}

    if 'dayLength' in new_dat and new_dat['dayLength'] is not None and new_dat['dayLength'].size:
        cumdays = np.cumsum(new_dat['dayLength'])
        starts = np.concatenate(([0], cumdays[:-1]))
        keep = np.where((starts < END) & (cumdays > START))[0]
        clipped_start = np.maximum(starts[keep], START)
        clipped_end = np.minimum(cumdays[keep], END)
        new_dat['dayLength'] = (clipped_end - clipped_start).astype(int)

    new_dat['skimmed'] = {'START': START, 'END': END}
    return new_dat


def DT_X_D(ddlogprior, K):
    """Compute D.T @ X @ D using the block difference matrix structure."""
    dd = ddlogprior.diagonal().reshape((K, -1)).copy()
    main_diag = dd.copy()
    main_diag[:, :-1] += main_diag[:, 1:]
    main_diag = main_diag.flatten()
    off_diags = dd.copy()
    off_diags[:, 0] = 0
    off_diags = -off_diags.flatten()[1:]
    NK = main_diag.shape[0]
    A = np.zeros((3, NK))
    A[0] = main_diag
    A[1, :-1] = off_diags
    A[2, :-1] = off_diags
    return diags(A, [0, -1, 1], shape=(NK, NK), format='csc')


# ---------------------------------------------------------------------------
# Learning-rule prior-mean helpers
# ---------------------------------------------------------------------------

def build_v_mean_flat(lr_hat, alpha, K, N):
    """Build the learning-rule mean vector for the Gaussian walk.

    The random-walk transition model is:

        w_{t+1} − w_t  ∼  N(v_t, diag(σ²))

    where  v_t = diag(α) · lr_hat_t.  This function stacks the v_t values
    into the (K*N,) layout that matches the ``invSigma`` matrix:

        v[k*N + 0] = 0          (no mean shift on the initial state)
        v[k*N + t] = α_k · lr_hat[k, t-1]    for t = 1, …, N−1

    Args:
        lr_hat : (K, N-1) raw learning-rule outputs at each trial
        alpha  : scalar or (K,) learning rates
        K, N   : parameter count and trial count

    Returns:
        (K*N,) flat vector
    """
    alpha = _broadcast_hyper_vector(alpha, K, 'alpha')
    v = np.zeros((K, N))
    v[:, 1:] = alpha[:, None] * lr_hat      # lr_hat[:, t] drives transition t → t+1
    return v.flatten()


def build_prior_mean_flat(init_mean, K, N):
    """Build a flat absolute prior-mean trajectory.

    ``init_mean`` may be a length-K vector, a full ``(K, N)`` trajectory, or an
    already-flat ``(K*N,)`` vector.  A length-K vector is repeated across trials,
    which corresponds to a non-zero mean for the initial state and zero-mean
    random-walk transitions.
    """
    if init_mean is None:
        return None

    arr = np.asarray(init_mean, dtype=float)
    if arr.shape == (K,):
        mean = np.tile(arr[:, None], (1, N))
    elif arr.shape == (K, 1):
        mean = np.tile(arr, (1, N))
    elif arr.shape == (K, N):
        mean = arr
    elif arr.shape == (K * N,):
        mean = arr.reshape(K, N)
    else:
        raise ValueError(
            f'init_mean must have shape ({K},), ({K}, {N}), or ({K*N},), '
            f'got {arr.shape}'
        )
    if not np.all(np.isfinite(mean)):
        raise ValueError('init_mean must contain only finite values')
    return mean.flatten()


def compute_invC_mean(mean_flat, invSigma, K, N):
    """Compute the natural-parameter contribution Q @ mean_flat."""
    if mean_flat is None:
        return np.zeros(K * N)
    return DT_X_D(invSigma, K) @ mean_flat


def compute_invC_u(v_mean_flat, invSigma, K, N):
    """Compute  invC @ u  =  D^T  Σ⁻¹  v   without forming invC explicitly.

    This is the "natural-parameter" contribution of the prior mean to the
    posterior precision equation.  It is used by the decoupled Laplace
    approximation when a learning rule shifts the prior mean.

    D is the N×N first-difference matrix (identity on the diagonal, −1 on
    the sub-diagonal) applied block-independently to each of the K parameters.

    Args:
        v_mean_flat : (K*N,) output of :func:`build_v_mean_flat`
        invSigma    : sparse diagonal (K*N, K*N) from :func:`make_invSigma`
        K, N        : parameter count and trial count

    Returns:
        (K*N,) vector  D^T Σ⁻¹ v
    """
    inv_sig_diag = invSigma.diagonal()
    Sv = inv_sig_diag * v_mean_flat          # Σ⁻¹ v  (element-wise)
    Sv = Sv.reshape(K, N)

    # Apply D^T per parameter block:
    #   (D^T y)_t = y_t − y_{t+1}   for t < N-1
    #   (D^T y)_{N-1} = y_{N-1}
    result = np.zeros_like(Sv)
    result[:, :-1] = Sv[:, :-1] - Sv[:, 1:]
    result[:, -1] = Sv[:, -1]
    return result.flatten()


def correct_logprior_for_learning_rule(logprior_base, E_flat, v_mean_flat,
                                       invSigma, K, N, prior_mean_flat=None):
    """Correct a base log-prior to account for a learning-rule mean shift.

    Given the base log-prior  −½ (D(E−m))^T Σ⁻¹ (D(E−m)) − ½ log|C|  and the
    learning-rule mean vector *v*, return the shifted version:

        −½ (D(E−m) − v)^T Σ⁻¹ (D(E−m) − v)  − ½ log|C|

    The correction equals  (D(E−m))^T Σ⁻¹ v  −  ½ v^T Σ⁻¹ v.

    Args:
        logprior_base      : scalar, the log-prior computed with v = 0
        E_flat             : (K*N,) parameter vector
        v_mean_flat        : (K*N,) from :func:`build_v_mean_flat`
        invSigma           : sparse diagonal from :func:`make_invSigma`
        K, N               : parameter count and trial count
        prior_mean_flat    : optional (K*N,) absolute prior-mean trajectory

    Returns:
        scalar — corrected log-prior value
    """
    inv_sig_diag = invSigma.diagonal()

    # Compute DE (first differences, matching the invSigma layout)
    centered = E_flat if prior_mean_flat is None else E_flat - prior_mean_flat
    E = centered.reshape(K, N)
    DE = np.zeros((K, N))
    DE[:, 0] = E[:, 0]                 # initial state
    DE[:, 1:] = E[:, 1:] - E[:, :-1]   # transition differences
    DE_flat = DE.flatten()

    cross_term = np.dot(DE_flat * inv_sig_diag, v_mean_flat)  # (DE)^T Σ⁻¹ v
    v_quad     = np.dot(v_mean_flat * inv_sig_diag, v_mean_flat)  # v^T Σ⁻¹ v
    return logprior_base + cross_term - 0.5 * v_quad

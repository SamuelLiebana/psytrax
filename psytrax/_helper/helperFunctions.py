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
        min_id = np.where(cumdays > START)[0][0]
        max_id = np.where(cumdays < END)[0][-1] + 1
        new = new_dat['dayLength'][min_id:max_id + 1].copy()
        new[0] = cumdays[min_id] - START
        new[-1] = END - cumdays[max_id - 1]
        if len(new) == 1:
            new[0] = END - START
        new_dat['dayLength'] = new

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

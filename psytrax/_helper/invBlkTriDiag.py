import numpy as np
from scipy.sparse import csr_matrix, isspmatrix
from psytrax._helper.helperFunctions import DT_X_D


def getCredibleInterval(Hess):
    return np.sqrt(invDiagHess(Hess)).reshape(Hess['K'], -1)


def invDiagHess(Hess):
    """Diagonal of the inverse posterior Hessian via block-tridiagonal inversion."""
    center = -(Hess['ddlogprior'] + Hess['H'])
    K = Hess['K']
    N = int(Hess['ddlogprior'].shape[0] / K)
    ii = (np.reshape(np.arange(K * N), (N, -1), order='F').T).flatten(order='F')
    M = center[ii][:, ii]
    vdiag, _, _ = invBlkTriDiag(M, K)
    return vdiag[np.argsort(ii)]


def invBlkTriDiag(M, nn):
    """Invert a sparse block-tridiagonal matrix efficiently.

    Args:
        M: sparse square matrix with (nn, nn) blocks on the tri-diagonal
        nn: block size
    Returns:
        MinvDiag: diagonal of M^{-1}
        MinvBlocks: diagonal blocks of M^{-1}
        MinvBelowDiagBlocks: below-diagonal blocks of M^{-1}
    """
    if not isspmatrix(M):
        M = csr_matrix(M)

    nblocks = int(M.shape[0] / nn)
    A = np.zeros((nn, nn, nblocks))
    B = np.zeros((nn, nn, nblocks))
    C = np.zeros((nn, nn, nblocks))
    D = np.zeros((nn, nn, nblocks))
    E = np.zeros((nn, nn, nblocks))

    inds0 = np.arange(nn)
    B[:, :, 0] = M[np.ix_(inds0, inds0)].todense()
    C[:, :, 0] = M[np.ix_(inds0, inds0 + nn)].todense()
    D[:, :, 0] = np.linalg.solve(B[:, :, 0], C[:, :, 0])

    inds_last = (nblocks - 1) * nn + inds0
    A[:, :, -1] = M[np.ix_(inds_last, inds_last - nn)].todense()
    B[:, :, -1] = M[np.ix_(inds_last, inds_last)].todense()
    E[:, :, -1] = np.linalg.solve(B[:, :, -1], A[:, :, -1])

    for ii in range(1, nblocks - 1):
        inds = inds0 + ii * nn
        A[:, :, ii] = M[np.ix_(inds, inds - nn)].todense()
        B[:, :, ii] = M[np.ix_(inds, inds)].todense()
        C[:, :, ii] = M[np.ix_(inds, inds + nn)].todense()

    for ii in range(1, nblocks - 1):
        D[:, :, ii] = np.linalg.solve(B[:, :, ii] - A[:, :, ii] @ D[:, :, ii - 1], C[:, :, ii])
        jj = nblocks - ii - 1
        E[:, :, jj] = np.linalg.solve(B[:, :, jj] - C[:, :, jj] @ E[:, :, jj + 1], A[:, :, jj])

    I = np.eye(nn)
    MinvBlocks = np.zeros((nn, nn, nblocks))
    MinvBelowDiagBlocks = np.zeros((nn, nn, nblocks - 1))

    MinvBlocks[:, :, 0] = np.linalg.inv(B[:, :, 0] @ (I - D[:, :, 0] @ E[:, :, 1]))
    MinvBlocks[:, :, -1] = np.linalg.inv(B[:, :, -1] - A[:, :, -1] @ D[:, :, -2])

    for ii in range(1, nblocks - 1):
        MinvBlocks[:, :, ii] = np.linalg.inv(
            (B[:, :, ii] - A[:, :, ii] @ D[:, :, ii - 1]) @ (I - D[:, :, ii] @ E[:, :, ii + 1]))
        MinvBelowDiagBlocks[:, :, ii - 1] = -D[:, :, ii - 1] @ MinvBlocks[:, :, ii]

    MinvBelowDiagBlocks[:, :, -1] = -D[:, :, -2] @ MinvBlocks[:, :, -1]

    MinvDiag = np.zeros(nn * nblocks)
    for ii in range(nblocks):
        MinvDiag[ii * nn:(ii + 1) * nn] = np.diag(MinvBlocks[:, :, ii])

    return MinvDiag, MinvBlocks, MinvBelowDiagBlocks

import numpy as np


def jacHessCheck(fun, x0, *args, **kwargs):
    """Check analytic Jacobian and Hessian against finite differences."""
    fun(x0, *args, **kwargs)
    JJ = fun.jacobian(x0, *args, **kwargs)

    tol = 1e-8
    randjump = np.random.rand(len(x0)) * tol
    f1 = fun(x0 - randjump / 2, *args, **kwargs)
    JJ1 = fun.jacobian(x0 - randjump / 2, *args, **kwargs)
    f2 = fun(x0 + randjump / 2, *args, **kwargs)
    JJ2 = fun.jacobian(x0 + randjump / 2, *args, **kwargs)

    print('Analytic Jac:', np.dot(randjump, JJ))
    print('Finite Jac:  ', f2 - f1)

    HH = fun.hessian(x0, *args, **kwargs)
    if isinstance(HH, dict):
        print('Analytic Hess:', np.sum(fun.hessian_prod(x0, randjump, *args, **kwargs)))
    else:
        print('Analytic Hess:', np.sum(HH @ randjump))
    print('Finite Hess:  ', np.sum(JJ2 - JJ1))


def jacEltsCheck(fun, ind, x0, *args, **kwargs):
    """Check individual Jacobian elements against finite differences."""
    fun(x0, *args, **kwargs)
    JJ = fun.jacobian(x0, *args, **kwargs)

    eps = 1e-5
    mask = np.zeros(len(x0))
    mask[ind] = eps
    dJ = (fun(x0 + mask, *args, **kwargs) - fun(x0 - mask, *args, **kwargs)) / 2 / eps

    if np.sqrt((JJ[ind] - dJ) ** 2) > 1e-8:
        print(ind, ': ', np.sqrt((JJ[ind] - dJ) ** 2))
        print('Analytic Jac:', JJ[ind])
        print('Finite Jac:  ', dJ)


def compHess(fun, x0, dx, kwargs):
    """Numerically estimate the Hessian of fun at x0 via central differences."""
    n = len(x0)
    H = np.zeros((n, n))
    g = np.zeros(n)
    f0 = fun(x0, **kwargs)
    A = np.diag(dx * np.ones(n) / 2.0)

    for j in range(n):
        f1 = fun(x0 + 2 * A[:, j], **kwargs)
        f2 = fun(x0 - 2 * A[:, j], **kwargs)
        H[j, j] = f1 + f2 - 2 * f0
        g[j] = (f1 - f2) / 2

    for j in range(n - 1):
        for i in range(j + 1, n):
            f11 = fun(x0 + A[:, j] + A[:, i], **kwargs)
            f22 = fun(x0 - A[:, j] - A[:, i], **kwargs)
            f12 = fun(x0 + A[:, j] - A[:, i], **kwargs)
            f21 = fun(x0 - A[:, j] + A[:, i], **kwargs)
            H[j, i] = H[i, j] = f11 + f22 - f12 - f21

    return H / dx / dx, g / dx

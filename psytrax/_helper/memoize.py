import numpy as np


class memoize(object):
    """
    Caches the result of a function along with its 1st and 2nd derivatives.
    SciPy's minimize() needs distinct objects for the jacobian + hessian. With
    memoize, a single function returning [output, jac, hess] can serve all three
    roles without redundant computation.

    See: http://stackoverflow.com/a/17431749
    Args:
        fun: callable returning (value, jacobian, hessian)
    """

    def __init__(self, fun):
        self.fun = fun
        self.value, self.jac, self.hess = None, None, None
        self.x = None

    def _compute(self, x, *args, **kwargs):
        self.x = np.asarray(x).copy()
        self.value, self.jac, self.hess = self.fun(x, *args, **kwargs)

    def __call__(self, x, *args, **kwargs):
        if self.value is not None and np.all(x == self.x):
            return self.value
        self._compute(x, *args, **kwargs)
        return self.value

    def jacobian(self, x, *args, **kwargs):
        if self.jac is not None and np.all(x == self.x):
            return self.jac
        self._compute(x, *args, **kwargs)
        return self.jac

    def hessian(self, x, *args, **kwargs):
        if self.hess is not None and np.all(x == self.x):
            return self.hess
        self._compute(x, *args, **kwargs)
        return self.hess

    def hessian_prod(self, x, p, *args, **kwargs):
        if self.hess is None or not np.all(x == self.x):
            self._compute(x, *args, **kwargs)
        negH = self.hess['negH']
        negddlogprior = self.hess['negddlogprior']
        return (negddlogprior + negH) @ p

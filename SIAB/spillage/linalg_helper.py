import numpy as np

def mrdiv(X, Y):
    '''
    Right matrix division X @ inv(Y).

    The shapes of X and Y must be broadcastable as supported by
    numpy.linalg.solve. If X is 1-d, Y must be 2-d and the result
    is a 1-d array. For multi-dimensional arrays (ndim >= 3), the
    calculation is performed slice-wise.
    
    Note
    ----
    The documentation of numpy.linalg.solve seems to suggest that it is
    allowed to call solve(A,b) with A having shape (..., M, M) and b having
    shape (M,). However, this does not work for A with 3 or more dimensions
    (at least for v1.24.2). That's why len(Y.shape) is restricted to be 2
    when len(X.shape) is 1 in the assert statement below. This needs to be
    investigated further.

    '''
    assert (len(X.shape) == 1 and len(Y.shape) == 2) \
            or (len(X.shape) > 1 and len(Y.shape) > 1)

    return np.linalg.solve(Y.swapaxes(-2, -1), X.swapaxes(-2, -1)) \
            .swapaxes(-2, -1) \
            if len(X.shape) > 1 else np.linalg.solve(Y.T, X)


def rfrob(X, Y, rowwise=False):
    '''
    Real part of the Frobenius inner product.

    The Frobenius inner product between two matrices or vectors is defined as

        <X, Y> \equiv Tr(X @ Y.T.conj()) = (X * Y.conj()).sum()

    X and Y must have shapes compatible with element-wise multiplication.
    By default rowwise=False, in which case the inner product is computed
    slice-wise, i.e., sum() is taken over the last two axes. If rowwise is
    True, sum() is taken over the last axis only.

    Notes
    -----
    The inner product is assumed to have the Hermitian conjugate on the
    second argument, not the first.

    '''
    return (X * Y.conj()).real.sum(-1 if rowwise else (-2,-1))


############################################################
#                           Test
############################################################
import unittest

class _TestLinalgHelper(unittest.TestCase):

    def test_mrdiv(self):
        '''
        checks mrdiv with orthogonal matrices

        '''
        nk = 3
        m = 5
        n = 6

        X = np.random.randn(nk, m, n) + 1j * np.random.randn(nk, m, n)

        # have each slice of Y unitary to make it easier to verify
        Y = np.random.randn(nk, n, n) + 1j * np.random.randn(nk, n, n)
        Y = np.linalg.qr(Y)[0]

        # check that mrdiv works for broadcastable multiple slices
        Z = mrdiv(X, Y)
        self.assertEqual(Z.shape, X.shape)
        for i in range(nk):
            self.assertTrue(np.allclose(Z[i], X[i] @ Y[i].T.conj()))

        # check that mrdiv works for a single slice
        for k in range(nk):
            self.assertTrue(np.allclose(mrdiv(X[k], Y[k]),
                                        X[k] @ Y[k].T.conj()))

        # check that mrdiv works when X is 1-d and Y is 2-d
        X = np.random.randn(n) + 1j * np.random.randn(n)
        for k in range(nk):
            self.assertTrue(np.allclose(mrdiv(X, Y[k]),
                                        X @ Y[k].T.conj()))


    def test_rfrob(self):
        nk = 5
        m = 3
        n = 4
        X = np.random.randn(nk, m, n) + 1j * np.random.randn(nk, m, n)
        Y = np.random.randn(nk, m, n) + 1j * np.random.randn(nk, m, n)

        res = [np.trace(X[k] @ Y[k].T.conj()).real for k in range(nk)]
        self.assertTrue(np.allclose(rfrob(X, Y), res))

        res = [[(X[k,i] @ Y[k,i].T.conj()).real
                for i in range(m)]
               for k in range(nk)]
        self.assertTrue(np.allclose(rfrob(X, Y, rowwise=True), res))


if __name__ == '__main__':
    unittest.main()


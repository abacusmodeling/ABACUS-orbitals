import numpy as np

def mrdiv(X, Y):
    '''
    Right matrix division.

    Given two 3-d arrays X and Y, returns a 3-d array Z such that

        Z[k] = X[k] @ inv(Y[k])

    '''
    assert len(X.shape) == 3 and len(Y.shape) == 3
    return np.array([np.linalg.solve(Yk.T, Xk.T).T for Xk, Yk in zip(X, Y)])


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

        # make each slice of S unitary to make it easier to verify
        Y = np.random.randn(nk, n, n) + 1j * np.random.randn(nk, n, n)
        Y = np.linalg.qr(Y)[0]

        X = np.random.randn(nk, m, n) + 1j * np.random.randn(nk, m, n)
        Z = mrdiv(X, Y)

        self.assertEqual(Z.shape, X.shape)
        for i in range(nk):
            self.assertTrue( np.allclose(Z[i], X[i] @ Y[i].T.conj()) )


    def test_rfrob(self):
        nk = 5
        m = 3
        n = 4
        w = np.random.randn(nk)
        X = np.random.randn(nk, m, n) + 1j * np.random.randn(nk, m, n)
        Y = np.random.randn(nk, m, n) + 1j * np.random.randn(nk, m, n)

        wsum = 0.0
        for wk, Xk, Yk in zip(w, X, Y):
            wsum += wk * np.trace(Xk @ Yk.T.conj()).sum()

        self.assertAlmostEqual(w @ rfrob(X, Y), wsum.real)

        wsum = np.zeros(m, dtype=complex)
        for i in range(m):
            for k in range(nk):
                wsum[i] += w[k] * (X[k,i] @ Y[k,i].T.conj())

        self.assertTrue(np.allclose(w @ rfrob(X, Y, rowwise=True),
                                    wsum.real))


if __name__ == '__main__':
    unittest.main()


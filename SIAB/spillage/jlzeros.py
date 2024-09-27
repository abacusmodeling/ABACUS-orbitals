import numpy as np
from scipy.special import spherical_jn

def ikebe(l, nzeros):
    '''
    Returns the first few zeros of the l-th order spherical Bessel function
    by the method of Ikebe et al.
    
    Parameters
    ----------
        l : int
            Order of the spherical Bessel function.
        nzeros : int
            Number of zeros to be returned.
    
    Returns
    -------
        zeros : array
            The first n zeros of the l-th order spherical Bessel function.
    
    References
    ----------
        Ikebe, Y., Kikuchi, Y., & Fujishiro, I. (1991).
        Computing zeros and orders of Bessel functions.
        Journal of Computational and Applied Mathematics, 38(1-3), 169-184.
    
    '''
    from scipy.linalg import eigvalsh_tridiagonal

    nu = l + 0.5
    sz = nzeros*2 + l + 10

    alpha = nu + 2*np.arange(2, sz+1, dtype=int)

    A_diag = np.zeros(sz)
    A_diag[0] = 2. / ( (nu+3) * (nu+1) )
    A_diag[1:] = 2. / ( (alpha+1) * (alpha-1) )
    A_subdiag = 1. / ( (alpha-1) * np.sqrt(alpha*(alpha-2)) )

    eigval = eigvalsh_tridiagonal(A_diag, A_subdiag)[::-1]
    return 2. / np.sqrt(eigval[:nzeros])


def bracket(l, nzeros, return_all=False):
    '''
    Returns the first few zeros of the l-th order spherical Bessel
    function by iteratively using the bracketing method.
    
    The zeros of j_{l} and j_{l+1} are interlaced; so are
    the zeros of j_{l} and j_{l+2}. This property is exploited
    to find the zeros iteratively from the zeros of j_0.
    
    Parameters
    ----------
        l : int
            Order of the spherical Bessel function.
        nzeros : int
            Number of zeros to be returned.
        return_all : bool
            If True, all the zeros from j_0 to j_l will be returned.
    
    Returns
    -------
        array or list of array
            If return_all is False, an array that contains the first
            `nzeros` zeros of the l-th order spherical Bessel function
            is returned. Otherwise, a list of l+1 arrays is returned
            where the i-th array contains the zeros of the i-th order
            spherical Bessel function (i = 0, 1, ..., l).

    '''
    from scipy.optimize import brentq

    def _zerogen():
        ll = None # active l
        jl = lambda x: spherical_jn(ll, x)

        if return_all:
            nz = nzeros + l
            stride = 1
            l_start = 1
        else:
            # for odd  l: j_0 --> j_1 --> j_3 --> j_5 --> ... --> j_l
            # for even l: j_0 --> j_2 --> j_4 --> j_6 --> ... --> j_l
            nz = nzeros + (l+1)//2
            stride = 2
            l_start = 2 - l%2

        zeros = np.array([i * np.pi for i in range(1, nz+1)]) # zeros of j_0

        for ll in range(l_start, l+1, stride):
            return_all and (yield zeros[:nzeros])
            zeros = np.array([brentq(jl, zeros[i], zeros[i+1], xtol=1e-14)
                              for i in range(nz-1)])
            nz -= 1

        yield zeros[:nzeros]

    return list(_zerogen()) if return_all else next(_zerogen())


# tabulate some frequently used zeros
JLZEROS_LMAX = 20
JLZEROS_NZEROS = 100
JLZEROS = bracket(JLZEROS_LMAX, JLZEROS_NZEROS, return_all=True)

############################################################
#                       Test
############################################################
import unittest

class _TestJlZeros(unittest.TestCase):
    def test_ikebe(self):
        for l in range(20):
            for nzeros in range(1, 50):
                zeros = ikebe(l, nzeros)
                self.assertLess(np.linalg.norm(spherical_jn(l, zeros), np.inf),
                                1e-14)
    
    
    def test_bracket(self):
        for l in range(20):
            for nzeros in range(1, 5):
                zeros = bracket(l, nzeros, return_all=False)
                self.assertLess(np.linalg.norm(spherical_jn(l, zeros), np.inf),
                                1e-14)

        lmax = 20
        nzeros = 100
        zeros = bracket(lmax, nzeros, return_all=True)
        for l in range(lmax+1):
            self.assertLess(np.linalg.norm(spherical_jn(l, zeros[l]), np.inf),
                            1e-14)


if __name__ == '__main__':
    unittest.main()


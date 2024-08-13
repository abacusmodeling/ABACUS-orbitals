from SIAB.spillage.radial import jl_reduce, jl_raw_norm

import numpy as np

def coeff_reduced2raw(coeff, rcut):
    '''
    Converts coefficients w.r.t. reduced spherical Bessel functions 
    to those w.r.t raw truncated spherical Bessel functions.

    Parameters
    ----------
        coeff: nested list
            coefficients organized as coeff[l][zeta][q] -> float
        rcut : int or float
            Cutoff radius.

    Note
    ----
    Items of the same l must consist of the same number of basis functions.

    '''
    coeff_basis = [np.array(coeff_l).T for coeff_l in coeff]

    # we need to check if the array is empty because array([]) @ array([])
    # would give 0.0 while we actually want np.array([])
    return [(jl_reduce(l, coeff_l.shape[0] + 1, rcut, True) @ coeff_l)
            .T.tolist() if coeff_l.size > 0 else []
            for l, coeff_l in enumerate(coeff_basis)]


def coeff_raw2normalized(coeff, rcut):
    '''
    Converts coefficients w.r.t. raw truncated spherical Bessel functions
    to those w.r.t normalized truncated spherical Bessel functions.

    Parameters
    ----------
        coeff: nested list
            coefficients organized as coeff[l][zeta][q] -> float
        rcut : int or float
            Cutoff radius.

    '''
    return [[[coeff_lzq * jl_raw_norm(l, q, rcut)
              for q, coeff_lzq in enumerate(coeff_lz)]
             for coeff_lz in coeff_l]
            for l, coeff_l in enumerate(coeff)]


def coeff_normalized2raw(coeff, rcut):
    '''
    Converts coefficients w.r.t normalized truncated spherical Bessel
    functions to those w.r.t raw truncated spherical Bessel functions.

    Parameters
    ----------
        coeff: nested list
            coefficients organized as coeff[l][zeta][q] -> float
        rcut : int or float
            Cutoff radius.

    '''
    return [[[coeff_lzq / jl_raw_norm(l, q, rcut)
              for q, coeff_lzq in enumerate(coeff_lz)]
             for coeff_lz in coeff_l]
            for l, coeff_l in enumerate(coeff)]


############################################################
#                       Test
############################################################
import unittest
from numpy.linalg import norm

class _TestCoeffTrans(unittest.TestCase):

    def test_coeff_reduced2raw(self):
        rcut = 9.0
        nq = 10

        nzeta = [1, 2, 3, 4]
        lmax = len(nzeta) - 1
        coeff_rdc = [np.random.randn(nzeta[l], nq-1).tolist()
                     for l in range(lmax+1)]
        coeff_raw = coeff_reduced2raw(coeff_rdc, rcut) 

        for l in range(lmax+1):
            raw = np.array(coeff_raw[l]).T
            rdc = np.array(coeff_rdc[l]).T
            M = jl_reduce(l, nq, rcut)
            self.assertLess(norm(M @ rdc - raw, np.inf), 1e-12)

        nzeta = [0, 1, 0, 1]
        lmax = len(nzeta) - 1
        coeff_rdc = [np.random.randn(nzeta[l], nq-1).tolist()
                     for l in range(lmax+1)]
        coeff_raw = coeff_reduced2raw(coeff_rdc, rcut)

        self.assertTrue(coeff_raw[0] == [] and coeff_raw[2] == [])


    def test_coeff_raw_normalize(self):
        rcut = 9.0
        nq = 10

        nzeta = [1, 2, 3, 4]
        lmax = len(nzeta) - 1
        coeff_raw = [np.random.randn(nzeta[l], nq).tolist()
                     for l in range(lmax+1)]

        coeff_norm = coeff_raw2normalized(coeff_raw, rcut)
        coeff_raw2 = coeff_normalized2raw(coeff_norm, rcut)
        for l in range(lmax+1):
            self.assertTrue(np.allclose(coeff_raw[l], coeff_raw2[l]))
            for zeta in range(nzeta[l]):
                for q in range(nq):
                    self.assertAlmostEqual(coeff_norm[l][zeta][q],
                                           coeff_raw[l][zeta][q]
                                           * jl_raw_norm(l, q, rcut))



if __name__ == '__main__':
    unittest.main()



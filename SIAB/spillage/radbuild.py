from SIAB.spillage.jlzeros import JLZEROS
from SIAB.spillage.radial import _smooth, rad_norm, inner_prod
from SIAB.spillage.coeff_trans import coeff_reduced2raw

import numpy as np
from scipy.special import spherical_jn


def build_raw(coeff, rcut, r, sigma=0.0, orthonormal=False):
    '''
    Builds a set of numerical radial functions by linear combinations of
    truncated spherical Bessel functions.

    Parameters
    ----------
        coeff : list of list of list of float
            A nested list of spherical Bessel coefficients,
            coeff[l][zeta][q] -> float
        rcut : int or float
            Cutoff radius.
        r : array of float
            Grid where the radial functions are evaluated.
        sigma : float
            Smoothing parameter.
        orthonormal : bool
            Whether to orthonormalize the radial functions.
            If True, the resulting radial functions may not be consistent
            with the given coeff.

    Returns
    -------
        chi : list of list of array of float
            A nested list of numerical radial functions,
            chi[l][zeta][ir] -> float.

    Note
    ----
    rcut does not have to be the same as r[-1]; r[-1] can be either larger
    or smaller than rcut.

    '''

    g = _smooth(r, rcut, sigma)
    chi = [[None for _ in coeff_l] for coeff_l in coeff]

    for l, coeff_l in enumerate(coeff):
        for zeta, coeff_lz in enumerate(coeff_l):
            chi[l][zeta] = sum(clzq * spherical_jn(l, JLZEROS[l][q]*r/rcut)
                               for q, clzq in enumerate(coeff_lz))

            chi[l][zeta] *= g # smooth & truncate

            if orthonormal:
                chi[l][zeta] -= sum(inner_prod(chi[l][y], chi[l][zeta], r)
                                    * chi[l][y] for y in range(zeta))
                chi[l][zeta] /= rad_norm(chi[l][zeta], r)

    return chi


def build_reduced(coeff, rcut, r, orthonormal=False):
    '''
    Builds a set of numerical radial functions by linear combinations of
    orthonormal end-smoothed mixed spherical Bessel basis.

    Parameters
    ----------
        coeff : list of list of list of float
            A nested list of spherical Bessel coefficients,
            coeff[l][zeta][q] -> float
        rcut : int or float
            Cutoff radius.
        r : array of float
            Grid where the radial functions are evaluated.
        orthonormal : bool
            Whether to orthonormalize the radial functions.
            If True, the resulting radial functions may not be consistent
            with the given coeff.
    
    Returns
    -------
        chi : list of list of array of float
            A nested list of numerical radial functions,
            chi[l][zeta][ir] -> float.

    Notes
    -----
    Items of the same l must consist of the same number of basis functions.

    '''
    if orthonormal:
        coeff = [np.linalg.qr(np.array(coeff_l).T)[0].T.tolist()
                 if coeff_l else [] for coeff_l in coeff]

    return build_raw(coeff_reduced2raw(coeff, rcut), rcut, r, 0.0, False)


############################################################
#                       Test
############################################################
import unittest
import matplotlib.pyplot as plt
from numpy.linalg import norm

class _TestRadBuild(unittest.TestCase):

    def test_build_raw(self):
        from orbio import read_param, read_nao

        param = read_param('./testfiles/ORBITAL_RESULTS.txt')
        nao = read_nao('./testfiles/In_gga_10au_100Ry_3s3p3d2f.orb')

        nr = int(param['rcut']/nao['dr'])+1
        r = np.linspace(0, param['rcut'], nr)
        chi = build_raw(param['coeff'], param['rcut'], r, param['sigma'],
                        True)

        for l in range(len(chi)):
            for zeta in range(len(chi[l])):
                # check normalization
                self.assertAlmostEqual(rad_norm(chi[l][zeta], r),
                                       1.0, places=12)

                # check orthogonality
                for y in range(zeta):
                    self.assertAlmostEqual(inner_prod(chi[l][zeta],
                                                      chi[l][y],
                                                      r),
                                           0, places=12)

                # cross check with NAO file
                self.assertLess(norm(chi[l][zeta] - nao['chi'][l][zeta]),
                                1e-12)


    def test_build_reduced(self):
        rcut = 9.0
        nq = 10

        nzeta = [1, 2, 3, 4]
        lmax = len(nzeta) - 1
        coeff_rdc = [np.random.randn(nzeta[l], nq-1).tolist()
                     for l in range(lmax+1)]
        coeff_raw = coeff_reduced2raw(coeff_rdc, rcut)

        dr = 0.01
        r = np.linspace(0, rcut, int(rcut/dr)+1)

        for orthnorm in [True, False]:
            chi_rdc = build_reduced(coeff_rdc, rcut, r, orthnorm)
            chi_raw = build_raw(coeff_raw, rcut, r, 0.0, orthnorm)
            for l in range(lmax+1):
                for zeta in range(nzeta[l]):
                    # adjust the sign of the reduced chi to match the raw chi
                    idx = np.argmax(np.abs(chi_raw[l][zeta]))
                    if chi_raw[l][zeta][idx] * chi_rdc[l][zeta][idx] < 0:
                        chi_rdc[l][zeta] *= -1
                    self.assertLess(norm(chi_raw[l][zeta] - chi_rdc[l][zeta]),
                                    1e-12)


    def est_plot_reduced(self):
        lmax = 10
        nq = 7

        rcut = 7.0
        dr = 0.01
        nr = int(rcut/dr) + 1
        r = np.linspace(0, rcut, nr)

        coeff_reduced = [np.eye(nq).tolist()] * (lmax+1)
        chi_reduced = build_reduced(coeff_reduced, rcut, r, False)

        l = 3
        for chi in chi_reduced[l]:
            plt.plot(r, chi)

        plt.xlim([0, rcut])
        plt.show()


if __name__ == '__main__':
    unittest.main()



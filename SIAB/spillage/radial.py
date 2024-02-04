import numpy as np
from scipy.integrate import simpson
from scipy.special import spherical_jn
from jlzeros import JLZEROS

def _smooth(r, rcut, sigma):
    '''
    Smoothing function used in the generation of numerical radial functions.

    Parameters
    ----------
        r : array of float
            Radial grid.
        rcut : int or float
            Cutoff radius.
        sigma : float
            Smoothing parameter.

    Returns
    -------
        g : array of float
            Smoothing function on the radial grid.
    
    References
    ----------
        Chen, M., Guo, G. C., & He, L. (2010).
        Systematically improvable optimized atomic basis sets for ab initio calculations.
        Journal of Physics: Condensed Matter, 22(44), 445501.
    
    '''
    if abs(sigma) < 1e-15:
        g = np.ones_like(r)
    else:
        g = 1. - np.exp(-0.5*((r-rcut)/sigma)**2)

    g[r >= rcut] = 0.0
    return g


def build(coeff, rcut, dr, sigma, orth=False):
    '''
    Builds a set of numerical radial functions by linear combinations of spherical Bessel functions.
    
    Parameters
    ----------
        coeff : list of list of list of float
            A nested list of spherical Bessel coefficients organized as coeff[l][zeta][q]
            where l, zeta and q label the angular momentum, zeta number and wave number respectively.
        rcut : int or float
            Cutoff radius.
        dr : float
            Grid spacing.
        sigma : float
            Smoothing parameter.
        orth : bool
            Whether to orthogonalize the radial functions. If True, radial functions
            will NOT be consistent with the given spherical Bessel coefficients.
    
    Returns
    -------
        chi : list of list of array of float
            A nested list of numerical radial functions organized as chi[l][zeta][ir].

        r : array of float
            Radial grid.
    
    '''

    r = dr * np.arange(int(rcut/dr) + 1)
    g = _smooth(r, rcut, sigma)
    k = [ [ JLZEROS[l][i] / rcut for i in range(max(len(clz) for clz in cl))] for l, cl in enumerate(coeff) ]

    chi = [[None for _ in coeff_l] for coeff_l in coeff]
    for l, coeff_l in enumerate(coeff):
        for zeta, coeff_lz in enumerate(coeff_l):

            chi[l][zeta] = sum(coeff_lzq * spherical_jn(l, k[l][q]*r) for q, coeff_lzq in enumerate(coeff_lz))
            chi[l][zeta] *= g

            if orth: # Gram-Schmidt
                chi[l][zeta] -= sum(simpson(r**2 * chi[l][zeta] * chi[l][y], r) * chi[l][y] for y in range(zeta))

            chi[l][zeta] *= 1. / np.sqrt(simpson((r*chi[l][zeta])**2, r)) # normalize

    return chi, r


def norm_fac(l, q, rcut):
    '''
    Normalization factor for the truncated spherical Bessel function.

    This function returns a normalization factor N such that

    \int_0^{rcut} [ r * N * spherical_jn(l, JLZEROS[l][q] * r / rcut) ]**2 dr = 1.

    '''
    return np.sqrt(2) / (rcut**1.5 * np.abs(spherical_jn(l+1, JLZEROS[l][q])))


def jlbar(l, q, r, m, rcut=None):
    '''
    Normalized truncated spherical Bessel function and its derivatives.

    Given a cutoff radius "rcut", the q-th normalized truncated l-th order
    spherical Bessel function is defined as

            N * spherical_jn(l, JLZEROS[l][q] * r / rcut)

    where JLZEROS[l][q] is the q-th positive zero of the l-th order spherical
    Besesl function and N is a normalization factor.

    Parameters
    ----------
        l : int
            Order of the spherical Bessel function.
        q : int
            Number of nodes.
        r : array of float
            Grid where the function is evaluated.
        m : int
            Order of the derivative. 0 for the function itself.
        rcut : int or float, optional
            Cutoff radius. If not given, the last element of "r" is used.

    '''
    if rcut is None:
        rcut = r[-1] if hasattr(r, '__len__') else r

    k = JLZEROS[l][q] / rcut

    def _jlbar_recur(l, m):
        if m == 0:
            return norm_fac(l, q, rcut) * spherical_jn(l, k*r)
        else:
            if l == 0:
                return -norm_fac(0, q, rcut) * k * _jlbar_recur(1, m-1) / norm_fac(1, q, rcut)
            else:
                return  ( l * _jlbar_recur(l-1, m-1) / norm_fac(l-1, q, rcut) \
                        - (l+1) * _jlbar_recur(l+1, m-1) / norm_fac(l+1, q, rcut) ) \
                        * norm_fac(l, q, rcut) * k / (2 * l + 1)

    return _jlbar_recur(l, m)


############################################################
#                       Test
############################################################
import unittest

class _TestRadial(unittest.TestCase):

    def test_smooth(self):
        r = np.linspace(0, 10, 100)
        rcut = 5.0

        sigma = 0.0
        g = _smooth(r, rcut, sigma)
        self.assertTrue(np.all(g[r < rcut] == 1.0) and np.all(g[r >= rcut] == 0.0))
    
        sigma = 0.5
        g = _smooth(r, rcut, sigma)
        self.assertTrue(np.all(g[r < rcut] == 1.0 - np.exp(-0.5*((r[r < rcut]-rcut)/sigma)**2)))
        self.assertTrue(np.all(g[r >= rcut] == 0.0))
    
    
    def test_build(self):
        from scipy.integrate import simpson
        from orbio import read_param, read_nao

        param = read_param('./testfiles/ORBITAL_RESULTS.txt')
        nao = read_nao('./testfiles/In_gga_10au_100Ry_3s3p3d2f.orb')
        chi, r = build(param['coeff'], param['rcut'], nao['dr'], param['sigma'], orth=True)

        # check normalization
        for l in range(len(chi)):
            for zeta in range(len(chi[l])):
                self.assertAlmostEqual(simpson((r*chi[l][zeta])**2, dx=nao['dr']), 1.0, places=12)

        # check orthogonality
        for l in range(len(chi)):
            for zeta in range(1, len(chi[l])):
                for y in range(zeta):
                    self.assertAlmostEqual(simpson(r**2 * chi[l][zeta] * chi[l][y], dx=nao['dr']), 0.0, places=12)

        # cross check with NAO file
        for l in range(len(chi)):
            for zeta in range(len(chi[l])):
                self.assertTrue(np.all(np.abs(chi[l][zeta] - np.array(nao['chi'][l][zeta])) < 1e-12))


    def test_norm_fac(self):
        rcut = 7.0
        dr = 0.01
        r = np.linspace(0, rcut, int(rcut/dr) + 1)
        nzeros = 10
        for l in range(5):
            k = [ JLZEROS[l][i] / rcut for i in range(nzeros) ]
            for q in range(nzeros):
                f = norm_fac(l, q, rcut) * spherical_jn(l, k[q]*r)
                self.assertAlmostEqual(simpson((r * f)**2, r), 1.0, places=12)

    
    def test_jlbar(self):
        rcut = 7.0
        dr = 0.01
        r = np.linspace(0, rcut, int(rcut/dr) + 1)
        nzeros = 10
        for l in range(5):
            for q in range(nzeros):
                f = jlbar(l, q, r, 0, rcut)
                self.assertAlmostEqual(simpson((r * f)**2, r), 1.0, places=12)

        # finite difference check
        l = 0
        q = 2
        f = jlbar(l, q, r, 0, rcut)

        import matplotlib.pyplot as plt

        df = jlbar(l, q, r, 1, rcut)
        df_fd = np.gradient(f, r)

        self.assertTrue( simpson((r * (df-df_fd))**2, r) < 1e-6 )

        d2f = jlbar(l, q, r, 2, rcut)
        d2f_fd = np.gradient(df, r)
        self.assertTrue( simpson((r * (d2f-d2f_fd))**2, r) < 1e-6 )

        d3f = jlbar(l, q, r, 3, rcut)
        d4f = jlbar(l, q, r, 4, rcut)
        d5f = jlbar(l, q, r, 5, rcut)

        d3f_fd = np.gradient(d2f, r)
        d4f_fd = np.gradient(d3f, r)
        d5f_fd = np.gradient(d4f, r)



if __name__ == '__main__':
    unittest.main()



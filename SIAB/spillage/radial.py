import numpy as np
from scipy.integrate import simpson
from scipy.special import spherical_jn
from jlzeros import JLZEROS

from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

def _inner_prod(chi1, chi2, r):
    '''
    Inner product of two radial functions on a grid.

    Parameters
    ----------
        chi1 : array of float
            First radial function.
        chi2 : array of float
            Second radial function.
        r : array of float
            Radial grid.

    Returns
    -------
        float
            Inner product of chi1 and chi2.

    '''
    return simpson(r**2 * chi1 * chi2, r)


def _rad_norm(chi, r):
    '''
    Norm of a radial function.

    Parameters
    ----------
        chi : array of float
            Radial function.
        r : array of float
            Radial grid.

    Returns
    -------
        float
            Norm of chi.

    '''
    return np.sqrt(_inner_prod(chi, chi, r))


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
    Builds a set of numerical radial functions by linear combinations
    of spherical Bessel functions.
    
    Parameters
    ----------
        coeff : list of list of list of float
            A nested list of spherical Bessel coefficients organized as coeff[l][zeta][q]
            where l, zeta and q label the angular momentum, zeta number and wave number
            respectively.
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
    
    Notes
    -----
    In this function, the spherical Bessel basis where "coeff" applies are not normalized.
    Normalization is performed in the returned radial functions.

    When "orth" is set to True, the radial functions are Gram-Schmidt orthogonalized before
    normalization. In this case, the radial functions are not consistent with the given
    spherical Bessel coefficients. The user should be aware of this when using the returned
    radial functions.

    '''

    r = dr * np.arange(int(rcut/dr) + 1)
    g = _smooth(r, rcut, sigma)
    k = [ JLZEROS[l][:max(len(clz) for clz in cl)] / rcut for l, cl in enumerate(coeff) ]

    chi = [[None for _ in coeff_l] for coeff_l in coeff]
    for l, coeff_l in enumerate(coeff):
        for zeta, coeff_lz in enumerate(coeff_l):
            chi[l][zeta] = sum(coeff_lzq * spherical_jn(l, k[l][q]*r) \
                    for q, coeff_lzq in enumerate(coeff_lz))

            chi[l][zeta] *= g # smooth

            if orth: # Gram-Schmidt
                chi[l][zeta] -= sum(chi[l][y] * _inner_prod(chi[l][y], chi[l][zeta], r) \
                        for y in range(zeta))

            chi[l][zeta] /= _rad_norm(chi[l][zeta], r) # normalize

    return chi, r


def _norm_fac(l, q, rcut):
    '''
    Normalization factor for the node-truncated spherical Bessel function.

    This function returns a normalization factor N such that

    \int_0^{rcut} [ r * N * spherical_jn(l, JLZEROS[l][q] * r / rcut) ]**2 dr = 1.

    See also jlbar.

    '''
    return np.sqrt(2) / (rcut**1.5 * np.abs(spherical_jn(l+1, JLZEROS[l][q])))


def jlbar(l, q, r, m, rcut=None):
    '''
    Normalized node-truncated spherical Bessel function and its derivatives.

    Given a cutoff radius "rcut", the q-th normalized node-truncated l-th order
    spherical Bessel function is defined as

            N * spherical_jn(l, JLZEROS[l][q] * r / rcut)

    where JLZEROS[l][q] is the q-th positive zero of the l-th order spherical
    Besesl function and N is a normalization factor returned by _norm_fac.

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

    Returns
    -------
        array of float

    '''
    if rcut is None:
        rcut = r[-1] if hasattr(r, '__len__') else r

    k = JLZEROS[l][q] / rcut

    def _jl_recur(l, m):
        if m == 0:
            return spherical_jn(l, k*r)
        else:
            if l == 0:
                return - k * _jl_recur(1, m-1)
            else:
                return ( l * _jl_recur(l-1, m-1) - (l+1) * _jl_recur(l+1, m-1) ) \
                        * k / (2*l+1)

    return _norm_fac(l, q, rcut) * _jl_recur(l, m)


def jlbar_build(l, ecut, rcut, r):
    '''
    Builds a list of normalized node-truncated spherical Bessel functions.

    Parameters
    ----------
        l : int
            Order of the spherical Bessel function.
        ecut : int or float
            Energy cutoff. The kinetic energy of a normalized spherical Bessel
            function N*jl(k*r) is k**2. This parameter determines the number of
            items in the returned list.
        rcut : int or float
            Cutoff radius.
        r : array of float
            Grid where the function is evaluated.

    Returns
    -------
        list of array of float

    '''
    nq = np.sum( (JLZEROS[l] / rcut)**2 <= ecut )
    return [ jlbar(l, q, r, 0, rcut) for q in range(nq) ]


def jlbar_to_basis(l, n, rcut):
    '''
    Transformation matrix from jlbar to a tail-smoothed basis.

    This function considers a set of normalized node-truncated spherical Bessel
    functions (jlbar) below some energy cutoff "ecut"

                            [f1, f2, ..., fn]

    and returns a n-by-(n-1) matrix T such that the transformed basis

                [e1, e2, ..., e_{n-1}] = [f1, f2, ..., fn] @ T

    have zero first and second derivatives at the cutoff radius "rcut".

    See also jlbar.

    Parameters
    ----------
        l : int
            Order of the spherical Bessel function.
        n : int
            Number of initial spherical Bessel functions.
        rcut : int or float
            Cutoff radius.

    Notes
    -----
    The transformation matrix to have a basis of vanishing first M derivatives
    is simply the null space of

                D_{mq} = jlbar(l, q, r, m)      m = 1, ..., M       q = 0, ..., n-1

    Given that the ratio between the first and second derivatives of jlbar at rcut
    is a constant (-rcut/2), in order to have vanishing first and second derivatives,
    it is sufficient to consider D with M = 1.

    '''
    k = JLZEROS[l][:n] / rcut

    # first derivative of the normalized node-truncated spherical Bessel function at rcut
    D = np.array([[jlbar(l, q, rcut, 1) for q in range(n)]])
    _, _, vh = np.linalg.svd(D, full_matrices=True)
    C = vh.T[:,1:] # null space of D

    eigval, eigvec = np.linalg.eigh(C.T @ np.diag(k[:n]**2) @ C)
    return C @ eigvec


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
                self.assertAlmostEqual(_rad_norm(chi[l][zeta], r), 1.0, places=12)

        # check orthogonality
        for l in range(len(chi)):
            for zeta in range(1, len(chi[l])):
                for y in range(zeta):
                    self.assertAlmostEqual(_inner_prod(chi[l][zeta], chi[l][y], r), 0.0, places=12)

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
            k = JLZEROS[l] / rcut
            for q in range(nzeros):
                f = _norm_fac(l, q, rcut) * spherical_jn(l, k[q]*r)
                self.assertAlmostEqual(_rad_norm(f, r), 1.0, places=12)

    
    def test_jlbar(self):
        rcut = 5.0
        dr = 0.001 # we need a find grid to check derivatives
        r = np.linspace(0, rcut, int(rcut/dr) + 1)
        for l in range(5):
            for q in range(20):
                f = jlbar(l, q, r, 0, rcut)
                self.assertAlmostEqual(_rad_norm(f, r), 1.0, places=12)

        # check derivatives via spline interpolation
        for l in range(5):
            for q in range(5):
                f = jlbar(l, q, r, 0, rcut)
                df = jlbar(l, q, r, 1, rcut)
                d2f = jlbar(l, q, r, 2, rcut)

                f_spline = CubicSpline(r, f)
                df_spline = f_spline(r, 1)
                d2f_spline = f_spline(r, 2)

                self.assertTrue( simpson((r * (df-df_spline))**2, r) < 1e-12 )
                self.assertTrue( simpson((r * (d2f-d2f_spline))**2, r) < 1e-9 )

    
    def test_jlbar_build(self):
        l = 2
        rcut = 7
        dr = 0.001
        ecut = 60
        r = np.linspace(0, rcut, int(rcut/dr)+1)
        jlbar_all = jlbar_build(l, ecut, rcut, r)

        # check the kinetic energy
        T_ref = (JLZEROS[l] / rcut)**2
        for q, f in enumerate(jlbar_all):
            spl = CubicSpline(r, f)
            df = spl(r, 1)
            d2f = spl(r, 2)

            T = simpson(-2*r*f*df, r) - simpson(r**2*f*d2f, r) + l*(l+1)*simpson(f**2, r)
            self.assertAlmostEqual(T, T_ref[q], places=3)


    def test_jlbar_to_basis(self):
        l = 2
        rcut = 7
        nq = 10
        D = np.zeros((2, nq))
        for q in range(nq):
            D[0, q] = jlbar(l, q, rcut, 1)
            D[1, q] = jlbar(l, q, rcut, 2)

        C = jlbar_to_basis(l, nq, rcut)
        self.assertTrue(np.linalg.norm(D @ C, np.inf) < 1e-12)


if __name__ == '__main__':
    unittest.main()



from SIAB.spillage.jlzeros import JLZEROS

import numpy as np
from scipy.integrate import simpson
from scipy.special import spherical_jn
from scipy.interpolate import CubicSpline
from scipy.linalg import rq

def inner_prod(chi1, chi2, r):
    '''
    Inner product of two numerical radial functions.

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
    return simpson(r**2 * chi1 * chi2, x=r)


def rad_norm(chi, r):
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
    return np.sqrt(inner_prod(chi, chi, r))


def kinetic(r, l, chi):
    '''
    Kinetic energy of pseudo-atomic orbitals.

    The kinetic energy of a pseudo-atomic orbital

                phi(vec{r}) = chi(r) * Y_{lm}(hat{r})

    merely depends on the radial part chi(r) and l. Given a radial function
    chi(r) evaluated on grid r, this function evaluates the integral

        / rcut
        |      dr chi(r)*[-(d/dr)(r^2*(d/dr)chi) + l*(l+1)*chi(r)]
        / 0

    by Simpson's rule.

    Parameters
    ----------
        r : np.ndarray
            Radial grid.
        l : int
            Angular momentum quantum number.
        chi : np.ndarray 
            Radial part of the pseudo-atomic orbital evaluated on the
            radial grid r.

    Note
    ----
    This function does not check whether the input chi is normalized or not;
    it merely evaluates the integral.

    '''
    f = CubicSpline(r, chi)
    dchi = f(r, 1)
    d2chi = f(r, 2)
    return simpson((-2 * r * dchi - r**2 * d2chi + l*(l+1) * chi) * chi, x=r)


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
        Systematically improvable optimized atomic basis sets for ab initio
        calculations. Journal of Physics: Condensed Matter, 22(44), 445501.
    
    '''
    g = 1. - np.exp(-0.5*((r-rcut)/sigma)**2) if sigma != 0 \
            else np.ones_like(r)
    g[r >= rcut] = 0.0
    return g


def jl_raw(l, q, r, rcut=None, deriv=0):
    '''
    Truncated spherical Bessel functions and derivatives.

    The q-th rcut-truncated l-th order spherical Bessel function is defined as

                -
                |   spherical_jn(l, JLZEROS[l][q] * r / rcut)   r <= rcut
        f(r) =  |
                |   0                                           r > rcut
                -

    where JLZEROS[l][q] is the q-th positive zero of the l-th order spherical
    Besesl function.

    Parameters
    ----------
        l : int
            Order of the spherical Bessel function.
        q : int
            Wavenumber index. Also the number of nodes.
        r : array of float
            Grid where the function is evaluated.
        rcut : int or float, optional
            Cutoff radius. If not given, the last element of "r" is used.
        deriv : int
            Order of the derivative. 0 for the function itself.

    Returns
    -------
        array of float

    Notes
    -----
    Functions of the same l & rcut but different q are orthogonal in terms of
    inner_prod, but they are not normalized.

    '''
    if rcut is None:
        rcut = r[-1] if hasattr(r, '__len__') else r

    k = JLZEROS[l][q] / rcut

    def _recur(l, m):
        if m == 0:
            if hasattr(r, '__len__'):
                tmp = spherical_jn(l, k * r)
                tmp[r > rcut] = 0.0
                return tmp
            else:
                return 0.0 if r > rcut else spherical_jn(l, k * r)
        else:
            if l == 0:
                return - k * _recur(1, m-1)
            else:
                return ( l * _recur(l-1, m-1) - (l+1) * _recur(l+1, m-1) ) \
                        * k / (2*l+1)

    return _recur(l, deriv)


def jl_raw_norm(l, q, rcut):
    '''
    Norm of a truncated spherical Bessel function.

    Note
    ----
    The integral

        / rcut 
        |      [ r * spherical_jn(l, JLZEROS[l][q] * r / rcut) ]**2 dr
        /  0

    has an analytical expression, see, e.g., Arfken, Weber and Harris,
    Mathematical Methods for Physicists, 7th ed., p. 704.

    '''
    return (rcut**1.5 * np.abs(spherical_jn(l+1, JLZEROS[l][q]))) / np.sqrt(2)


def jl_reduce(l, n, rcut, from_raw=True):
    '''
    Transformation matrix from truncated spherical Bessel functions to
    "reduced" spherical Bessel functions.

    Consider a set of n truncated spherical Bessel functions {f} with cutoff
    radius rcut, this function returns an n-by-(n-1) matrix T such that the
    transformed basis

                [e1, e2, ..., e_{n-1}] = [f1, f2, ..., fn] @ T

    are orthonormal and have vanishing first and second derivatives at rcut.

    Parameters
    ----------
        l : int
            Order of the spherical Bessel function.
        n : int
            Number of raw/normalized spherical Bessel functions. Note that
            the size of the transformed basis is n-1.
        rcut : int or float
            Cutoff radius.
        from_raw : bool
            If False, the truncated functions are assumed to be normalized.

    Notes
    -----
    For a normalized truncated spherical Bessel function, the ratio between
    its first and second derivative at rcut is always a constant (-rcut/2).
    Therefore, in order to have vanishing first and second derivatives, it
    is sufficient to work on the first derivative only.

    '''
    if n == 1: # edge case
        return np.zeros((1,0))

    inv_raw_norm = np.array([1.0 / jl_raw_norm(l, q, rcut)
                             for q in range(n)])

    # first derivatives of the normalized truncated spherical Bessel
    # function at rcut
    D = np.array([[jl_raw(l, q, rcut, deriv=1) * inv_raw_norm[q]
                   for q in range(n)]])

    # null space of D
    C = np.linalg.svd(D, full_matrices=True)[2].T[:,1:]

    # Instead of a "canonicalization" in terms of the kinetic energy,
    # we choose to maintain the consistency of results w.r.t. different
    # numbers of spherical Bessel functions, i.e., the result from N
    # spherical Bessel functions (which is an N-by-(N-1) matrix) should
    # be identical to the upper-left N-by-(N-1) block of the result from
    # any M (M>N) spherical Bessel functions.
    T = inv_raw_norm.reshape(-1,1) * rq(C)[0] if from_raw else rq(C)[0]

    # make sure the largest-magnitude element in each column is positive
    idx = np.argmax(np.abs(T), axis=0)
    return T * np.sign(T[idx, range(n-1)])


def _nbes(l, rcut, ecut):
    '''
    Calculates the number of normalized truncated spherical Bessel functions
    whose kinetic energy is below the energy cutoff.

    Note
    ----
    1. The kinetic energy of a normalized truncated spherical Bessel basis
       j_l(k*r) * Y_{lm}(r) is k^2

    2. The wavenumbers of truncated spherical Bessel functions are chosen such
       that the function is zero at rcut, i.e., JLZEROS/rcut

    '''
    # make sure the tabulated zeros are sufficient
    assert (JLZEROS[l][-1]/rcut)**2 > ecut
    return sum((JLZEROS[l]/rcut)**2 < ecut)


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

from numpy.linalg import norm
import matplotlib.pyplot as plt

class _TestRadial(unittest.TestCase):

    def test_nbes(self):
        ecut = 100.0
        rcut = 6.0
        dr = 0.001
        r = np.linspace(0, rcut, int(rcut/dr) + 1)

        for l in range(10):
            q = _nbes(l, rcut, ecut)

            # verify that kinetic energy of the q-th normalized spherical
            # Bessel function is above the energy cutoff while that of the
            # (q-1)-th is below the cutoff.
            inv_raw_norm = 1.0 / jl_raw_norm(l, q, rcut)
            f = jl_raw(l, q, r, rcut, 0) * inv_raw_norm
            df = jl_raw(l, q, r, rcut, 1) * inv_raw_norm
            d2f = jl_raw(l, q, r, rcut, 2) * inv_raw_norm

            kin = simpson((-2 * r * df - r**2 * d2f + l*(l+1) * f) * f, x=r)
            self.assertGreater(kin, ecut)

            inv_raw_norm = 1.0 / jl_raw_norm(l, q-1, rcut)
            f = jl_raw(l, q-1, r, rcut, 0) * inv_raw_norm
            df = jl_raw(l, q-1, r, rcut, 1) * inv_raw_norm
            d2f = jl_raw(l, q-1, r, rcut, 2) * inv_raw_norm

            kin = simpson((-2 * r * df - r**2 * d2f + l*(l+1) * f) * f, x=r)
            self.assertLess(kin, ecut)


    def test_inner_prod(self):
        # checks inner_prod by verifying the orthogonality
        # of truncated spherical Bessel functions.
        rcut = 3.0
        dr = 0.001
        r = np.linspace(0, rcut, int(rcut/dr)+1)
        lmax = 5
        nq = 10
        for l in range(lmax+1):
            k = JLZEROS[l] / rcut
            for q in range(1, nq):
                for p in range(q):
                    self.assertAlmostEqual(inner_prod(spherical_jn(l, k[q]*r),
                                                      spherical_jn(l, k[p]*r),
                                                      r),
                                           0, places=12)


    def test_rad_norm(self):
        # checks rad_norm by some simple analytical cases.
        a = 7.0
        dr = 0.001
        r = np.linspace(0, a, int(a/dr)+1)

        self.assertAlmostEqual(rad_norm(np.cos(np.pi * r / a), r),
                               np.sqrt((2.0 + 3.0/np.pi**2) * a**3 / 12.0),
                               places=12)

        self.assertAlmostEqual(rad_norm(np.sin(np.pi * r / a), r),
                               np.sqrt((2.0 - 3.0/np.pi**2) * a**3 / 12.0),
                               places=12)

        self.assertAlmostEqual(rad_norm(np.sin(2.0 * np.pi * r / a), r),
                               np.sqrt((8.0 * np.pi**2 - 3) * a**3 / 48.0)
                               / np.pi,
                               places=12)


    def test_kinetic(self):
        rcut = 7.0
        dr = 0.001
        nr = int(rcut/dr) + 1
        r = dr * np.arange(nr)

        # check the numerical kinetic energies with analytical expressions
        nq = 5
        lmax = 4
        for l in range(lmax+1):
            for q in range(nq):
                chi = jl_raw(l, q, r, rcut) / jl_raw_norm(l, q, rcut)
                self.assertAlmostEqual(kinetic(r, l, chi),
                                       (JLZEROS[l][q] / rcut)**2,
                                       places=5)


    def test_smooth(self):
        r = np.linspace(0, 10, 100)
        rcut = 5.0
        rm = r < rcut
        rp = r >= rcut

        sigma = 0.0
        g = _smooth(r, rcut, sigma)
        self.assertTrue(np.all(g[rm] == 1.0) and np.all(g[rp] == 0.0))
    
        sigma = 0.5
        g = _smooth(r, rcut, sigma)
        self.assertTrue(np.all(g[rm] == 
                               1.0 - np.exp(-0.5*((r[rm]-rcut)/sigma)**2)))
        self.assertTrue(np.all(g[rp] == 0.0))


    def test_jl_raw_norm(self):
        # cross-checks jl_raw_norm with rad_norm
        rcut = 7.0
        dr = 0.001
        r = np.linspace(0, rcut, int(rcut/dr) + 1)
        nzeros = 30
        for l in range(5):
            k = JLZEROS[l] / rcut
            for q in range(nzeros):
                f = spherical_jn(l, k[q]*r) / jl_raw_norm(l, q, rcut)
                self.assertAlmostEqual(rad_norm(f, r), 1.0, places=12)


    def test_jl_raw(self):
        rcut = 5.0
        dr = 0.001
        r = np.linspace(0, rcut, int(rcut/dr) + 1)

        # checks the first and second derivatives via spline interpolation
        # and kinetic energies
        for l in range(5):
            kin_ref = (JLZEROS[l] / rcut)**2
            for q in range(5):
                inv_raw_norm = 1.0 / jl_raw_norm(l, q, rcut)
                f = jl_raw(l, q, r, rcut, 0) * inv_raw_norm
                df = jl_raw(l, q, r, rcut, 1) * inv_raw_norm
                d2f = jl_raw(l, q, r, rcut, 2) * inv_raw_norm

                spline = CubicSpline(r, f)
                df_spline = spline(r, 1)
                d2f_spline = spline(r, 2)

                self.assertLess(simpson((r * (df-df_spline))**2, x=r), 1e-12)
                self.assertLess(simpson((r * (d2f-d2f_spline))**2, x=r), 1e-9)

                kin = simpson((-2 * r * df - r**2 * d2f + l*(l+1) * f) * f,
                              x=r)
                self.assertLess(abs(kin - kin_ref[q]), 1e-8)


    def test_jl_reduce(self):
        # checks if transformations by jl_reduce indeed zero-out
        # the first and second derivatives.
        lmax = 5
        rcut = 9.0
        nq = 10
        raw = np.zeros((2, nq))
        nrm = np.zeros((2, nq))
        for l in range(lmax+1):
            for q in range(nq):
                raw[0, q] = jl_raw(l, q, rcut, deriv=1)
                raw[1, q] = jl_raw(l, q, rcut, deriv=2)

                fac = 1.0 / jl_raw_norm(l, q, rcut)
                nrm[0, q] = jl_raw(l, q, rcut, deriv=1) * fac
                nrm[1, q] = jl_raw(l, q, rcut, deriv=2) * fac

            self.assertLess(norm(raw @ jl_reduce(l, nq, rcut, True)),
                            1e-12)
            self.assertLess(norm(nrm @ jl_reduce(l, nq, rcut, False)),
                            1e-12)

        # checks the consistency of jl_reduce w.r.t. different numbers of
        # basis functions
        for l in range(lmax+1):
            T_raw = jl_reduce(l, nq, rcut, True)
            T_nrm = jl_reduce(l, nq, rcut, False)

            for n in range(2, nq):
                T_raw_ = jl_reduce(l, n, rcut, True)
                T_nrm_ = jl_reduce(l, n, rcut, False)

                self.assertLess(norm(T_raw[:n, :n-1] - T_raw_), 1e-12)
                self.assertLess(norm(T_nrm[:n, :n-1] - T_nrm_), 1e-12)


    def test_build_raw(self):
        from orbio import read_param, read_nao
        import os
        here = os.path.dirname(os.path.abspath(__file__))
        jobdir = os.path.join(here, 'testfiles')

        #param = read_param('./testfiles/ORBITAL_RESULTS.txt')
        param = read_param(os.path.join(jobdir, 'ORBITAL_RESULTS.txt'))
        #nao = read_nao('./testfiles/In_gga_10au_100Ry_3s3p3d2f.orb')
        nao = read_nao(os.path.join(jobdir, 'In_gga_10au_100Ry_3s3p3d2f.orb'))

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
                    self.assertTrue(np.allclose(chi_raw[l][zeta],
                                                chi_rdc[l][zeta]))


    def est_plot_reduced(self):
        lmax = 4
        nq = 5

        rcut = 7.0
        dr = 0.01
        nr = int(rcut/dr) + 1
        r = np.linspace(0, rcut, nr)

        coeff_reduced = [np.eye(nq).tolist()] * (lmax+1)
        chi_reduced = build_reduced(coeff_reduced, rcut, r, False)

        fig, ax = plt.subplots(1, 2, figsize=(10,4), layout='tight')

        # same l, different q
        l = 2
        for q, chi in enumerate(chi_reduced[l]):
            ax[0].plot(r, chi, label='q = %d' % q)

        ax[0].set_xlim([0, rcut])
        ax[0].set_title('l = %d' % l)
        ax[0].legend()

        # same q, different l
        q = 0
        for l, chi in enumerate(chi_reduced):
            ax[1].plot(r, chi[q], label='l = %d' % l)
        ax[1].set_xlim([0, rcut])
        ax[1].set_title('q = %d' % q)
        ax[1].legend()


        plt.show()


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



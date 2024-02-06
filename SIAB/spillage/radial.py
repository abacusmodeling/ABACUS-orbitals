import numpy as np
from scipy.integrate import simpson
from scipy.special import spherical_jn
from jlzeros import JLZEROS

from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

def _inner_prod(chi1, chi2, r):
    '''
    Inner product of two radial functions.

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
    g = 1. - np.exp(-0.5*((r-rcut)/sigma)**2) if sigma != 0 else np.ones_like(r)
    g[r >= rcut] = 0.0
    return g


def jl_trunc(l, q, r, rcut=None, deriv=0):
    '''
    Truncated spherical Bessel functions and derivatives.

    Given a cutoff radius "rcut", the q-th truncated l-th order spherical
    Bessel function is defined as

            spherical_jn(l, JLZEROS[l][q] * r / rcut)

    where JLZEROS[l][q] is the q-th positive zero of the l-th order spherical
    Besesl function.

    Parameters
    ----------
        l : int
            Order of the spherical Bessel function.
        q : int
            Number of nodes.
        r : array of float
            Grid where the function is evaluated.
        rcut : int or float, optional
            Cutoff radius. If not given, the last element of "r" is used.
        deriv : int
            Order of the derivative. 0 for the function itself.

    Returns
    -------
        array of float

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


def jl_trunc_norm(l, q, rcut):
    '''
    Norm of a truncated spherical Bessel function.

    This function returns the norm of the q-th rcut-truncated l-th order spherical
    Bessel function:

    N = sqrt( \int_0^{rcut} [ r * spherical_jn(l, JLZEROS[l][q] * r / rcut) ]**2 dr )

    '''
    return ( rcut**1.5 * np.abs(spherical_jn(l+1, JLZEROS[l][q])) ) / np.sqrt(2)


def jl_trunc_reduce(l, n, rcut):
    '''
    This function returns a transformation matrix from truncated spherical Bessel
    functions to orthonormal end-smoothed mixed spherical Bessel basis.

    Consider a set of n truncated spherical Bessel functions {f} with cutoff radius
    rcut, this function returns an n-by-(n-1) matrix T such that the transformed basis

                [e1, e2, ..., e_{n-1}] = [f1, f2, ..., fn] @ T

    are orthonormal and have vanishing first and second derivatives at rcut.

    Parameters
    ----------
        l : int
            Order of the spherical Bessel function.
        n : int
            Number of initial truncated spherical Bessel functions. Note that the
            size of the transformed basis is n-1.
        rcut : int or float
            Cutoff radius.

    Notes
    -----
    For a normalized truncated spherical Bessel function, the ratio between its
    first and second derivative at rcut is always a constant (-rcut/2). Therefore,
    in order to have vanishing first and second derivatives, it is sufficient to
    focus on the first derivative only.

    '''
    norm_fac = np.array([1.0 / jl_trunc_norm(l, q, rcut) for q in range(n)])

    # first derivative of the normalized truncated spherical Bessel function at rcut
    D = np.array([[jl_trunc(l, q, rcut, deriv=1) * norm_fac[q] for q in range(n)]])

    # null space of D
    C = np.linalg.svd(D, full_matrices=True)[2].T[:,1:]

    # diagonalize the kinetic matrix in this subspace ("canonical")
    _, vec = np.linalg.eigh(C.T @ np.diag((JLZEROS[l][:n]/rcut)**2) @ C)

    return np.diag(norm_fac) @ C @ vec


def coeff_recover(coeff, rcut):
    '''
    Converts the coefficients w.r.t the orthonormal end-smoothed mixed spherical
    Bessel basis to those w.r.t the truncated spherical Bessel functions.

    Parameters
    ----------
        coeff: list of list of list of float
            A nested list of basis coefficients organized as coeff[l][zeta][p]
            where l, zeta and p label the angular momentum, zeta number and basis
            index respectively.
        rcut : int or float
            Cutoff radius.

    Returns
    -------
        coeff_raw : list of list of list of float
            A nested list of truncated spherical Bessel coefficients.

    Notes
    -----
    Items of the same l must consist of the same number of basis functions.

    '''
    # temporarily use a column-wise coefficient layout within each l
    coeff_basis = [ np.array(coeff_l).T for coeff_l in coeff ]

    return [ (jl_trunc_reduce(l, coeff_l.shape[0] + 1, rcut) @ coeff_l).T.tolist() \
            for l, coeff_l in enumerate(coeff_basis)]


def _build_raw(coeff, rcut, r, sigma=0.0, orth=False, normalize=False):
    '''
    Builds a set of numerical radial functions by linear combinations of
    truncated spherical Bessel functions.

    See build() for more details.

    '''
    g = _smooth(r, rcut, sigma)
    chi = [[None for _ in coeff_l] for coeff_l in coeff]

    for l, coeff_l in enumerate(coeff):
        for zeta, coeff_lz in enumerate(coeff_l):
            chi[l][zeta] = sum(coeff_lzq * spherical_jn(l, JLZEROS[l][q]*r/rcut) \
                    for q, coeff_lzq in enumerate(coeff_lz))

            chi[l][zeta] *= g # smooth

            if orth: # Gram-Schmidt
                chi[l][zeta] -= sum(chi[l][y] * _inner_prod(chi[l][y], chi[l][zeta], r) \
                        for y in range(zeta))

            if normalize:
                chi[l][zeta] /= _rad_norm(chi[l][zeta], r)

    return chi


def _build_reduced(coeff, rcut, r, orth=False, normalize=False):
    '''
    Builds a set of numerical radial functions by linear combinations of end-smoothed
    mixed spherical Bessel orthonormal basis.

    See build() for more details.

    Notes
    -----
    Items of the same l must consist of the same number of basis functions.

    '''
    # temporarily change to a column-wise coefficient layout within each l
    coeff_reduced = [ np.array(coeff_l).T for coeff_l in coeff ]

    # The end-smoothed mixed spherical Bessel basis is orthonormal.
    # If orthgonalization or normalization is requested, it is easier to do it now than later.
    if orth:
        for l, coeff_l in enumerate(coeff_reduced):
            coeff_reduced[l] = np.linalg.qr(coeff_l, mode='reduced')[0]

    if normalize:
        for l, coeff_l in enumerate(coeff_reduced):
            coeff_reduced[l] /= np.linalg.norm(coeff_l, axis=0)

    coeff_reduced = [ coeff_l.T.tolist() for coeff_l in coeff_reduced ]

    return _build_raw(coeff_recover(coeff_reduced, rcut), rcut, r, 0.0, False, False)


def build(coeff, rcut, r, basis_type, sigma=0.0, orth=False, normalize=False):
    '''
    Builds a set of numerical radial functions by linear combinations of
    spherical Bessel functions.
    
    Parameters
    ----------
        coeff : list of list of list of float
            A nested list of spherical Bessel coefficients organized as coeff[l][zeta][p]
            where l, zeta and p label the angular momentum, zeta number and basis index
            respectively.
        rcut : int or float
            Cutoff radius.
        r : array of float
            Grid where the radial functions are evaluated.
        basis_type : str
            Type of the spherical Bessel basis. Can be 'raw' or 'reduced'.
        sigma : float
            Smoothing parameter. Only used when basis_type is 'raw'.
        orth : bool
            Whether to Gram-Schmidt orthogonalize the radial functions. If True,
            the resulting radial functions may not be consistent with the given
            spherical Bessel coefficients.
        normalize : bool
            Whether to normalize the radial functions. If True, the resulting
            radial functions may not be consistent with the given coefficients.
    
    Returns
    -------
        chi : list of list of array of float
            A nested list of numerical radial functions organized as chi[l][zeta][ir].

    '''
    if basis_type == 'raw':
        return _build_raw(coeff, rcut, r, sigma, orth, normalize)
    elif basis_type == 'reduced':
        return _build_reduced(coeff, rcut, r, orth, normalize)
    else:
        raise ValueError('Unknown basis type: {}'.format(basis_type))


############################################################
#                       Test
############################################################
import unittest

class _TestRadial(unittest.TestCase):

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
        self.assertTrue(np.all(g[rm] == 1.0 - np.exp(-0.5*((r[rm]-rcut)/sigma)**2)))
        self.assertTrue(np.all(g[rp] == 0.0))
    
    
    def test_build(self):
        from orbio import read_param, read_nao

        param = read_param('./testfiles/ORBITAL_RESULTS.txt')
        nao = read_nao('./testfiles/In_gga_10au_100Ry_3s3p3d2f.orb')
        r = np.linspace(0, param['rcut'], int(param['rcut']/nao['dr'])+1)
        chi = build(param['coeff'], param['rcut'], r, 'raw', param['sigma'], True, True)

        # check normalization
        for l in range(len(chi)):
            for zeta in range(len(chi[l])):
                self.assertLess(abs(_rad_norm(chi[l][zeta], r) - 1.0), 1e-12)

        # check orthogonality
        for l in range(len(chi)):
            for zeta in range(1, len(chi[l])):
                for y in range(zeta):
                    self.assertLess(abs(_inner_prod(chi[l][zeta], chi[l][y], r)), 1e-12)

        # cross check with NAO file
        for l in range(len(chi)):
            for zeta in range(len(chi[l])):
                self.assertLess(np.linalg.norm(chi[l][zeta] - nao['chi'][l][zeta]), 1e-12)


    def test_jl_trunc_norm(self):
        rcut = 7.0
        dr = 0.001
        r = np.linspace(0, rcut, int(rcut/dr) + 1)
        nzeros = 30
        for l in range(5):
            k = JLZEROS[l] / rcut
            for q in range(nzeros):
                f = spherical_jn(l, k[q]*r) / jl_trunc_norm(l, q, rcut)
                self.assertLess(abs(_rad_norm(f, r) - 1.0), 1e-12)

    def test_jl_trunc(self):
        rcut = 5.0
        dr = 0.001 # we need a find grid to check derivatives
        r = np.linspace(0, rcut, int(rcut/dr) + 1)
        for l in range(5):
            for q in range(20):
                f = jl_trunc(l, q, r, rcut, 0) / jl_trunc_norm(l, q, rcut)
                self.assertLess(abs(_rad_norm(f, r) - 1.0), 1e-12)

        # check derivatives via spline interpolation and kinetic energies
        for l in range(5):
            kin_ref = (JLZEROS[l] / rcut)**2
            for q in range(5):
                norm_fac = 1.0 / jl_trunc_norm(l, q, rcut)
                f = jl_trunc(l, q, r, rcut, 0) * norm_fac
                df = jl_trunc(l, q, r, rcut, 1) * norm_fac
                d2f = jl_trunc(l, q, r, rcut, 2) * norm_fac

                spline = CubicSpline(r, f)
                df_spline = spline(r, 1)
                d2f_spline = spline(r, 2)

                self.assertLess(simpson((r * (df-df_spline))**2, r), 1e-12)
                self.assertLess(simpson((r * (d2f-d2f_spline))**2, r), 1e-9)

                kin = simpson(-2 * r * f * df, r) - simpson(r**2 * f * d2f, r) \
                        + l*(l+1) * simpson(f**2, r)
                self.assertLess(abs(kin - kin_ref[q]), 1e-8)


    def test_build_from_basis(self):
        pass


    def test_jl_trunc_reduce(self):
        l = 3
        rcut = 10
        dr = 0.01
        r = np.linspace(0, rcut, int(rcut/dr)+1)
        nbes = 20
        raw = np.zeros((len(r), nbes))
        for q in range(nbes):
            raw[:, q] = jl_trunc(l, q, r, rcut, deriv=0)

        new = raw @ jl_trunc_reduce(l, nbes, rcut)

        #for q in range(min(nbes-1, 5)):
        #    plt.plot(r, new[:, q])
        #plt.show()


if __name__ == '__main__':
    unittest.main()



import numpy as np

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
            A nested list of spherical Bessel coefficients organized as coeff[l][zeta][iq]
            where l, zeta and iq label the angular momentum, zeta number and wave number respectively.
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
    from scipy.integrate import simpson
    from scipy.special import spherical_jn
    from jlzeros import ikebe

    r = dr * np.arange(int(rcut/dr) + 1)
    g = _smooth(r, rcut, sigma)
    q = [ikebe(l, max(len(clz) for clz in cl)) / rcut for l, cl in enumerate(coeff)] # wave numbers

    chi = [[None for _ in coeff_l] for coeff_l in coeff]
    for l, coeff_l in enumerate(coeff):
        for zeta, coeff_lz in enumerate(coeff_l):

            chi[l][zeta] = sum(coeff_lzq * spherical_jn(l, q[l][iq]*r) for iq, coeff_lzq in enumerate(coeff_lz))
            chi[l][zeta] *= g

            if orth: # Gram-Schmidt
                chi[l][zeta] -= sum(simpson(r**2 * chi[l][zeta] * chi[l][y], r) * chi[l][y] for y in range(zeta))

            chi[l][zeta] *= 1. / np.sqrt(simpson((r*chi[l][zeta])**2, r)) # normalize

    return chi, r


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


if __name__ == '__main__':
    unittest.main()



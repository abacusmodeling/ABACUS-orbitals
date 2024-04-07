"""for evaluate quality of orbital"""
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.integrate import simps as simpson # for python 3.6, scipy <= 1.10
from SIAB.spillage.orbio import read_nao

def rad_norm(r, chi):
    return np.sqrt( simpson((r*chi)**2, r) )


def kinetic(r, l, chi):
    '''
    Kinetic energy of pseudo-atomic orbitals.

    The kinetic energy of a pseudo-atomic orbital

                phi(vec{r}) = chi(r) * Y_{lm}(hat{r})

    merely depends on the radial part chi(r) and l. Given a radial grid r and
    the radial part chi(r), this function evaluates the following integral:

        \int_0^\infty dr chi(r)*[-(d/dr)(r^2*(d/dr)chi) + l*(l+1)*chi(r)]

    by Simpson's rule.

    Parameters
    ----------
        r : np.ndarray
            Radial grid.
        l : int
            Angular momentum quantum number.
        chi : np.ndarray 
            Radial part of the pseudo-atomic orbital evaluated on the radial grid r.

    Note
    ----
    This function does not check whether the input chi is normalized or not;
    it merely evaluates the integral.

    '''
    f = CubicSpline(r, chi)
    dchi = f(r, 1)
    d2chi = f(r, 2)

    # Simpson's rule

    return simpson(-2 * r * chi * dchi - r**2 * chi * d2chi + l*(l+1) * chi**2, r)


def screener(r, chi, l, item):
    if item == "T":
        return kinetic(r, l, chi)
    else:
        raise ValueError("Unknown item: %s"%item)


def screen(fnao, item="T"):
    nao = read_nao(fnao)
    r = nao['dr'] * np.arange(nao['nr'])
    chi = nao['chi']

    # apply 'screener' to individual numerical radial functions
    return [np.array([screener(r, chi_lz, l, item) for chi_lz in chi_l]) \
            for l, chi_l in enumerate(chi)]


############################################################
#                       Test
############################################################
import unittest

class _TestKinetic(unittest.TestCase):

    def test_kinetic(self):
        from scipy.special import spherical_jn

        rcut = 7.0
        dr = 0.001
        nr = int(rcut/dr) + 1
        r = dr * np.arange(nr)

        # check the kinetic energy of truncated j0(q*r) by the analytical expression
        nq = 5
        for iq in range(nq):
            q = (iq + 1) * np.pi / rcut

            chi = spherical_jn(0, q*r)
            norm_fac = ( rcut**1.5 * np.abs(spherical_jn(1, (iq+1)*np.pi)) ) / np.sqrt(2)
            chi /= norm_fac

            self.assertLess(abs(kinetic(r, 0, chi) - q**2), 1e-5)


    def test_screen(self):
        T_In = screen('./testfiles/In_gga_10au_100Ry_3s3p3d2f.orb', item="T")
        self.assertEqual([len(T_l) for T_l in T_In], [3, 3, 3, 2])


if __name__ == '__main__':
    unittest.main()


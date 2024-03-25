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


def screen_foreach(r, chi, l, term):
    if term == "T":
        return kinetic(r, l, chi)
    else:
        raise ValueError("Unknown term: %s"%term)


def screen(fnao, term="T"):
    # no matter what term-screen, screen zeta-by-zeta.
    nao = read_nao(fnao)
    r = nao['dr'] * np.arange(nao['nr'])
    chi = nao['chi']
    lmax = len(chi)-1
    nzeta = [len(chi[l]) for l in range(lmax+1)]

    vals = [np.zeros(nzeta[l]) for l in range(lmax+1)]
    for l in range(lmax+1):
        for zeta in range(nzeta[l]):
            vals[l][zeta] = screen_foreach(r, chi[l][zeta], l, term)

    return vals
    

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


    #def test_nao_kinetic(self):
    #    from orbio import read_nao
    #    nao = read_nao('./Cr_gga_8au_100Ry_4s2p2d1f.orb')
    #    r = nao['dr'] * np.arange(nao['nr'])

    #    for l, chi_l in enumerate(nao['chi']):
    #        for zeta, chi in enumerate(chi_l):
    #            print('l = %i   zeta = %i    T = %8.3e'%(l, zeta, kinetic(r, l, chi)))


if __name__ == '__main__':
    unittest.main()


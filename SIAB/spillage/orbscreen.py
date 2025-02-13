"""for evaluate quality of orbital"""
import numpy as np
from SIAB.spillage.radial import kinetic 
from SIAB.spillage.orbio import read_nao
from scipy.integrate import simps
from scipy.special import spherical_jn
import os

def _screener(r, chi, l, item):
    if item == "T":
        return kinetic(r, l, chi)
    else:
        raise ValueError("Unknown item: %s"%item)


def screen(fnao, item="T"):
    nao = read_nao(fnao)
    r = nao['dr'] * np.arange(nao['nr'])
    chi = nao['chi']

    # apply '_screener' to individual numerical radial functions
    return [np.array([_screener(r, chi_lz, l, item) for chi_lz in chi_l])
            for l, chi_l in enumerate(chi)]

def _sphbes_transform(r, f, l, q):
    '''perform spherical bessel transform
    
    Parameters
    ----------
    r : np.ndarray
        radial grid
    chi : np.ndarray
        radial function
    l : int
        angular momentum
    q : np.ndarray
        q grid

    Return
    ------
    array
        transform result
    '''
    q = np.array([q]) if np.isscalar(q) else np.array(q)
    return np.array([simps(r**2 * f * spherical_jn(l, q_ * r), r) for q_ in q])


############################################################
#                       Test
############################################################
import unittest
from SIAB.spillage.orbio import read_nao
class TestScreen(unittest.TestCase):

    @unittest.skip('Deprecated')
    def _test_screen(self):
        import os
        here = os.path.dirname(os.path.abspath(__file__))
        jobdir = os.path.join(here, 'testfiles')
        
        T_In = screen(os.path.join(jobdir, 'In_gga_10au_100Ry_3s3p3d2f.orb'), item="T")
        
        self.assertEqual([len(T_l) for T_l in T_In], [3, 3, 3, 2])


    def test_sphbes_transform(self):
        # 
        # sin(qr)/qr is the zero-order spherical bessel function
        # so the spherical bessel transform gets the same result
        r = np.linspace(0, 50, 1000)
        y0 = spherical_jn(0, r)
        y0[0] = 1
        q = np.linspace(0, 10, 1000)
        T = _sphbes_transform(r, y0, 0, q)
        
        # there should be a peak at q=1
        idx = np.argmax(T)
        self.assertAlmostEqual(q[idx], 1, places=1)

        y1 = spherical_jn(1, 5 * r)
        T = _sphbes_transform(r, y1, 1, q)
        # there should be a peak at q=5
        idx = np.argmax(T)
        self.assertAlmostEqual(q[idx], 5, places=1)

    # @unittest.skip('Development use')
    def test_sphbes_t_orbfile(self):
        import matplotlib.pyplot as plt
        forb = '/root/abacus-develop/numerical_orbitals/SG15-Version1p0__AllOrbitals-VersionJY'\
        '/Cl/primitive_jy/Cl_gga_10au_100Ry_30s30p29d.orb'

        nao = read_nao(forb)
        r = nao['dr'] * np.arange(nao['nr'])
        chi = nao['chi']
        q = np.linspace(0, 20, 1000) # r in bohr, then q in 1/Bohr

        lmax = len(chi) - 1
        nzeta_max = max([len(chil) for chil in chi])
        fig, ax = plt.subplots(lmax+1, nzeta_max, figsize=(5*nzeta_max, 10))
        for l, chil in enumerate(chi):
            for iz, chilz in enumerate(chil):
                ax[l, iz].plot(q**2, np.abs(q**2 * _sphbes_transform(r, chilz, l, q)))
                ax[l, iz].set_title('l=%d, zeta=%d'%(l, iz))
                ax[l, iz].axhline(0, color='k', lw=0.5)
                ax[l, iz].set_xlabel('$\mathbf{k}^2$ (Bohr$^{-2}$)')
                ax[l, iz].set_ylabel('$|\mathbf{k}^2 \chi(|\mathbf{k}|)|$')
        plt.tight_layout()
        plt.savefig(f'{os.path.basename(forb)}.png')
        # plt.show()
        plt.close()

if __name__ == '__main__':
    unittest.main()


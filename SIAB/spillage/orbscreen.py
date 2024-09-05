"""for evaluate quality of orbital"""
import numpy as np
from SIAB.spillage.radial import kinetic 
from SIAB.spillage.orbio import read_nao

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


############################################################
#                       Test
############################################################
import unittest

class _TestScreen(unittest.TestCase):


    def test_screen(self):
        T_In = screen('./testfiles/In_gga_10au_100Ry_3s3p3d2f.orb', item="T")
        self.assertEqual([len(T_l) for T_l in T_In], [3, 3, 3, 2])


if __name__ == '__main__':
    unittest.main()


import unittest
import SIAB.io.file_manage as sifm
import re
import os
class TestFileManage(unittest.TestCase):

    def test_skip(self):
        ov_pattern = r"^(orb_matrix\.)([01])(\.dat)$"
        nv_pattern = r"^(orb_matrix_rcut)([0-9]+)(deriv)([01])(\.dat)$"
        self.assertTrue(re.match(ov_pattern, "orb_matrix.0.dat"))
        self.assertTrue(re.match(ov_pattern, "orb_matrix.1.dat"))
        self.assertTrue(re.match(nv_pattern, "orb_matrix_rcut1deriv0.dat"))
        self.assertTrue(re.match(nv_pattern, "orb_matrix_rcut1deriv1.dat"))
        self.assertTrue(re.match(nv_pattern, "orb_matrix_rcut10deriv0.dat"))
        self.assertTrue(re.match(nv_pattern, "orb_matrix_rcut10deriv1.dat"))

        # create orb_matrixdat and orb_matrix.1.dat in the folder
        with open("orb_matrix.0.dat", "w") as f:
            f.write("orb_matrixdat")
        with open("orb_matrix.1.dat", "w") as f:
            f.write("orb_matrix.1.dat")
        self.assertTrue(sifm.skip("."))
        # remove orb_matrixdat and orb_matrix.1.dat
        os.remove("orb_matrix.0.dat")
        self.assertFalse(sifm.skip("."))
        os.remove("orb_matrix.1.dat")
        self.assertFalse(sifm.skip("."))
        # create orb_matrix_rcut1deriv0.dat and orb_matrix_rcut1deriv1.dat in the folder
        with open("orb_matrix_rcut1deriv0.dat", "w") as f:
            f.write("orb_matrix_rcut1deriv0.dat")
        with open("orb_matrix_rcut1deriv1.dat", "w") as f:
            f.write("orb_matrix_rcut1deriv1.dat")
        self.assertTrue(sifm.skip("."))
        # remove orb_matrix_rcut1deriv0.dat and orb_matrix_rcut1deriv1.dat
        os.remove("orb_matrix_rcut1deriv0.dat")
        self.assertFalse(sifm.skip("."))
        os.remove("orb_matrix_rcut1deriv1.dat")
        self.assertFalse(sifm.skip("."))
        # create orb_matrix_* files for rcut from 7 to 10
        for i in range(7, 11):
            with open("orb_matrix_rcut%sderiv0.dat"%str(i), "w") as f:
                f.write("orb_matrix_rcut%sderiv0.dat"%str(i))
            with open("orb_matrix_rcut%sderiv1.dat"%str(i), "w") as f:
                f.write("orb_matrix_rcut%sderiv1.dat"%str(i))
        self.assertTrue(sifm.skip("."))
        # remove all deriv0
        for i in range(7, 11):
            os.remove("orb_matrix_rcut%sderiv0.dat"%str(i))
            self.assertFalse(sifm.skip("."))
        # remove all deriv1
        for i in range(7, 11):
            os.remove("orb_matrix_rcut%sderiv1.dat"%str(i))
            self.assertFalse(sifm.skip("."))


if __name__ == "__main__":
    unittest.main()
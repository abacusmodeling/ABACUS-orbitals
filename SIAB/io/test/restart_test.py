import unittest
import SIAB.io.restart as restart
import time
import os
import re
class TestRestart(unittest.TestCase):

    def test_checkpoint(self):
        user_settings = {
            "element": "H",
            "Ecut": 10.0
        }
        rcut = 10.0
        orbital_config = "2s1p" # DZP
        src = "./SIAB/io/test/support/"
        dst = "./%s"%(time.strftime("%Y%m%d%H%M%S"))
        restart.checkpoint(
            src=src, 
            dst=dst, 
            user_settings=user_settings, 
            rcut=rcut, 
            orbital_config=orbital_config)
        self.assertTrue(os.path.exists(dst+"/SIAB_INPUT"))
        self.assertTrue(os.path.exists(dst+"/spillage.dat"))
        self.assertTrue(os.path.exists(dst+"/ORBITAL_PLOTU.dat"))
        self.assertTrue(os.path.exists(dst+"/ORBITAL_RESULTS.txt"))
        self.assertTrue(os.path.exists(dst+"/H_gga_10.0Ry_10.0au_2s1p.orb"))
        self.assertTrue(os.path.exists(dst+"/ORBITAL_1U.dat"))
        time.sleep(10)
        os.system("rm %s/H_gga_10.0Ry_10.0au_2s1p.orb"%dst)
        restart.checkpoint(
            src=dst,
            dst=src,
            user_settings=user_settings,
            rcut=rcut,
            orbital_config=orbital_config)
        self.assertTrue(os.path.exists(src+"SIAB_INPUT"))
        self.assertTrue(os.path.exists(src+"spillage.dat"))
        self.assertTrue(os.path.exists(src+"ORBITAL_PLOTU.dat"))
        self.assertTrue(os.path.exists(src+"ORBITAL_RESULTS.txt"))
        self.assertTrue(os.path.exists(src+"H_gga_10.0Ry_10.0au_2s1p.orb"))
        self.assertTrue(os.path.exists(src+"ORBITAL_1U.dat"))
        os.system("rm %s"%src+"H_gga_10.0Ry_10.0au_2s1p.orb")
        os.system("rm -rf %s"%dst)

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
        self.assertTrue(restart.abacus_skip("."))
        # remove orb_matrixdat and orb_matrix.1.dat
        os.remove("orb_matrix.0.dat")
        self.assertFalse(restart.abacus_skip("."))
        os.remove("orb_matrix.1.dat")
        self.assertFalse(restart.abacus_skip("."))
        # create orb_matrix_rcut1deriv0.dat and orb_matrix_rcut1deriv1.dat in the folder
        with open("orb_matrix_rcut1deriv0.dat", "w") as f:
            f.write("orb_matrix_rcut1deriv0.dat")
        with open("orb_matrix_rcut1deriv1.dat", "w") as f:
            f.write("orb_matrix_rcut1deriv1.dat")
        self.assertTrue(restart.abacus_skip("."))
        # remove orb_matrix_rcut1deriv0.dat and orb_matrix_rcut1deriv1.dat
        os.remove("orb_matrix_rcut1deriv0.dat")
        self.assertFalse(restart.abacus_skip("."))
        os.remove("orb_matrix_rcut1deriv1.dat")
        self.assertFalse(restart.abacus_skip("."))
        # create orb_matrix_* files for rcut from 7 to 10
        for i in range(7, 11):
            with open("orb_matrix_rcut%sderiv0.dat"%str(i), "w") as f:
                f.write("orb_matrix_rcut%sderiv0.dat"%str(i))
            with open("orb_matrix_rcut%sderiv1.dat"%str(i), "w") as f:
                f.write("orb_matrix_rcut%sderiv1.dat"%str(i))
        self.assertTrue(restart.abacus_skip("."))
        # remove all deriv0
        for i in range(7, 11):
            os.remove("orb_matrix_rcut%sderiv0.dat"%str(i))
            self.assertFalse(restart.abacus_skip("."))
        # remove all deriv1
        for i in range(7, 11):
            os.remove("orb_matrix_rcut%sderiv1.dat"%str(i))
            self.assertFalse(restart.abacus_skip("."))


if __name__ == '__main__':
    unittest.main()
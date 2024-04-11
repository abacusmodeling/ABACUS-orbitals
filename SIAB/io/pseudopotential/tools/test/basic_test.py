import unittest
import SIAB.io.pseudopotential.tools.basic as basic

class TestBasic(unittest.TestCase):

    def test_is_numeric_data(self):

        self.assertTrue(basic.is_numeric_data("1 2 3"))
        self.assertTrue(basic.is_numeric_data("1.0 2.0 3.0"))
        self.assertTrue(basic.is_numeric_data("1.0e-1 2.0e-1 3.0e-1"))
    
    def test_decompose_data(self):

        self.assertListEqual(basic.decompose_data("1 2 3"), [1, 2, 3])
        self.assertListEqual(basic.decompose_data("1.0 2.0 3.0"), [1.0, 2.0, 3.0])
        self.assertListEqual(basic.decompose_data("1.0e-1 2.0e-1 3.0e-1"), [0.1, 0.2, 0.3])
        self.assertEqual(basic.decompose_data("1"), 1)
        self.assertEqual(basic.decompose_data("1.0"), 1.0)
        self.assertEqual(basic.decompose_data("1.0e-1"), 0.1)
        self.assertEqual(basic.decompose_data("1.0e-1 2.0e-1 3.0e-1"), [0.1, 0.2, 0.3])
    
    def test_zeta_notation_toorbitalconfig(self):

        result = basic.orbconf_fromxzyp("DZP", minimal_basis=[1, 1])
        self.assertEqual(result, "2s2p1d")
        result = basic.orbconf_fromxzyp("TZDP", minimal_basis=[1, 1])
        self.assertEqual(result, "3s3p2d")
        result = basic.orbconf_fromxzyp("TZ5P", minimal_basis=[1, 1])
        self.assertEqual(result, "3s3p5d")
        result = basic.orbconf_fromxzyp("TZ5P", minimal_basis=[1, 1], as_list=True)
        self.assertListEqual(result, [3, 3, 5])
        result = basic.orbconf_fromxzyp("TZ5P", minimal_basis=[["1S"], ["1P"]])
        self.assertEqual(result, "3s3p5d")
        result = basic.orbconf_fromxzyp("TZ5P", minimal_basis=[["1S"], ["1P"]], as_list=True)
        self.assertListEqual(result, [3, 3, 5])
        result = basic.orbconf_fromxzyp("TZ5P", minimal_basis=[["1S"], ["1P", "2P"]])
        self.assertEqual(result, "3s6p5d")
        result = basic.orbconf_fromxzyp("TZ5P", minimal_basis=[["1S"], ["1P", "2P"]], as_list=True)
        self.assertListEqual(result, [3, 6, 5])
        # test the Mn case: 3d5 4s2, without p orbital
        result = basic.orbconf_fromxzyp("DZP", minimal_basis=[["4S"], [], ["3D"]])
        self.assertEqual(result, "2s2d1p")
        result = basic.orbconf_fromxzyp("DZP", minimal_basis=[["4S"], [], ["3D"]], as_list=True)
        #                             s  p  d
        self.assertListEqual(result, [2, 1, 2])
        result = basic.orbconf_fromxzyp("TZ5P", minimal_basis=[["4S"], [], ["3D"]], as_list=True)
        #                             s  p  d
        self.assertListEqual(result, [3, 5, 3])
        result = basic.orbconf_fromxzyp("TZ5P", minimal_basis=[["4S"], [], ["3D"]])
        self.assertEqual(result, "3s3d5p")


if __name__ == "__main__":
    unittest.main()
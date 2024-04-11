import unittest
import SIAB.io.pseudopotential.api as siapi

class TestAPI(unittest.TestCase):

    def test_parse(self):
        fname = "./SIAB/io/pseudopotential/tools/test/support/Mn_adc.upf"
        parsed = siapi.parse(fname=fname)
        self.assertEqual(parsed["PP_HEADER"]["attrib"]["element"], "Mn")

    def test_towards_siab(self):
        fname = "./SIAB/io/pseudopotential/tools/test/support/Mn_adc.upf"
        info = siapi.towards_siab(fname=fname)
        self.assertEqual(info["element"], "Mn")
        self.assertEqual(info["val_conf"], [['4S'], [], ['3D']])

if __name__ == "__main__":
    unittest.main()
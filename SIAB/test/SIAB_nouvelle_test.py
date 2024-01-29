import unittest
import SIAB.SIAB_nouvelle as siab

class TestSIABNouvelle(unittest.TestCase):

    def test_initialize(self):
        result = siab.initialize(fname="./SIAB_INPUT")
        self.assertEqual(len(result), 4)
        self.assertEqual(type(result[0]), dict)
        self.assertEqual(type(result[1]), list)
        self.assertEqual(type(result[2]), list)
        self.assertEqual(type(result[3]), list)

if __name__ == "__main__":
    unittest.main()
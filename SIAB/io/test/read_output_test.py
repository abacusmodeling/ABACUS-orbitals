import unittest
import SIAB.io.read_output as ro

class TestReadOutput(unittest.TestCase):

    def test_read_energy(self):
        energy = -1
        energy = ro.read_energy(folder = "./SIAB/io/test/support",
                                calculation = "scf",
                                suffix = "unittest")
        self.assertNotEqual(energy, -1)
        print("energy = %s eV"%energy)

if __name__ == "__main__":
    unittest.main()
import unittest
import SIAB.interface.submit as submit
import numpy as np

class TestSubmit(unittest.TestCase):

    def test_morse_potential_fitting(self):
        def morse_potential(r, De, a, re):
            return De * (1.0 - np.exp(-a*(r-re)))**2.0

        r = np.linspace(1.5, 5.0, 100) # Angstrom
        De = 2.5 # eV
        a = 1.0
        re = 2.0 # Angstrom
        y = morse_potential(r, De, a, re)
        x = r
        e_dis, _, bleq, _ = submit._morse_potential_fitting(x, y)
        self.assertAlmostEqual(e_dis, De, places=2)
        self.assertAlmostEqual(bleq, re, places=2)

        # add noise
        nprecision = 3
        y = y + np.random.normal(0, 10**(-nprecision), 100)
        e_dis, _, bleq, _ = submit._morse_potential_fitting(x, y)
        self.assertAlmostEqual(e_dis, De, places=nprecision-1)
        self.assertAlmostEqual(bleq, re, places=nprecision-1)

    def test_get_blrange(self):
        def morse_potential(r, De, a, re):
            return De * (1.0 - np.exp(-a*(r-re)))**2.0
        r = np.linspace(1.5, 5.0, 100)
        De = 2.5
        a = 1.0
        re = 2.0
        y = morse_potential(r, De, a, re)
        x = r
        e_dis, _, bleq, e0 = submit._morse_potential_fitting(x, y)
        blrange = submit._summarize_blrange(bl0=bleq, ener0=e0, bond_lengths=x, energies=y, ener_thr=1.5)
        self.assertListEqual(blrange, [1.5, 1.75, 1.99, 2.77, 3.48])

if __name__ == "__main__":
    unittest.main()
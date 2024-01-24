import SIAB.database as db
import unittest

class TestDatabase(unittest.TestCase):

    def test_unit_conversion(self):
        self.assertEqual(db.unit_conversion("Ha", "Ha"), 1)
        self.assertAlmostEqual(db.unit_conversion("Ha", "eV"), 27.21138602, 5)
        self.assertAlmostEqual(db.unit_conversion("Ha", "kcal/mol"), 627.509469, 5)
        self.assertAlmostEqual(db.unit_conversion("eV", "Ha"), 0.0367493, 5)
        self.assertEqual(db.unit_conversion("eV", "eV"), 1)
        self.assertAlmostEqual(db.unit_conversion("eV", "kcal/mol"), 23.060548, 5)
        self.assertAlmostEqual(db.unit_conversion("kcal/mol", "Ha"), 0.00159362, 5)
        self.assertAlmostEqual(db.unit_conversion("kcal/mol", "eV"), 0.0433641, 5)
        self.assertEqual(db.unit_conversion("kcal/mol", "kcal/mol"), 1)

    def test_orbital_configration2list(self):
        self.assertEqual(db.orbital_configration2list("2s", 0), [2])
        self.assertEqual(db.orbital_configration2list("2s", 1), [2, 0])
        self.assertEqual(db.orbital_configration2list("2s", 2), [2, 0, 0])
        self.assertEqual(db.orbital_configration2list("2s2p1d", 2), [2, 2, 1])
        self.assertEqual(db.orbital_configration2list("2s2p1d", 3), [2, 2, 1, 0])
        self.assertEqual(db.orbital_configration2list("2s1d", 2), [2, 0, 1])

if __name__ == "__main__":
    unittest.main()
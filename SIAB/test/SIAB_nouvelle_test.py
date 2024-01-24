import unittest
import SIAB.SIAB_nouvelle as siab

class TestSIABNouvelle(unittest.TestCase):

    def test_keywords_translate(self):

        self.assertEqual(siab.keywords_translate("Ecut"), "ecutwfc")
        self.assertEqual(siab.keywords_translate("Rcut"), "bessel_nao_rcut")
        self.assertEqual(siab.keywords_translate("Pseudo_dir"), "pseudo_dir")
        self.assertEqual(siab.keywords_translate("sigma"), "smearing_sigma")

    def test_unpack_siab_settings(self):

        user_settings = {
            "EXE_mpi": ['mpiexec.hydra', '-np', 20],
            "EXE_pw": ['ABACUS'],
            "element": "Si",
            "Ecut": 100,
            "Rcut": [6, 7],
            "Pseudo_dir": ['/gpfs/home/nic/wszhang/abacus/delta_dft/CIF_POT/SG15_ONCV_PBE-1.0'],
            "Pseudo_name": ['Si_ONCV_PBE-1.0.upf'],
            "sigma": 0.01,
            "STRU1": ['dimer', 8, 2, 1, 1.8, 2.0, 2.3, 2.8, 3.8],
            "STRU2": ['trimer', 10, 2, 1, 1.9, 2.1, 2.6],
            "max_steps": [200],
            "Level1": ['STRU1', 4, 'none', '1s1p'],
            "Level2": ['STRU1', 4, 'fix', '2s2p1d'],
            "Level3": ['STRU2', 6, 'fix', '3s3p2d'],
            "Save1": ['Level1', 'Z'],
            "Save2": ['Level2', 'DZP'],
            "Save3": ['Level3', 'TZDP']
        }

        result = siab.unpack_siab_settings(user_settings)
        self.assertListEqual(result[0], ['dimer', 'trimer'])
        self.assertListEqual(result[1], [[1.8, 2.0, 2.3, 2.8, 3.8], [1.9, 2.1, 2.6]])
        self.assertListEqual(result[2], [
            {
                'ecutwfc': 100,
                'bessel_nao_rcut': '6 7',
                'pseudo_dir': '/gpfs/home/nic/wszhang/abacus/delta_dft/CIF_POT/SG15_ONCV_PBE-1.0',
                'smearing_sigma': 0.01,
                'nbands': 8,
                'lmaxmax': 2,
                'nspin': 1
            },
            {
                'ecutwfc': 100,
                'bessel_nao_rcut': '6 7',
                'pseudo_dir': '/gpfs/home/nic/wszhang/abacus/delta_dft/CIF_POT/SG15_ONCV_PBE-1.0',
                'smearing_sigma': 0.01,
                'nbands': 10,
                'lmaxmax': 2,
                'nspin': 1
            }
        ])

    def test_archive(self):
        pass

if __name__ == "__main__":
    unittest.main()
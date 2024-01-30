import unittest
import SIAB.io.read_input as ri

class TestReadInput(unittest.TestCase):

    """use example input as test material
    ./SIAB/example_Si/SIAB_INPUT
    """
    def test_parse(self):

        result = ri.parse("SIAB/example_Si/SIAB_INPUT")

        self.assertListEqual(result["EXE_mpi"], ['mpirun', '-np', 1])
        self.assertListEqual(result["EXE_pw"], ['abacus'])
        self.assertEqual(result["element"], "Si")
        self.assertEqual(result["Ecut"], 100)
        self.assertListEqual(result["Rcut"], [6, 7])
        self.assertListEqual(result["Pseudo_dir"], ['/root/abacus-develop/pseudopotentials/SG15_ONCV_v1.0_upf'])
        self.assertListEqual(result["Pseudo_name"], ['Si_ONCV_PBE-1.0.upf'])
        self.assertEqual(result["sigma"], 0.01)
        self.assertListEqual(result["STRU1"], ['dimer', 8, 2, 1, 1.8, 2.0, 2.3, 2.8, 3.8])
        self.assertListEqual(result["STRU2"], ['trimer', 10, 2, 1, 1.9, 2.1, 2.6])
        self.assertListEqual(result["max_steps"], [200])
        self.assertListEqual(result["Level1"], ['STRU1', 4, 'none', '1s1p'])
        self.assertListEqual(result["Level2"], ['STRU1', 4, 'fix', '2s2p1d'])
        self.assertListEqual(result["Level3"], ['STRU2', 6, 'fix', '3s3p2d'])
        self.assertListEqual(result["Save1"], ['Level1', 'Z'])
        self.assertListEqual(result["Save2"], ['Level2', 'DZP'])
        self.assertListEqual(result["Save3"], ['Level3', 'TZDP'])

    def test_default(self):

        result = ri.parse("SIAB/example_Si/SIAB_INPUT")
        result, optimizer_path = ri.default(result)
        self.assertEqual(result["EXE_opt"], "/opt_orb_pytorch_dpsi/main.py (default)")
        self.assertEqual(optimizer_path, "/opt_orb_pytorch_dpsi")

    def test_wash(self):

        result = ri.parse("SIAB/example_Si/SIAB_INPUT")
        result, optimizer_path = ri.default(result)
        result = ri.wash(result)
        self.assertEqual(result["EXE_opt"], "/opt_orb_pytorch_dpsi/main.py (default)")
        self.assertEqual(optimizer_path, "/opt_orb_pytorch_dpsi")
        self.assertEqual(result["EXE_pw"], "abacus")
        self.assertEqual(result["Pseudo_dir"], "/root/abacus-develop/pseudopotentials/SG15_ONCV_v1.0_upf")
        self.assertEqual(result["Pseudo_name"], "Si_ONCV_PBE-1.0.upf")

    def test_keywords_translate(self):

        self.assertEqual(ri.keywords_translate("Ecut"), "ecutwfc")
        self.assertEqual(ri.keywords_translate("Rcut"), "bessel_nao_rcut")
        self.assertEqual(ri.keywords_translate("Pseudo_dir"), "pseudo_dir")
        self.assertEqual(ri.keywords_translate("sigma"), "smearing_sigma")      


    def test_unpack_siab_settings(self):

        user_settings = {
            "EXE_mpi": ['mpiexec.hydra', '-np', 20],
            "EXE_pw": ['ABACUS'],
            "element": "Si",
            "Ecut": 100,
            "Rcut": [6, 7],
            "Pseudo_dir": '/gpfs/home/nic/wszhang/abacus/delta_dft/CIF_POT/SG15_ONCV_PBE-1.0',
            "Pseudo_name": 'Si_ONCV_PBE-1.0.upf',
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

        result = ri.unpack_siab_settings(user_settings)
        self.assertListEqual(result[0], ['dimer', 'trimer'])
        self.assertListEqual(result[1], [[1.8, 2.0, 2.3, 2.8, 3.8], [1.9, 2.1, 2.6]])
        self.assertDictEqual(result[2][0], {
                'ecutwfc': 100,
                'bessel_nao_rcut': '6 7',
                'pseudo_dir': '/gpfs/home/nic/wszhang/abacus/delta_dft/CIF_POT/SG15_ONCV_PBE-1.0',
                'smearing_sigma': 0.01,
                'nbands': 8,
                'lmaxmax': 2,
                'nspin': 1
            })
        self.assertDictEqual(result[2][1],
            {
                'ecutwfc': 100,
                'bessel_nao_rcut': '6 7',
                'pseudo_dir': '/gpfs/home/nic/wszhang/abacus/delta_dft/CIF_POT/SG15_ONCV_PBE-1.0',
                'smearing_sigma': 0.01,
                'nbands': 10,
                'lmaxmax': 2,
                'nspin': 1
            }
        )
    def test_abacus_params(self):
        abacus_params = ri.abacus_params()
        self.assertGreater(len(abacus_params), 0)

if __name__ == "__main__":
    unittest.main()
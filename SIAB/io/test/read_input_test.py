import unittest
import SIAB.io.read_input as ri

class TestReadInput(unittest.TestCase):

    """use example input as test material
    ./SIAB/example_Si/SIAB_INPUT
    """
    def test_parse(self):

        result = ri.parse("SIAB/example_Si/SIAB_INPUT")
        #print(result)

    def test_keywords_translate(self):

        self.assertEqual(ri.keywords_translate("Ecut"), "ecutwfc")
        self.assertEqual(ri.keywords_translate("Rcut"), "bessel_nao_rcut")
        self.assertEqual(ri.keywords_translate("Pseudo_dir"), "pseudo_dir")
        self.assertEqual(ri.keywords_translate("sigma"), "smearing_sigma")      

    def test_unpack_siab_settings(self):

        result = ri.parse("SIAB/example_Si/SIAB_INPUT")
        result = ri.unpack_siab_input(result, "Si", [["1S"], ["1P"]])
        self.assertEqual(len(result), 6)
        self.assertListEqual(result[0], ['dimer', 'trimer'])
        self.assertListEqual(result[1], [[1.8, 2.0, 2.3, 2.8, 3.8], [1.9, 2.1, 2.6]])
        self.assertListEqual(result[2], [
            {'pseudo_dir': '/root/abacus-develop/pseudopotentials/SG15_ONCV_v1.0_upf', 
             'ecutwfc': 100, 'bessel_nao_rcut': [6, 7], 'smearing_sigma': 0.01, 
             'nbands': 8, 'lmaxmax': 2, 'nspin': 1}, 
             {'pseudo_dir': '/root/abacus-develop/pseudopotentials/SG15_ONCV_v1.0_upf', 
              'ecutwfc': 100, 'bessel_nao_rcut': [6, 7], 'smearing_sigma': 0.01, 
              'nbands': 10, 'lmaxmax': 2, 'nspin': 1}
              ])
        self.assertDictEqual(result[3], {'optimizer': 'pytorch.SWAT', 'max_steps': [200], 
                                         'spillage_coeff': [0.5, 0.5], 
                                         'orbitals': [
                                        {'zeta_notation': 'Z', 'shape': 'dimer', 
                                         'nbands_ref': 4, 'orb_ref': 'none'}, 
                                        {'zeta_notation': 'DZP', 'shape': 'dimer', 
                                         'nbands_ref': 4, 'orb_ref': 'Z'}, 
                                        {'zeta_notation': 'TZDP', 'shape': 'trimer', 
                                         'nbands_ref': 6, 'orb_ref': 'DZP'}
                                         ]})
        self.assertDictEqual(result[4], {'environment': '', 
                                         'mpi_command': 'mpirun -np 1', 
                                         'abacus_command': 'abacus'})
        self.assertDictEqual(result[5], {'element': 'Si', 
                                         'pseudo_dir': '/root/abacus-develop/pseudopotentials/SG15_ONCV_v1.0_upf', 
                                         'pseudo_name': 'Si_ONCV_PBE-1.0.upf'})

    def test_abacus_params(self):
        abacus_params = ri.abacus_params()
        self.assertGreater(len(abacus_params), 0)

    def test_compatibility_convert(self):
        clean_oldversion_input = {
            'EXE_mpi': 'mpirun -np 1', 
            'EXE_pw': 'abacus --version', 
            'element': 'Si', 
            'Ecut': 100, 
            'Rcut': [6, 7], 
            'Pseudo_dir': '/root/abacus-develop/pseudopotentials/SG15_ONCV_v1.0_upf', 
            'Pseudo_name': 'Si_ONCV_PBE-1.0.upf', 
            'sigma': 0.01, 
            'STRU1': ['dimer', 8, 2, 1, 1.8, 2.0, 2.3, 2.8, 3.8], 
            'STRU2': ['trimer', 10, 2, 1, 1.9, 2.1, 2.6], 
            'max_steps': [200], 
            'Level1': ['STRU1', 4, 'none', '1s1p'], 
            'Level2': ['STRU1', 4, 'fix', '2s2p1d'], 
            'Level3': ['STRU2', 6, 'fix', '3s3p2d'], 
            'Save1': ['Level1', 'SZ'], 
            'Save2': ['Level2', 'DZP'], 
            'Save3': ['Level3', 'TZDP'], 
            'EXE_opt': '/opt_orb_pytorch_dpsi/main.py (default)', 
            'EXE_env': '', 
            'environment': '', 
            'mpi_command': 'mpirun -np 1', 
            'abacus_command': 'abacus', 
            'optimizer': 'pytorch.SWAT', 
            'spillage_coeff': [0.5, 0.5]
        }
        result = ri.compatibility_convert(clean_oldversion_input)
        clean_oldversion_input = {
            'EXE_mpi': 'mpirun -np 1', 
            'EXE_pw': 'abacus --version', 
            'element': 'Si', 
            'Ecut': 100, 
            'Rcut': [6, 7], 
            'Pseudo_dir': '/root/abacus-develop/pseudopotentials/SG15_ONCV_v1.0_upf', 
            'Pseudo_name': 'Si_ONCV_PBE-1.0.upf', 
            'sigma': 0.01, 
            'STRU1': ['dimer', 8, 2, 1, "auto"], 
            'STRU2': ['trimer', 10, 2, 1, 1.9, 2.1, 2.6], 
            'max_steps': [200], 
            'Level1': ['STRU1', 4, 'none', '1s1p'], 
            'Level2': ['STRU1', 4, 'fix', '2s2p1d'], 
            'Level3': ['STRU2', 6, 'fix', '3s3p2d'], 
            'Save1': ['Level1', 'SZ'], 
            'Save2': ['Level2', 'DZP'], 
            'Save3': ['Level3', 'TZDP'], 
            'EXE_opt': '/opt_orb_pytorch_dpsi/main.py (default)', 
            'EXE_env': '', 
            'environment': '', 
            'mpi_command': 'mpirun -np 1', 
            'abacus_command': 'abacus', 
            'optimizer': 'pytorch.SWAT', 
            'spillage_coeff': [0.5, 0.5]
        }
        result = ri.compatibility_convert(clean_oldversion_input)

if __name__ == "__main__":
    unittest.main()
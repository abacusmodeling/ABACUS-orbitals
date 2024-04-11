import unittest
import SIAB.driver.front as front

class TestFront(unittest.TestCase):

    def test_initialzie(self):
        result = front.initialize(version="0.1.0",
                                  fname="./SIAB/driver/test/support/SIAB_INPUT",
                                  pseudopotential_check=False)
        self.assertEqual(len(result), 6)
        self.assertListEqual(result[0], ["dimer", "trimer"])
        self.assertListEqual(result[1], [['auto'], [1.9, 2.1, 2.6]])
        self.assertListEqual(result[2], [
            {'pseudo_dir': '/root/abacus-develop/pseudopotentials/SG15_ONCV_v1.0_upf', 
             'ecutwfc': 100, 
             'bessel_nao_rcut': [6, 7], 
             'smearing_sigma': 0.01, 
             'nbands': 8, 
             'lmaxmax': 2, 
             'nspin': 1}, 
            {'pseudo_dir': '/root/abacus-develop/pseudopotentials/SG15_ONCV_v1.0_upf', 
             'ecutwfc': 100, 
             'bessel_nao_rcut': [6, 7], 
             'smearing_sigma': 0.01, 
             'nbands': 10, 
             'lmaxmax': 2, 
             'nspin': 1}
        ])
        self.assertDictEqual(result[3], {
            'optimizer': 'pytorch.SWAT', 
            'max_steps': [200], 
            'spill_coefs': [2.0, 1.0], 
            'orbitals': [
                {'nzeta': [1, 1], 'nzeta_from': None, 'nbands_ref': 4, 'folder': 0}, 
                {'nzeta': [2, 2, 1], 'nzeta_from': [1, 1], 'nbands_ref': 4, 'folder': 0}, 
                {'nzeta': [3, 3, 2], 'nzeta_from': [2, 2, 1], 'nbands_ref': 6, 'folder': 1}
            ]
        })
        self.assertDictEqual(result[4], {
            'environment': '', 
            'mpi_command': 'mpirun -np 8', 
            'abacus_command': 'abacus'}
        )
        self.assertDictEqual(result[5], {
            'element': 'Si', 
            'pseudo_dir': '/root/abacus-develop/pseudopotentials/SG15_ONCV_v1.0_upf', 
            'pseudo_name': 'Si_ONCV_PBE-1.0.upf'}
        )

    def test_abacus(self):
        # not need to run because it is a simple wrapper
        pass

    

if __name__ == "__main__":
    unittest.main()
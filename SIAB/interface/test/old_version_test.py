import unittest
import SIAB.interface.old_version as old_version

class TestToBc(unittest.TestCase):

    def test_bc_parameters(self):
        result = old_version.ov_parameters(element="H", 
                                           orbital_config=[2, 2, 1], 
                                           bessel_nao_rcut=2.0, 
                                           lmax=2,
                                           ecutwfc=2.0, 
                                           dr=0.01, 
                                           opt_maxsteps=1000, 
                                           lr=0.03, 
                                           calc_kinetic_ener=False, 
                                           calc_smooth=True)
        self.assertDictEqual(result, 
                             {'Nt_all': ['H'], 
                              'Nu': {'H': [2, 2, 1]}, 
                              'Rcut': {'H': 2.0}, 
                              'dr': {'H': 0.01}, 
                              'Ecut': {'H': 2.0}, 
                              'lr': 0.03, 
                              'cal_T': False, 
                              'cal_smooth': True, 'max_steps': 1000})

    def test_merge_param(self):
        self.assertEqual(old_version.merge_ovparam(
            [
                {"Nt_all": "H", "Nu": [2, 2, 1], "Rcut": 2.0, "dr": 0.01, "Ecut": 2.0,
                 "lr": 0.03, "cal_T": False, "cal_smooth": True, "max_steps": 1000},
                {"Nt_all": "He", "Nu": [3, 2, 1], "Rcut": 3.0, "dr": 0.02, "Ecut": 3.0,
                 "lr": 0.03, "cal_T": False, "cal_smooth": True, "max_steps": 1000}
            ]),
                {"Nt_all": ["H", "He"], "lr": 0.03, "cal_T": False, "cal_smooth": True, "max_steps": 1000,
                  "Nu": {"H": [2, 2, 1], "He": [3, 2, 1]}, "Rcut": {"H": 2.0, "He": 3.0},
                  "dr": {"H": 0.01, "He": 0.02}, "Ecut": {"H": 2.0, "He": 3.0}})
        """want a ValueError to raise if bcparams have different values for lr, cal_T, cal_smooth and max_steps"""
        with self.assertRaises(ValueError):
            old_version.merge_ovparam([{"Nt_all": "H", "Nu": [2, 2, 1], "Rcut": 2.0, "dr": 0.01, "Ecut": 2.0,
                          "lr": 0.03, "cal_T": False, "cal_smooth": True, "max_steps": 1000},
                         {"Nt_all": "He", "Nu": [3, 2, 1], "Rcut": 3.0, "dr": 0.02, "Ecut": 3.0,
                          "lr": 0.04, "cal_T": False, "cal_smooth": True, "max_steps": 1000}])
        with self.assertRaises(ValueError):
            old_version.merge_ovparam([{"Nt_all": "H", "Nu": [2, 2, 1], "Rcut": 2.0, "dr": 0.01, "Ecut": 2.0,
                          "lr": 0.03, "cal_T": False, "cal_smooth": True, "max_steps": 1000},
                         {"Nt_all": "He", "Nu": [3, 2, 1], "Rcut": 3.0, "dr": 0.02, "Ecut": 3.0,
                          "lr": 0.03, "cal_T": True, "cal_smooth": True, "max_steps": 1000}])
        with self.assertRaises(ValueError):
            old_version.merge_ovparam([{"Nt_all": "H", "Nu": [2, 2, 1], "Rcut": 2.0, "dr": 0.01, "Ecut": 2.0,
                          "lr": 0.03, "cal_T": False, "cal_smooth": True, "max_steps": 1000},
                         {"Nt_all": "He", "Nu": [3, 2, 1], "Rcut": 3.0, "dr": 0.02, "Ecut": 3.0,
                          "lr": 0.03, "cal_T": False, "cal_smooth": False, "max_steps": 1000}])
        with self.assertRaises(ValueError):
            old_version.merge_ovparam([{"Nt_all": "H", "Nu": [2, 2, 1], "Rcut": 2.0, "dr": 0.01, "Ecut": 2.0,
                          "lr": 0.03, "cal_T": False, "cal_smooth": True, "max_steps": 1000},
                         {"Nt_all": "He", "Nu": [3, 2, 1], "Rcut": 3.0, "dr": 0.02, "Ecut": 3.0,
                          "lr": 0.03, "cal_T": False, "cal_smooth": True, "max_steps": 2000}])
    
    def test_ov_reference_states(self):
        result = old_version.ov_reference_states(
            element="H",
            reference_shapes=["dimer", "trimer"],
            bond_lengths=[[0.8, 1.2], [0.8, 1.2, 1.6]],
            orbitals=[
                {'nzeta': [1, 1], 'nzeta_from': None, 
                 'nbands_ref': 4, 'folder': [
                     "Si-dimer-1.4", "Si-dimer-1.6"
                 ]}, 
                {'nzeta': [2, 2, 1], 'nzeta_from': [1, 1], 
                 'nbands_ref': 6, 'folder': [
                     "Si-trimer-1.4", "Si-trimer-1.6", "Si-trimer-1.8"
                 ]}
            ]
        )
        self.assertListEqual(result[0], [[4, 4], [6, 6, 6]])

        result = old_version.ov_reference_states(
            element="Si",
            reference_shapes=["dimer", "trimer"],
            bond_lengths=[[1.4, 1.6], [1.6, 1.8, 2.0]],
            orbitals=[
                {'nzeta': [1, 1], 'nzeta_from': None, 
                 'nbands_ref': "auto", 'folder': [
                     "Si-dimer-1.4", "Si-dimer-1.6"
                 ]}, 
                {'nzeta': [2, 2, 1], 'nzeta_from': [1, 1], 
                 'nbands_ref': 6, 'folder': [
                     "Si-trimer-1.6", "Si-trimer-1.8", "Si-trimer-2.0"
                 ]}
            ]
        )
        self.assertListEqual(result[0], [
            ['Si-dimer-1.4/istate.info', 'Si-dimer-1.6/istate.info'], 
            [6, 6, 6]
            ])

    def test_ov_weights(self):
        reference_states = [
        ['H-dimer-0.8/istate.info', 'H-dimer-1.2/istate.info'], 
        [4, 4], 
        ['H-trimer-0.8/istate.info', 'H-trimer-1.2/istate.info', 'H-trimer-1.6/istate.info']
        ]
        result = old_version.ov_weights(reference_states=reference_states)
        self.assertEqual(len(result), 3) # equal to the number of levels
        for ilevel in range(len(reference_states)):
            self.assertTrue("stru" in result[ilevel].keys())
            stru = result[ilevel]["stru"]
            self.assertListEqual(stru, [1]*len(reference_states[ilevel]))
        
            if "bands_range" in result[ilevel].keys():
                bands_range = result[ilevel]["bands_range"]
                self.assertListEqual(bands_range, reference_states[ilevel])
            elif "bands_file" in result[ilevel].keys():
                bands_file = result[ilevel]["bands_file"]
                self.assertListEqual(bands_file, reference_states[ilevel])
            else:
                self.assertTrue(False)

    def test_ov_c_init(self):
        result = old_version.ov_c_init(orbitals=[
            {
                "nzeta": [2, 1, 1],
                "folder": ["X-dimer-Y1", "X-dimer-Y2"],
                "nbands_ref": 4,
                "nzeta_from": None
            },
            {
                "nzeta": [4, 2, 2, 1],
                "folder": ["X-dimer-Y1", "X-dimer-Y2"],
                "nbands_ref": 4,
                "nzeta_from": [2, 1, 1]
            },
            {
                "nzeta": [6, 3, 3, 2],
                "folder": ["X-trimer-Y1", "X-trimer-Y2", "X-trimer-Y3"],
                "nbands_ref": 6,
                "nzeta_from": [4, 2, 2, 1]
            }
        ])
        self.assertEqual(len(result), 3)
        # test init_from_file
        self.assertFalse(result[0]["init_from_file"])
        self.assertTrue(result[1]["init_from_file"])
        self.assertTrue(result[2]["init_from_file"])
        # test C_init_file
        self.assertFalse("C_init_file" in result[0].keys())
        self.assertTrue("C_init_file" in result[1].keys())
        self.assertTrue("C_init_file" in result[2].keys())
        # test_C_init_file
        self.assertEqual(result[1]["C_init_file"], "Level0.ORBITAL_RESULTS.txt")
        self.assertEqual(result[2]["C_init_file"], "Level1.ORBITAL_RESULTS.txt")
        # test opt_C_read
        self.assertFalse("opt_C_read" in result[0].keys())
        self.assertTrue("opt_C_read" in result[1].keys())
        self.assertTrue("opt_C_read" in result[2].keys())
        self.assertFalse(result[1]["opt_C_read"])
        self.assertFalse(result[2]["opt_C_read"])

    def test_convert(self):
        calculation_setting = {
            "bessel_nao_rcut": [6, 7],
            "ecutwfc": 100
        }
        siab_settings = {
            'optimizer': 'pytorch.SWAT', 
            'max_steps': 200, 
            'spill_coefs': [2.0, 1.0], 
            'orbitals': [
                {'nzeta': [1, 1], 'nzeta_from': None, 
                 'nbands_ref': 4, 
                 'folder': ["Si-dimer-1.4", "Si-dimer-1.6"],
                 'lmax': 2
                 }, 
                {'nzeta': [2, 2, 1], 'nzeta_from': [1, 1], 
                 'nbands_ref': "auto", 
                 'folder': ["Si-trimer-1.4", "Si-trimer-1.6"],
                 'lmax': 2
                 }
            ]
        }
        for result in old_version.convert(calculation_setting=calculation_setting,
                                          siab_settings=siab_settings):
            pass
            #print(result)
        """what printed are
{'file_list': 
    {'origin': [4, 4], 
     'linear': [[4, 4]]}, 
 'info': 
    {'Nt_all': ['Si'], 'Nu': {'Si': [1, 1]}, 
     'Rcut': {'Si': 6}, 'dr': {'Si': 0.01}, 
     'Ecut': {'Si': 100}, 
     'lr': 0.03, 'cal_T': False, 'cal_smooth': True, 'max_steps': 1000}, 
 'weight': {'stru': [1, 1], 'bands_range': [4, 4]}, 
 'C_init_info': {'init_from_file': False}, 
 'V_info': {'init_from_file': True, 'same_band': True}
}

{'file_list': 
    {'origin': ['Si-trimer-1.4/orb_matrix.0.dat', 
                'Si-trimer-1.6/orb_matrix.0.dat'], 
     'linear': [['Si-trimer-1.4/orb_matrix.1.dat', 
                 'Si-trimer-1.6/orb_matrix.1.dat']]}, 
 'info': 
    {'Nt_all': ['Si'], 'Nu': {'Si': [2, 2, 1]}, 
     'Rcut': {'Si': 6}, 'dr': {'Si': 0.01}, 
     'Ecut': {'Si': 100}, 
     'lr': 0.03, 'cal_T': False, 'cal_smooth': True, 'max_steps': 1000}, 
 'weights': 
    {'stru': [1, 1], 
     'bands_file': ['Si-trimer-1.4/istate.info', 
                    'Si-trimer-1.6/istate.info']}, 
 'C_init_info': 
    {'init_from_file': True, 
     'C_init_file': 'Level0.ORBITAL_RESULTS.txt', 
     'opt_C_read': False}, 
 'V_info': {'init_from_file': True, 'same_band': True}
}

{'file_list': 
    {'origin': [4, 4], 
     'linear': [[4, 4]]}, 
 'info': 
    {'Nt_all': ['Si'], 'Nu': {'Si': [1, 1]}, 
     'Rcut': {'Si': 7}, 'dr': {'Si': 0.01}, 
     'Ecut': {'Si': 100}, 
     'lr': 0.03, 'cal_T': False, 'cal_smooth': True, 'max_steps': 1000}, 
 'weights': 
    {'stru': [1, 1], 'bands_range': [4, 4]}, 
 'C_init_info': {'init_from_file': False}, 
 'V_info': {'init_from_file': True, 'same_band': True}
}
... 
        """
if __name__ == '__main__':
    unittest.main()
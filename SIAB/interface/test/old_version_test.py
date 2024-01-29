import unittest
import SIAB.interface.old_version as old_version

class TestToBc(unittest.TestCase):

    def test_bc_parameters(self):
        self.assertEqual(old_version.ov_parameters("H", [2, 2, 1], 2.0, 2.0, 0.01, 1000, 0.03, False, True),
                         {"Nt_all": "H", "Nu": [2, 2, 1], "Rcut": 2.0, "dr": 0.01, "Ecut": 2.0,
                          "lr": 0.03, "cal_T": False, "cal_smooth": True, "max_steps": 1000})
    
    def test_merge_param(self):
        self.assertEqual(old_version.merge_ovparam([{"Nt_all": "H", "Nu": [2, 2, 1], "Rcut": 2.0, "dr": 0.01, "Ecut": 2.0,
                          "lr": 0.03, "cal_T": False, "cal_smooth": True, "max_steps": 1000},
                         {"Nt_all": "He", "Nu": [3, 2, 1], "Rcut": 3.0, "dr": 0.02, "Ecut": 3.0,
                          "lr": 0.03, "cal_T": False, "cal_smooth": True, "max_steps": 1000}]),
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
                        
if __name__ == '__main__':
    unittest.main()
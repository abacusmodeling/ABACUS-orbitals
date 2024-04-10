"""manually control the stdout of some dict, instead of using pprint"""

def kst(info_kst: dict):

    # key: Nt_all
    contents =  "PRINT INFO_KST INFORMATION\n"
    contents += "--------------------------\n"
    contents += "General Information: \n"
    contents += "All atom types: " + ", ".join(info_kst["Nt_all"]) + "\n"
    # key: Nu
    contents += "Orbital configuration for each atom type: \n"
    contents += "Symbol, l: 0, 1, 2, 3, ... \n"
    for key, value in info_kst["Nu"].items():
        contents += f"{key}: " + ", ".join([str(i) for i in value]) + "\n"
    # key: Rcut
    contents += "Realspace cutoff radius (rcut), grid (dr), kinetic cutoff (ecutwfc) and maximal angular"
    contents += "momentum (lmax) for each atom type: \n"
    contents += "%-5s %-5s %-5s %-5s %-5s\n" % ("Atom", "Rcut", "dr", "ecutwfc", "lmax")
    for key in info_kst["Nt_all"]:
        contents += "%-5s %-5.2f %-5.2f %-5.2f %-5d\n" % (key, info_kst["Rcut"][key], info_kst["dr"][key], info_kst["Ecut"][key], info_kst["Nl"][key])
    # key: lr
    contents += "Optimizer Learning Rate: " + str(info_kst["lr"]) + "\n"
    # key: cal_T
    contents += "Including additional kinetic term in Spillage: " + str(info_kst["cal_T"]) + "\n"
    # key: cal_smooth
    contents += "Gaussian smoothing for orbitals at rcut: " + str(info_kst["cal_smooth"]) + "\n"
    # key: max_steps
    contents += "Max steps for optimization: " + str(info_kst["max_steps"]) + "\n"
    # key: Nl
    contents += "lmax for each atom type: \n"
    for key, value in info_kst["Nl"].items():
        contents += f"{key}: " + str(value) + "\n\n"
    
    contents += "Structure specific information:\n"
    # key: Nst
    contents += "Number of reference structure: " + str(info_kst["Nst"]) + "\n"
    # key: Nt
    contents += "Atom type for each reference structure: \n"
    for i in range(len(info_kst["Nt"])):
        contents += f"Structure {i}: " + ", ".join(info_kst["Nt"][i]) + "\n"
    # key: Na
    contents += "Number of atoms for each atom type for each reference structure: \n"
    for i in range(len(info_kst["Na"])):
        contents += f"Structure {i}: "
        for key, value in info_kst["Na"][i].items():
            contents += f"{key}: " + f"{value}" + " "
        contents += "\n"
    # key: Nb
    contents += "Number of bands selected to learn for each reference structure: \n"
    contents += "Struectures: "
    for i in range(len(info_kst["Nb"])):
        contents += f"{i}: " + str(info_kst["Nb"][i]) + " "
    contents += "\n"
    
    contents += "Spherical Bessel function:\n"
    # key: Ne
    contents += "Number of Spherical Bessel functions (Sphbes) for each atom type: \n"
    for key, value in info_kst["Ne"].items():
        contents += f"{key}: " + str(value) + " "
    contents += "\n"
    
    contents += "PRINT INFO_KST INFORMATION END.\n\n"
    return contents

def stru(info_stru: dict):
    
    # key: Nt_all
    contents =  "PRINT INFO_STRU INFORMATION\n"
    contents += "--------------------------\n"

    for i in range(len(info_stru)):
        # loop over all structures
        contents += f"Structure {i}:\n"
        contents += "Number of atoms for each type: \n"
        for key, value in info_stru[i]["Na"].items():
            contents += f"{key}: " + f"{value}" + "\n"
        contents += "Number of bands calculated for present structure: " + str(info_stru[i]["Nb"]) + "\n"
        contents += "Number of bands taken INFO consideration for learning: " + str(info_stru[i]["Nb_true"]) + "\n"
        contents += "Detailed weight information for each band: \n"
        for iw in range(len(info_stru[i]["weight"])):
            contents += "%6s%4d: "%("Band", iw) + "%8.4e" % info_stru[i]["weight"][iw] + "\n"
        contents += "\n"

    contents += "PRINT INFO_STRU INFORMATION END.\n\n"

    return contents

def element(info_element: dict):

    subshells = ["s", "p", "d", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "q", "r", "t", "u", "v", "w", "x", "y", "z"]
    contents =  "PRINT INFO_ELEMENT INFORMATION\n"
    contents += "--------------------------\n"
    contents += "Element-wise information: \n"
    for element, info in info_element.items():
        if info is not None:
            contents += f"Element {element}:\n"
            contents += "nsphbes: " + str(info["Ne"]) + "\n"
            contents += "Number of subshells: " + str(info["Nl"]) + "\n"
            contents += "Orbital configuration: " + ", ".join([str(s)+subshells[i] for i, s in enumerate(info["Nu"])]) + "\n"
            contents += "rcut: " + str(info["Rcut"]) + "\n"
            contents += "dr: " + str(info["dr"]) + "\n"
            contents += "atomic index: " + str(info["index"]) + "\n\n"

    contents += "PRINT INFO_ELEMENT INFORMATION END.\n\n"

    return contents

def opt(info_opt: dict, spillage_coeff: list):

    contents =  "PRINT INFO_OPT INFORMATION\n"
    contents += "--------------------------\n"
    contents += "Optimizer information: \n"
    contents += "Calculate kinetic term: " + str(info_opt["cal_T"]) + "\n"
    contents += "Calculate smooth term: " + str(info_opt["cal_smooth"]) + "\n"
    contents += "Optimizer learning rate: " + str(info_opt["lr"]) + "\n"
    contents += "Max steps: " + str(info_opt["max_steps"]) + "\n"
    contents += "Spillage coefficients of PSI and DPSI terms: \n"
    contents += " ".join(["%.2f" % i for i in spillage_coeff]) + "\n\n"

    contents += "PRINT INFO_OPT INFORMATION END.\n\n"

    return contents

def the_max(info_max: dict):

    contents =  "PRINT INFO_MAX INFORMATION\n"
    contents += "--------------------------\n"
    contents += "The data dimension information for each reference structure: \n"
    for i in range(len(info_max)):
        contents += f"Structure {i}:\n"
        contents += "Number of atom types: " + str(info_max[i]["Nt"]) + "\n"
        contents += "Number of atoms: " + str(info_max[i]["Na"]) + "\n"
        contents += "Number of bands: " + str(info_max[i]["Nb"]) + "\n"
        contents += "Number of Sphbes: " + str(info_max[i]["Ne"]) + "\n"
        contents += "Number of subshells: " + str(info_max[i]["Nl"]) + "\n"
        contents += "Maximal number of magnetic channels: " + str(info_max[i]["Nm"]) + "\n"
        contents += "\n"
        
    contents += "PRINT INFO_MAX INFORMATION END.\n\n"
    return contents

def bundle_print(info_kst: dict, 
                 info_stru: dict, 
                 info_element: dict, 
                 info_opt: dict, 
                 info_max: dict,
                 spillage_coeff: list = [2, 1]):

    contents = kst(info_kst.__dict__)
    contents += stru(info_stru)
    contents += element(info_element)
    contents += opt(info_opt, spillage_coeff)
    contents += the_max(info_max)

    return contents

import unittest
class TestStdout(unittest.TestCase):

    def test_kst(self):
        """
        info_kst:
Nt_all  ['Ce']
Nu      {'Ce': [4, 2, 2, 1]}
Rcut    {'Ce': 8}
dr      {'Ce': 0.01}
Ecut    {'Ce': 100}
lr      0.03
cal_T   False
cal_smooth      True
max_steps       9000
Nl      {'Ce': 4}
Nst     5
Nt      [['Ce'], ['Ce'], ['Ce'], ['Ce'], ['Ce']]
Na      [{'Ce': 2}, {'Ce': 2}, {'Ce': 2}, {'Ce': 2}, {'Ce': 2}]
Nb      [22, 22, 22, 22, 22]
Ne      {'Ce': 25}
"""
        info_kst = {
            "Nt_all": ["Ce"],
            "Nu": {"Ce": [4, 2, 2, 1]},
            "Rcut": {"Ce": 8},
            "dr": {"Ce": 0.01},
            "ecutwfc": {"Ce": 100},
            "lr": 0.03,
            "cal_T": False,
            "cal_smooth": True,
            "max_steps": 9000,
            "Nl": {"Ce": 4},
            "Nst": 5,
            "Nt": [["Ce"], ["Ce"], ["Ce"], ["Ce"], ["Ce"]],
            "Na": [{"Ce": 2}, {"Ce": 2}, {"Ce": 2}, {"Ce": 2}, {"Ce": 2}],
            "Nb": [22, 22, 22, 22, 22],
            "Ne": {"Ce": 25}
        }

        result = kst(info_kst)
        print(result)

    def test_stru(self):
        """info_stru:
[{'Na': {'Ce': 2},
  'Nb': 22,
  'Nb_true': 14,
  'weight': tensor([1.8182e-02, 1.8182e-02, 1.8182e-02, 1.8182e-02, 1.8182e-02, 1.8182e-02,
        1.8182e-02, 1.8182e-02, 1.8182e-02, 1.8154e-02, 1.8154e-02, 3.3398e-05,
        2.0629e-05, 2.1154e-06, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00])},
 {'Na': {'Ce': 2},
  'Nb': 22,
  'Nb_true': 15,
  'weight': tensor([1.8182e-02, 1.8182e-02, 1.8182e-02, 1.8182e-02, 1.8182e-02, 1.8182e-02,
        1.8182e-02, 1.8182e-02, 1.8174e-02, 1.8092e-02, 1.8092e-02, 1.8591e-04,
        6.4017e-07, 3.4685e-07, 3.9362e-17, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00])},
 {'Na': {'Ce': 2},
  'Nb': 22,
  'Nb_true': 15,
  'weight': tensor([1.8182e-02, 1.8182e-02, 1.8182e-02, 1.8182e-02, 1.8182e-02, 1.8182e-02,
        1.8182e-02, 1.8182e-02, 1.8182e-02, 1.7078e-02, 1.7078e-02, 2.1952e-03,
        8.3129e-06, 4.7098e-06, 1.7681e-08, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00])},
 {'Na': {'Ce': 2},
  'Nb': 22,
  'Nb_true': 19,
  'weight': tensor([1.8182e-02, 1.8182e-02, 1.8182e-02, 1.8182e-02, 1.8182e-02, 1.8182e-02,
        1.8182e-02, 1.8182e-02, 1.8182e-02, 1.5403e-02, 1.5403e-02, 3.2374e-03,
        1.2543e-03, 6.3727e-04, 4.2917e-04, 5.2057e-09, 3.3145e-09, 1.0093e-18,
        1.0093e-18, 0.0000e+00, 0.0000e+00, 0.0000e+00])},
 {'Na': {'Ce': 2},
  'Nb': 22,
  'Nb_true': 20,
  'weight': tensor([1.8182e-02, 1.8182e-02, 1.8182e-02, 1.8182e-02, 1.8182e-02, 1.8182e-02,
        1.8182e-02, 1.8182e-02, 1.8182e-02, 1.0818e-02, 1.0818e-02, 7.7592e-03,
        4.3669e-03, 1.4670e-03, 1.1039e-03, 1.7096e-05, 1.2773e-05, 3.3934e-12,
        3.3934e-12, 2.7251e-17, 0.0000e+00, 0.0000e+00])}]
        """
        info_stru = [
            {'Na': {'Ce': 2},
            'Nb': 22,
            'Nb_true': 14,
            'weight': [1.8182e-02, 1.8182e-02, 1.8182e-02, 1.8182e-02, 1.8182e-02, 1.8182e-02,
        1.8182e-02, 1.8182e-02, 1.8182e-02, 1.8154e-02, 1.8154e-02, 3.3398e-05,
        2.0629e-05, 2.1154e-06, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00]
            },
            {'Na': {'Ce': 2},
            'Nb': 22,
            'Nb_true': 15,
            'weight': [1.8182e-02, 1.8182e-02, 1.8182e-02, 1.8182e-02, 1.8182e-02, 1.8182e-02,
        1.8182e-02, 1.8182e-02, 1.8174e-02, 1.8092e-02, 1.8092e-02, 1.8591e-04,
        6.4017e-07, 3.4685e-07, 3.9362e-17, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00]
            },
            {'Na': {'Ce': 2},
            'Nb': 22,
            'Nb_true': 15,
            'weight': [1.8182e-02, 1.8182e-02, 1.8182e-02, 1.8182e-02, 1.8182e-02, 1.8182e-02,
        1.8182e-02, 1.8182e-02, 1.8182e-02, 1.7078e-02, 1.7078e-02, 2.1952e-03,
        8.3129e-06, 4.7098e-06, 1.7681e-08, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00]
            },
            {'Na': {'Ce': 2},
            'Nb': 22,
            'Nb_true': 19,
            'weight': [1.8182e-02, 1.8182e-02, 1.8182e-02, 1.8182e-02, 1.8182e-02, 1.8182e-02,
        1.8182e-02, 1.8182e-02, 1.8182e-02, 1.5403e-02, 1.5403e-02, 3.2374e-03,
        1.2543e-03, 6.3727e-04, 4.2917e-04, 5.2057e-09, 3.3145e-09, 1.0093e-18,
        1.0093e-18, 0.0000e+00, 0.0000e+00, 0.0000e+00]
            },
            {'Na': {'Ce': 2},
            'Nb': 22,
            'Nb_true': 20,
            'weight': [1.8182e-02, 1.8182e-02, 1.8182e-02, 1.8182e-02, 1.8182e-02, 1.8182e-02,
        1.8182e-02, 1.8182e-02, 1.8182e-02, 1.0818e-02, 1.0818e-02, 7.7592e-03,
        4.3669e-03, 1.4670e-03, 1.1039e-03, 1.7096e-05, 1.2773e-05, 3.3934e-12,
        3.3934e-12, 2.7251e-17, 0.0000e+00, 0.0000e+00]
            }
        ]

        result = stru(info_stru)
        print(result)

    def test_element(self):
        """info_element:
{'Ce': {'Ecut': 100,
        'Ne': 25,
        'Nl': 4,
        'Nu': [4, 2, 2, 1],
        'Rcut': 8,
        'dr': 0.01,
        'index': 0}}"""

        info_element = {
            "Ce": {
                "Ecut": 100,
                "Ne": 25,
                "Nl": 4,
                "Nu": [4, 2, 2, 1],
                "Rcut": 8,
                "dr": 0.01,
                "index": 0
            }
        }

        result = element(info_element)
        print(result)

    def test_opt(self):
        """info_opt:
{'cal_T': False,
 'cal_smooth': True,
 'lr': 0.03,
 'max_steps': 9000}"""
            
        info_opt = {
            "cal_T": False,
            "cal_smooth": True,
            "lr": 0.03,
            "max_steps": 9000
        }

        result = opt(info_opt)
        print(result)

    def test_the_max(self):
        """info_max:
[{'Na': 2, 'Nb': 22, 'Ne': 25, 'Nl': 4, 'Nm': 7, 'Nt': 1, 'Nu': 4},
 {'Na': 2, 'Nb': 22, 'Ne': 25, 'Nl': 4, 'Nm': 7, 'Nt': 1, 'Nu': 4},
 {'Na': 2, 'Nb': 22, 'Ne': 25, 'Nl': 4, 'Nm': 7, 'Nt': 1, 'Nu': 4},
 {'Na': 2, 'Nb': 22, 'Ne': 25, 'Nl': 4, 'Nm': 7, 'Nt': 1, 'Nu': 4},
 {'Na': 2, 'Nb': 22, 'Ne': 25, 'Nl': 4, 'Nm': 7, 'Nt': 1, 'Nu': 4}]"""
        
        info_max = [
            {'Na': 2, 'Nb': 22, 'Ne': 25, 'Nl': 4, 'Nm': 7, 'Nt': 1, 'Nu': 4},
            {'Na': 2, 'Nb': 22, 'Ne': 25, 'Nl': 4, 'Nm': 7, 'Nt': 1, 'Nu': 4},
            {'Na': 2, 'Nb': 22, 'Ne': 25, 'Nl': 4, 'Nm': 7, 'Nt': 1, 'Nu': 4},
            {'Na': 2, 'Nb': 22, 'Ne': 25, 'Nl': 4, 'Nm': 7, 'Nt': 1, 'Nu': 4},
            {'Na': 2, 'Nb': 22, 'Ne': 25, 'Nl': 4, 'Nm': 7, 'Nt': 1, 'Nu': 4}
        ]

        result = the_max(info_max)
        print(result)

if __name__ == "__main__":
    unittest.main()
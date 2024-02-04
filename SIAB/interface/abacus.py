"""this file contains functions for generating ABACUS input files"""

class abacus_driver:
    """not actually used presently"""
    def __init__(self, general: dict, calculation_setting: dict, shape: str):
        self.element = general["element"]
        self.fpseudo = general["Pseudo_name"]
        self.calculation_setting = calculation_setting
        self.shape = shape

    def stru(self, bond_length: float = 0.0, nspin: int = 1):
        """generate structure"""
        if self.shape == "":
            raise ValueError("shape is not specified")
        if bond_length == 0.0:
            raise ValueError("bond_length is not specified")
        if self.shape == "dimer":
            return dimer(self.element, 1.0, self.fpseudo, 20.0, bond_length, nspin), 2
        elif self.shape == "trimer":
            return trimer(self.element, 1.0, self.fpseudo, 20.0, bond_length, nspin), 3
        elif self.shape == "tetramer":
            return tetramer(self.element, 1.0, self.fpseudo, 20.0, bond_length, nspin), 4
        else:
            raise NotImplementedError("Unknown shape %s"%self.shape)

    def kpt(self):
        """For ABACUS-orbitals numerical orbitals generation workflow specifically"""
        return "K_POINTS\n0\nGamma\n1 1 1 0 0 0\n"
    
    def input(self, suffix: str = "") -> str:
        return INPUT(self.calculation_setting, suffix=suffix)
    
    def generation(self, stru_setting: dict):
        return generation(self.calculation_setting, stru_setting)

def dimer(element, mass, fpseudo, lattice_constant, bond_length, nspin):
    """generate dimer structure"""
    starting_magnetization = 0.0 if nspin == 1 else 2.0
    result = "ATOMIC_SPECIES\n%s %.6f %s\n"%(element, mass, fpseudo)
    result += "LATTICE_CONSTANT\n%.6f  // add lattice constant(a.u.)\n"%lattice_constant
    result += "LATTICE_VECTORS\n"
    result += "%10.8f %10.8f %10.8f\n"%(1.0, 0.0, 0.0)
    result += "%10.8f %10.8f %10.8f\n"%(0.0, 1.0, 0.0)
    result += "%10.8f %10.8f %10.8f\n"%(0.0, 0.0, 1.0)
    result += "ATOMIC_POSITIONS\nCartesian_angstrom  //Cartesian or Direct coordinate.\n"
    result += "%s      //Element Label\n"%element
    result += "%.2f     //starting magnetism\n"%starting_magnetization
    result += "2       //number of atoms\n"
    result += "%10.8f %10.8f %10.8f 0 0 0\n"%(0.0, 0.0, 0.0)
    result += "%10.8f %10.8f %10.8f 0 0 0\n"%(0.0, 0.0, bond_length)
    return result

def trimer(element, mass, fpseudo, lattice_constant, bond_length, nspin):
    """generate trimer structure"""
    starting_magnetization = 0.0 if nspin == 1 else 2.0
    dis1 = bond_length * 0.86603
    dis2 = bond_length * 0.5
    result = "ATOMIC_SPECIES\n%s %.6f %s\n"%(element, mass, fpseudo)
    result += "LATTICE_CONSTANT\n%.6f  // add lattice constant(a.u.)\n"%lattice_constant
    result += "LATTICE_VECTORS\n"
    result += "%10.8f %10.8f %10.8f\n"%(1.0, 0.0, 0.0)
    result += "%10.8f %10.8f %10.8f\n"%(0.0, 1.0, 0.0)
    result += "%10.8f %10.8f %10.8f\n"%(0.0, 0.0, 1.0)
    result += "ATOMIC_POSITIONS\nCartesian_angstrom  //Cartesian or Direct coordinate.\n"
    result += "%s      //Element Label\n"%element
    result += "%.2f     //starting magnetism\n"%starting_magnetization
    result += "3       //number of atoms\n"
    result += "%10.8f %10.8f %10.8f 0 0 0\n"%(0.0, 0.0, 0.0)
    result += "%10.8f %10.8f %10.8f 0 0 0\n"%(0.0, 0.0, bond_length)
    result += "%10.8f %10.8f %10.8f 0 0 0\n"%(0.0, dis1, dis2)
    return result

def tetramer(element, mass, fpseudo, lattice_constant, bond_length, nspin):
    """generate tetramer structure"""
    starting_magnetization = 0.0 if nspin == 1 else 2.0
    dis1 = bond_length * 0.86603
    dis2 = bond_length * 0.5
    dis3 = bond_length * 0.81649
    dis4 = bond_length * 0.28867
    result = "ATOMIC_SPECIES\n%s %.6f %s\n"%(element, mass, fpseudo)
    result += "LATTICE_CONSTANT\n%.6f  // add lattice constant(a.u.)\n"%lattice_constant
    result += "LATTICE_VECTORS\n"
    result += "%10.8f %10.8f %10.8f\n"%(1.0, 0.0, 0.0)
    result += "%10.8f %10.8f %10.8f\n"%(0.0, 1.0, 0.0)
    result += "%10.8f %10.8f %10.8f\n"%(0.0, 0.0, 1.0)
    result += "ATOMIC_POSITIONS\nCartesian_angstrom  //Cartesian or Direct coordinate.\n"
    result += "%s      //Element Label\n"%element
    result += "%.2f     //starting magnetism\n"%starting_magnetization
    result += "4       //number of atoms\n"
    result += "%10.8f %10.8f %10.8f 0 0 0\n"%(0.0, 0.0, 0.0)
    result += "%10.8f %10.8f %10.8f 0 0 0\n"%(0.0, 0.0, bond_length)
    result += "%10.8f %10.8f %10.8f 0 0 0\n"%(0.0, dis1, dis2)
    result += "%10.8f %10.8f %10.8f 0 0 0\n"%(dis3, dis4, dis2)
    return result

def STRU(shape: str = "", element: str = "", mass: float = 1.0, fpseudo: str = "", 
         lattice_constant: float = 1.0, bond_length: float = 3.0, nspin: int = 1):
    """generate structure"""
    if shape == "":
        raise ValueError("shape is not specified")
    if element == "":
        raise ValueError("element is not specified")
    if fpseudo == "":
        raise ValueError("fpseudo is not specified")
    if lattice_constant == 0.0:
        raise ValueError("lattice_constant is not specified")
    if bond_length == 0.0:
        raise ValueError("bond_length is not specified")
    if shape == "dimer":
        return dimer(element, mass, fpseudo, lattice_constant, bond_length, nspin), 2
    elif shape == "trimer":
        return trimer(element, mass, fpseudo, lattice_constant, bond_length, nspin), 3
    elif shape == "tetramer":
        return tetramer(element, mass, fpseudo, lattice_constant, bond_length, nspin), 4
    else:
        raise NotImplementedError("Unknown shape %s"%shape)

def KPOINTS():
    """For ABACUS-orbitals numerical orbitals generation workflow specifically"""
    return "K_POINTS\n0\nGamma\n1 1 1 0 0 0\n"

def INPUT(calculation_setting: dict,
          suffix: str = "") -> str:
    """generate INPUT file for orbital generation task"""
    inbuilt_template = {
        "suffix": "ABACUS", "stru_file": "STRU", "kpoint_file": "KPT", # wannier_card is deprecated
        "wannier_card": "INPUTw",
        "pseudo_dir": "./",
        "calculation": "scf", # calculation, definitely to be scf for orbital generation
        "basis_type": "pw", "ecutwfc": "100",
        "ks_solver": "dav", "nbands": "auto", "scf_thr": "1.0e-7", "scf_nmax": "9000", # scf control
        "ntype": "1", "nspin": "1", # system info
        "lmaxmax": "4", "bessel_nao_rcut": "10", # orbital generation control
        "smearing_method": "gauss", "smearing_sigma": "0.015", # for improving convergence
        "mixing_type": "broyden", "mixing_beta": "0.8", "mixing_ndim": "8", "mixing_gg0": "1", # mixing control
        "printe": "1" # print energy
    }
    result = "INPUT_PARAMETERS"
    for key in calculation_setting.keys():
        if key in inbuilt_template.keys():
            value = calculation_setting[key]
            value = " ".join([str(v) for v in value]) if isinstance(value, list) else value
            inbuilt_template[key] = value
        else:
            print("Warning: unknown key %s"%key)
    if suffix != "":
        inbuilt_template["suffix"] = suffix
        inbuilt_template["stru_file"] += "-"+suffix
        inbuilt_template["kpoint_file"] += "-"+suffix
    for key, value in inbuilt_template.items():
        result += "\n%-20s %s"%(key, value)

    return result

"""because INPUTw and INPUTs have been deprecated, these two functions are not included in
refactor plan"""

def generation(input_setting: dict,
               stru_setting: dict):
    """generate input files for orbital generation
    
    input_settings: dict, INPUT settings for ABACUS
    stru_settings: dict, structure settings for ABACUS

    Return:
        folder: str, a string used to distinguish different orbital generation tasks
    Details:
        in `stru_settings`, at least contain `shape`, `element`, `fpseudo` and `bond_length`
        information.
    """
    necessary_keys = ["element", "shape", "fpseudo", "bond_length"]
    for necessary_key in necessary_keys:
        if necessary_key not in stru_setting.keys():
            raise ValueError("key %s is not specified"%necessary_key)
    folder = "-".join([str(stru_setting[key]) for key in necessary_keys if key != "fpseudo"])
    _input = INPUT(input_setting, suffix=folder)
    _stru, natom = STRU(**stru_setting)
    _kpt = KPOINTS()

    """to make code expresses clear"""
    suffix = folder
    with open("INPUT-"+suffix, "w") as f:
        f.write(_input)
    with open("STRU-"+suffix, "w") as f:
        f.write(_stru)
    with open("KPT-"+suffix, "w") as f:
        f.write(_kpt)
    with open("INPUTw", "w") as f:
        f.write("WANNIER_PARAMETERS\n")
        f.write("out_spillage 2\n")
    return folder

import re
def read_INPUT(folder: str = "") -> dict:
    """parse ABACUS INPUT file, return a dict"""
    pattern = r"^(\s*)([\w_]+)(\s+)([^\#]+)(.*)$"
    with open(folder+"/INPUT", "r") as f:
        lines = f.readlines()
    result = {}
    for line in lines:
        if line == "INPUT_PARAMETERS":
            continue
        else:
            match = re.match(pattern, line.strip())
            if match is not None:
                result[match.group(2)] = match.group(4)
    return result

def version_compare(version_1: str, version_2: str) -> bool:
    """compare two version strings, return True if version_1 <= version_2"""
    version_1 = version_1.split(".")
    version_2 = version_2.split(".")
    for i in range(len(version_1)):
        if int(version_1[i]) < int(version_2[i]):
            return True
        elif int(version_1[i]) > int(version_2[i]):
            return False
        else:
            continue
    return True
##############################################
#           general information              #
##############################################
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

##############################################
#         input files preparation            #
##############################################
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
    """generate INPUT file for orbital generation task. This function is designed with following
    logic:
    1. user will not use keywords more than this function's consideration
    2. user may define different values for some keywords, if that happens, overwrite the default
    3. write to INPUT from the inbuilt_template
    """
    inbuilt_template = {
        "suffix": "ABACUS", "stru_file": "STRU", "kpoint_file": "KPT", 
        "wannier_card": "INPUTw", # wannier_card is deprecated
        "pseudo_dir": "./",
        "calculation": "scf",     # calculation, definitely to be scf for orbital generation
        "basis_type": "pw", "ecutwfc": "100",
        "ks_solver": "dav", "nbands": "auto", "scf_thr": "1.0e-7", "scf_nmax": "9000", # scf control
        "ntype": "1", "nspin": "1", # system info
        "lmaxmax": "4", "bessel_nao_rcut": "10", # orbital generation control
        "smearing_method": "gauss", "smearing_sigma": "0.015", # for improving convergence
        "mixing_type": "broyden", "mixing_beta": "0.8", "mixing_ndim": "8", "mixing_gg0": "0", # mixing control
        "gamma_only": "1", # force gamma point only calculation
        "printe": "1" # print energy
    }
    if "nspin" in calculation_setting.keys():
        if calculation_setting["nspin"] == 2:
            inbuilt_template["nspin"] = 2
            inbuilt_template["mixing_beta"] = 0.4
            inbuilt_template["mixing_beta_mag"] = 0.4
            
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

##############################################
#              file operations               #
##############################################
def configure(input_setting: dict,
              stru_setting: dict) -> str:
    """generate input files for orbital generation in present folder
    
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

import SIAB.interface.env as sienv
def archive(footer: str = "", env: str = "local"):

    """mkdir and move correspnding input files to folder"""
    headers = ["INPUT", "STRU", "KPT"]
    if footer != "":
        sienv.op("mkdir", footer, additional_args=["-p"], env=env)
        for header in headers:
            if header == "INPUT":
                sienv.op("mv", "%s-%s"%(header, footer), "%s/INPUT"%(footer), env=env)
            else:
                sienv.op("mv", "%s-%s"%(header, footer), "%s/"%(footer), env=env)
        sienv.op("mv", "INPUTw", "%s/INPUTw"%(footer), env=env)
    else:
        raise ValueError("footer is not specified")

##############################################
#     job submission and restart control     #
##############################################
# entry point for running ABACUS calculation
def run_all(general: dict,
            reference_shapes: list,
            bond_lengths: list,
            calculation_settings: list,
            env_settings: tuple,
            test: bool = False):
    """iterately calculate planewave wavefunctions for reference shapes and bond lengths"""
    folders = []
    for i in range(len(reference_shapes)):
        folders_istructure = []
        """abacus_driver can be created iteratively in this layer, and feed in following functions"""
        if "auto" in bond_lengths[i]:
            """search bond lengths"""
            folders_istructure = blscan(general=general,
                                        calculation_setting=calculation_settings[i],
                                        env_settings=env_settings,
                                        reference_shape=reference_shapes[i],
                                        nstep_bidirection=5,
                                        stepsize=[0.2, 0.5],
                                        ener_thr=1.5,
                                        test=test)
        else:
            folders_istructure = normal(general=general,
                                        reference_shape=reference_shapes[i],
                                        bond_lengths=bond_lengths[i],
                                        calculation_setting=calculation_settings[i],
                                        env_settings=env_settings,
                                        test=test)
        folders.append(folders_istructure)
    return folders

import os
def is_duplicate(folder: str, abacus_setting: dict):
    """check if the abacus calculation can be safely (really?)
    skipped"""
    # STAGE1: existence of folder
    if not os.path.isdir(folder):
        return False
    files = os.listdir(folder)
    print("DUPLICATE CHECK-1 pass: folder %s exists"%folder)
    # STAGE2: existence of INPUT files   
    for fcplsry in ["INPUT", "INPUTw"]:
        if fcplsry not in files:
            return False
    print("DUPLICATE CHECK-2 pass: INPUT and INPUTw exist")
    # STAGE3: correspondence of INPUT settings
    for key in ["bessel_nao_rcut", "lmaxmax"]:
        if key not in abacus_setting.keys():
            raise ValueError("NECESSARY KEYWORD %s is not specified"%key)
    original = read_INPUT(folder)
    for key in abacus_setting.keys():
        value = abacus_setting[key]
        if isinstance(value, list):
            value = " ".join([str(v) for v in value])
        else:
            value = str(value)
        if original[key] != value:
            print("KEYWORD \"%s\" has different values. Original: %s, new: %s\nDifference detected, start a new job."%(key, original[key], value))
            return False
    rcuts = abacus_setting["bessel_nao_rcut"]
    print("DUPLICATE CHECK-3 pass: INPUT settings are consistent")
    # STAGE4: existence of crucial output files
    if len(rcuts) == 1:
        if "orb_matrix.0.dat" not in files:
            return False
        if "orb_matrix.1.dat" not in files:
            return False
    else:
        for rcut in rcuts:
            if "orb_matrix_rcut%sderiv0.dat"%rcut not in files:
                return False
            if "orb_matrix_rcut%sderiv1.dat"%rcut not in files:
                return False
    print("DUPLICATE CHECK-4 pass: crucial output files exist")
    return True

# in the following, defines different tasks
# -------------------------------------------#
# TASK KIND1 - blscan                        #
# DESCRIPTION: search bond lengths           #
# -------------------------------------------#
import numpy as np
import SIAB.io.read_output as read_output
import SIAB.data.interface as db
from scipy.optimize import curve_fit
def blscan(general: dict,                  # general settings
           calculation_setting: dict,      # calculation setting, for setting up INPUT file
           env_settings: dict,             # calculation environment settings
           reference_shape: str,           # reference shape, always to be dimer
           nstep_bidirection: int = 5,     # number of steps for searching bond lengths per direction
           stepsize: list = [0.2, 0.5],    # stepsize for searching bond lengths, unit in angstrom
           ener_thr: float = 1.5,          # energy threshold for searching bond lengths
           test: bool = True):
    # functions that only allowed to use in this function are defined here with slightly different
    # way to name the function
    # 1. guessbls: generate initial guess for bond lengths
    # 2. fitmorse: fitting morse potential, return D_e, a, r_e, e_0 in the equation below:
    # V(r) = D_e * (1-exp(-a(r-r_e)))^2 + e_0
    # 3. returnbls: get the range of bond lengths corresponding to energies lower than ener_thr

    bl0 = db.get_radius(general["element"]) * 2.0
    if bl0 >= 2.7: # this is quite an empirical threshold
        print("WARNING: default covalent radius is %4.2f Angstrom, which is larger than 2.7 Angstrom."%bl0)
        while bl0 > 2.7:
            bl0 /= 1.1
            print("SHRINK-> new bond length is %4.2f Angstrom, shrink with factor 1.1"%bl0)

    bls = blscan_guessbls(bl0=bl0,
                          stepsize=stepsize,
                          nstep_bidirection=nstep_bidirection)
    """generate folders"""
    folders = normal(general=general,
                     reference_shape=reference_shape,
                     bond_lengths=bls,
                     calculation_setting=calculation_setting,
                     env_settings=env_settings,
                     test=test)
    """wait for all jobs to finish"""
    """read energies"""
    bond_lengths = [float(folder.split("-")[-1]) for folder in folders]
    energies = [read_output.read_energy(folder=folder,
                                        suffix=folder) for folder in folders]
    """fitting morse potential"""
    De, a, re, e0 = blscan_fitmorse(bond_lengths, energies)

    """search bond lengths"""
    bond_lengths = blscan_returnbls(bl0=re,
                                    ener0=e0,
                                    bond_lengths=bond_lengths,
                                    energies=energies,
                                    ener_thr=ener_thr)

    folders_to_use = [folder for folder in folders for bond_length in bond_lengths if "%3.2f"%bond_length in folder]
    return folders_to_use

def blscan_guessbls(bl0: float, 
                    stepsize: list, 
                    nstep_bidirection: int = 5):
    """generate initial guess for bond lengths"""
    blmin = bl0 - stepsize[0]*nstep_bidirection
    blmax = bl0 + stepsize[1]*nstep_bidirection
    print("Searching bond lengths from %4.2f to %4.2f Angstrom, with stepsize %s."%(blmin, blmax, stepsize))
    left = np.linspace(blmin, bl0, nstep_bidirection+1).tolist()
    right = np.linspace(bl0, blmax, nstep_bidirection+1, endpoint=True).tolist()
    bond_lengths = left + right[1:]
    return [round(bl, 2) for bl in bond_lengths]

def blscan_fitmorse(bond_lengths: list, 
                energies: list):
    """fitting morse potential, return D_e, a, r_e, e_0 in the equation below:

    V(r) = D_e * (1-exp(-a(r-r_e)))^2 + e_0

    Use scipy.optimize.curve_fit to fit the parameters
    
    Return:
        D_e: float, eV
        a: float
        r_e: float, Angstrom
        e_0: float, eV
    """
    def morse_potential(r, De, a, re, e0=0.0):
        return De * (1.0 - np.exp(-a*(r-re)))**2.0 + e0
    
    popt, pcov = curve_fit(f=morse_potential, 
                            xdata=bond_lengths, 
                            ydata=energies,
                            p0=[-100, 1.0, 2.7, -100])
    if pcov is None:
        raise ValueError("fitting failed.")
    elif np.any(np.diag(pcov) < 0):
        raise ValueError("fitting failed.")
    elif np.any(np.diag(pcov) > 1e5):
        print("WARNING: fitting parameters are not accurate.")

    print("Morse potential fitting results:")
    print("%6s: %15.10f %10s (Bond dissociation energy)"%("D_e", popt[0], "eV"))
    print("%6s: %15.10f %10s (Morse potential parameter)"%("a", popt[1], ""))
    print("%6s: %15.10f %10s (Equilibrium bond length)"%("r_e", popt[2], "Angstrom"))
    print("%6s: %15.10f %10s (Zero point energy)"%("e_0", popt[3], "eV"))
    
    if popt[2] <= 0:
        print("bond lengths: ", " ".join([str(bl) for bl in bond_lengths]))
        print("energies: ", " ".join([str(e) for e in energies]))
    assert popt[2] > 0 # equilibrium bond length should be positive

    return popt[0], popt[1], popt[2], popt[3]

def blscan_returnbls(bl0: float, 
                     ener0: float, 
                     bond_lengths: list, 
                     energies: list, 
                     ener_thr: float = 1.5):
    """Get the range of bond lengths corresponding to energies lower than ener_thr"""

    delta_energies = [e-ener0 for e in energies]
    emin = min(delta_energies)
    i_emin = delta_energies.index(emin)
    delta_e_r = delta_energies[i_emin+1] - delta_energies[i_emin]
    delta_e_l = delta_energies[i_emin-1] - delta_energies[i_emin]
    
    assert i_emin > 2
    assert i_emin < len(delta_energies) - 2
    assert delta_e_r > 0
    assert delta_e_l > 0
    assert all(delta_energies) > 0

    i_emax_r, i_emax_l = 0, -1
    for i in range(i_emin, len(delta_energies)):
        if delta_energies[i] > ener_thr:
            i_emax_r = i
            break
    for i in range(i_emin, 0, -1):
        if delta_energies[i] > ener_thr:
            i_emax_l = i
            break

    if i_emax_r == 0:
        print("""WANRING: No bond length found with energy higher than the best energy threshold %4.2f eV."
         The highest energy during the search at right side (bond length increase direction) is %4.2f eV."""%(ener_thr, delta_energies[-1]))
        print("""If not satisfied, please consider:
1. check the dissociation energy of present element, 
2. enlarge the search range, 
3. lower energy threshold.""")
        print("Set the bond length to the highest energy point...")
        i_emax_r = len(delta_energies) - 1

    if i_emax_l == -1:
        print("\nSummary of bond lengths and energies:".upper())
        print("| Bond length (Angstrom) |   Energy (eV)   |")
        print("|------------------------|-----------------|")
        for bl, e in zip(bond_lengths, energies):
            line = "|%24.2f|%17.10f|"%(bl, e)
            print(line)

        raise ValueError("""WARNING: No bond length found with energy higher than %4.2f eV in bond length
search in left direction (bond length decrease direction), this is absolutely unacceptable compared with 
the right direction which may because of low dissociation energy. Exit."""%ener_thr)

    indices = [i_emax_l, (i_emax_l+i_emin)//2, i_emin, (i_emax_r+i_emin)//2, i_emax_r]
    print("\nSummary of bond lengths and energies:".upper())
    print("| Bond length (Angstrom) |   Energy (eV)   |")
    print("|------------------------|-----------------|")
    for bl, e in zip(bond_lengths, energies):
        line = "|%24.2f|%17.10f|"%(bl, e)
        if bond_lengths.index(bl) in indices:
            line += " <=="
        print(line)
    return [bond_lengths[i] for i in indices]

# -------------------------------------------#
# TASK KIND2 - normal                        #
# DESCRIPTION: run ABACUS calculation on     #
#              reference structures simply   #
# -------------------------------------------#
def normal(general: dict,
           reference_shape: str,
           bond_lengths: list,
           calculation_setting: dict,
           env_settings: dict,
           test: bool = True):
    """iteratively run ABACUS calculation on reference structures
    To let optimizer be easy to find output, return names of folders"""

    folders = []
    for bond_length in bond_lengths:
        stru_setting = {
            "shape": reference_shape,
            "bond_length": bond_length,
            "element": general["element"],
            "fpseudo": general["pseudo_name"],
            "lattice_constant": 20.0
        }
        folder = configure(input_setting=calculation_setting,
                           stru_setting=stru_setting)
        folders.append(folder)

        if is_duplicate(folder, calculation_setting):
            print("ABACUS calculation on reference structure %s with bond length %s is skipped."%(reference_shape, bond_length))
            sienv.op("rm", "INPUT-%s KPT-%s STRU-%s INPUTw"%(folder, folder, folder), env="local")
            continue
        # else...
        archive(footer=folder)
        print("""Run ABACUS calculation on reference structure.
Reference structure: %s
Bond length: %s"""%(reference_shape, bond_length))
        # need a better design here
        _jtg = sienv.submit(folder=folder,
                            module_load_command=env_settings["environment"],
                            mpi_command=env_settings["mpi_command"],
                            program_command=env_settings["abacus_command"],
                            test=test)
        
    """wait for all jobs to finish"""
    return folders

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

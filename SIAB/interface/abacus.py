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
def monomer(element, mass, fpseudo, lattice_constant, nspin):
    """generate monomer structure"""
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
    result += "1       //number of atoms\n"
    result += "%10.8f %10.8f %10.8f 0 0 0\n"%(0.0, 0.0, 0.0)
    return result

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

def STRU(shape: str, element: str, mass: float, fpseudo: str, 
         lattice_constant: float, bond_length: float, nspin: int):
    """generate structure"""
    if shape == "monomer":
        return monomer(element, mass, fpseudo, lattice_constant, nspin), 1
    elif shape == "dimer":
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
            
    all_params = abacus_params()
    result = "INPUT_PARAMETERS"
    for key in calculation_setting.keys():
        if key in all_params:
            value = calculation_setting[key]
            value = " ".join([str(v) for v in value]) if isinstance(value, list) else value
            inbuilt_template[key] = value
        else:
            print("WARNING: keyword %s might be unknown."%key, flush=True)

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
    # mostly value will not be None, except the case the monomer is included to be referred
    # in initial guess of coefficients of sphbes
    keys_in_foldername = ["element", "shape"]
    keys_in_foldername.append("bond_length") if stru_setting["shape"] != "monomer" else None
    # because bond_length is not necessary for monomer
    folder = f"{stru_setting['element']}-{stru_setting['shape']}"
    folder += "-%3.2f"%stru_setting["bond_length"] if stru_setting["shape"] != "monomer" else ""
    _input = INPUT(input_setting, suffix=folder)
    _stru, _ = STRU(**stru_setting)
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
            structures: dict,
            calculation_settings: list,
            env_settings: tuple,
            test: bool = False):
    """iterately calculate planewave wavefunctions for reference shapes and bond lengths"""
    folders = []
    for isp, shape in enumerate(structures.keys()):
        folders_istructure = []
        """abacus_driver can be created iteratively in this layer, and feed in following functions"""
        if structures[shape] == "auto" and shape != "monomer":
            """search bond lengths"""
            folders_istructure = blscan(general=general,
                                        calculation_setting=calculation_settings[isp],
                                        env_settings=env_settings,
                                        reference_shape=shape,
                                        nstep_bidirection=5,
                                        stepsize=[0.2, 0.5],
                                        ener_thr=1.5,
                                        test=test)
        else:
            bond_lengths = structures[shape] if shape != "monomer" else [0.0]
            folders_istructure = normal(general=general,
                                        reference_shape=shape,
                                        bond_lengths=bond_lengths,
                                        calculation_setting=calculation_settings[isp],
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
    print("DUPLICATE CHECK-1 pass: folder %s exists"%folder, flush=True)
    # STAGE2: existence of INPUT files   
    for fcplsry in ["INPUT", "INPUTw"]:
        if fcplsry not in files:
            return False
    print("DUPLICATE CHECK-2 pass: INPUT and INPUTw exist", flush=True)
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
            print("KEYWORD \"%s\" has different values. Original: %s, new: %s\nDifference detected, start a new job."%(key, original[key], value), flush=True)
            return False
    rcuts = abacus_setting["bessel_nao_rcut"]
    print("DUPLICATE CHECK-3 pass: INPUT settings are consistent", flush=True)
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
    print("DUPLICATE CHECK-4 pass: crucial output files exist", flush=True)
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
        print("WARNING: default covalent radius is %4.2f Angstrom, which is larger than 2.7 Angstrom."%bl0, flush=True)
        while bl0 > 2.7:
            bl0 /= 1.1
            print("SHRINK-> new bond length is %4.2f Angstrom, shrink with factor 1.1"%bl0, flush=True)

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
    print("Searching bond lengths from %4.2f to %4.2f Angstrom, with stepsize %s."%(blmin, blmax, stepsize), flush=True)
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
    
    # precondition the fitting problem, first assert the location of minimum energy point
    # always be sure there are at least two points on the both
    # left and right side of the minimum energy point
    idx_min = energies.index(min(energies))
    assert idx_min > 1, "There are fewer than 2 points on the left side of the minimum energy point."
    assert idx_min < len(energies) - 2, "There are fewer than 2 points on the right side of the minimum energy point."
    assert len(energies) > 5, "There are fewer than 5 points in total."
    # set threshold to be 10, this will force the point with the energy no higher than 10 eV
    cndt_thr = 10 # eV
    ediff = max(energies) - min(energies)
    conditioned = ediff < cndt_thr # if true, the fitting problem is relatively balanced
    while not conditioned:
        # remove the highest energy point
        idx_remove = energies.index(max(energies))
        if idx_remove >= idx_min:
            break # it means all points are evenly distributed around the minimum energy point
        print("MORSE POTENTIAL FITTING: remove the highest energy point %4.2f eV at bond length %4.2f Angstrom."%(energies[idx_remove], bond_lengths[idx_remove]), flush=True)
        energies.pop(idx_remove)
        bond_lengths.pop(idx_remove)
        # refresh the condition
        ediff = max(energies) - min(energies)
        conditioned = ediff < cndt_thr or len(energies) == 5

    popt, pcov = curve_fit(f=morse_potential, 
                           xdata=bond_lengths, 
                           ydata=energies,
                           p0=[energies[-1] - min(energies), 1.0, 2.7, min(energies)])
    if pcov is None:
        raise ValueError("fitting failed.")
    elif np.any(np.diag(pcov) < 0):
        raise ValueError("fitting failed.")
    elif np.any(np.diag(pcov) > 1e5):
        print("WARNING: fitting parameters are not accurate.", flush=True)

    # MUST SATISFY THE PHYSICAL MEANING
    assert popt[0] > 0 # D_e, dissociation energy MUST be positive
    assert popt[1] > 0 # a, Morse potential parameter MUST be positive
    assert popt[2] > 0 # r_e, equilibrium bond length MUST be positive
    assert popt[3] < 0 # e_0, zero point energy ALWAYS be negative

    print("Morse potential fitting results:", flush=True)
    print("%6s: %15.10f %10s (Bond dissociation energy)"%("D_e", popt[0], "eV"), flush=True)
    print("%6s: %15.10f %10s (Morse potential parameter)"%("a", popt[1], ""), flush=True)
    print("%6s: %15.10f %10s (Equilibrium bond length)"%("r_e", popt[2], "Angstrom"), flush=True)
    print("%6s: %15.10f %10s (Zero point energy)"%("e_0", popt[3], "eV"), flush=True)
    
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

    # always be sure there are at least two points on the both
    assert i_emin > 1, "There are fewer than 2 points on the left side of the minimum energy point."
    assert i_emin < len(delta_energies) - 2, "There are fewer than 2 points on the right side of the minimum energy point."
    assert delta_e_r > 0, "The energy difference between the minimum energy point and the right side is not positive."
    assert delta_e_l > 0, "The energy difference between the minimum energy point and the left side is not positive."
    assert all(delta_energies) > 0, "The energy difference is not positive."

    i_emax_r, i_emax_l = 0, -1 # initialize the right index to be the left-most, and vice versa
    for i in range(i_emin, len(delta_energies)):
        if delta_energies[i] >= ener_thr:
            i_emax_r = i
            break
    for i in range(i_emin, -1, -1):
        if delta_energies[i] >= ener_thr:
            i_emax_l = i
            break

    if i_emax_r == 0:
        print("""WANRING: No bond length found with energy higher than the best energy threshold %4.2f eV."
         The highest energy during the search at right side (bond length increase direction) is %4.2f eV."""%(ener_thr, delta_energies[-1]), flush=True)
        print("""If not satisfied, please consider:
1. check the dissociation energy of present element, 
2. enlarge the search range, 
3. lower energy threshold.""", flush=True)
        print("Set the bond length to the highest energy point...", flush=True)
        i_emax_r = len(delta_energies) - 1

    if i_emax_l == -1:
        print("\nSummary of bond lengths and energies:".upper(), flush=True)
        print("| Bond length (Angstrom) |   Energy (eV)   | Relative Energy (eV) |", flush=True)
        print("|------------------------|-----------------|----------------------|", flush=True)
        for bl, e, de in zip(bond_lengths, energies, delta_energies):
            line = "|%24.2f|%17.10f|%22.10f|"%(bl, e, de)
            print(line, flush=True)

        raise ValueError("""WARNING: No bond length found with energy higher than %4.2f eV in bond length
search in left direction (bond length decrease direction), this is absolutely unacceptable compared with 
the right direction which may because of low dissociation energy. Exit."""%ener_thr)

    indices = [i_emax_l, (i_emax_l+i_emin)//2, i_emin, (i_emax_r+i_emin)//2, i_emax_r]
    print("\nSummary of bond lengths and energies:".upper(), flush=True)
    print("| Bond length (Angstrom) |   Energy (eV)   | Relative Energy (eV) |", flush=True)
    print("|------------------------|-----------------|----------------------|", flush=True)
    for bl, e, de in zip(bond_lengths, energies, delta_energies):
        line = "|%24.2f|%17.10f|%22.10f|"%(bl, e, de)
        if bond_lengths.index(bl) in indices:
            line += " <=="
        print(line, flush=True)
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
        stru_setting = {"element": general["element"], "shape": reference_shape, "bond_length": bond_length,
            "fpseudo": general["pseudo_name"], "lattice_constant": 20.0, "nspin": calculation_setting["nspin"],
            "mass": 1.0}
        folder = configure(input_setting=calculation_setting,
                           stru_setting=stru_setting)
        folders.append(folder) if "monomer" not in folder else None
        # check if the calculation is duplicate, if so, skip
        if is_duplicate(folder, calculation_setting):
            print("ABACUS calculation on reference structure %s with bond length %s is skipped."%(reference_shape, bond_length), flush=True)
            sienv.op("rm", "INPUT-%s KPT-%s STRU-%s INPUTw"%(folder, folder, folder), env="local")
            continue
        # else...
        archive(footer=folder)
        print("""Run ABACUS calculation on reference structure.
Reference structure: %s
Bond length: %s"""%(reference_shape, bond_length), flush=True)
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
    if folder.startswith("INPUT_PARAMETERS"):
        lines = folder.split("\n")
    else:
        with open(folder+"/INPUT", "r") as f:
            lines = f.readlines()

    pattern = r"^(\s*)([\w_]+)(\s+)([^\#]+)(.*)$"
    result = {}
    for line in lines:
        if line == "INPUT_PARAMETERS":
            continue
        else:
            match = re.match(pattern, line.strip())
            if match is not None:
                result[match.group(2)] = match.group(4)
    return result

def abacus_params():
    return list(read_INPUT(ABACUS_INPUT_TEMPLATE).keys())

ABACUS_INPUT_TEMPLATE = """INPUT_PARAMETERS
#Parameters (1.General)
suffix                         ABACUS #the name of main output directory
latname                        none #the name of lattice name
stru_file                      STRU #the filename of file containing atom positions
kpoint_file                    KPT #the name of file containing k points
pseudo_dir                     ../../../tests/PP_ORB/ #the directory containing pseudo files
orbital_dir                     #the directory containing orbital files
pseudo_rcut                    15 #cut-off radius for radial integration
pseudo_mesh                    0 #0: use our own mesh to do radial renormalization; 1: use mesh as in QE
lmaxmax                        2 #maximum of l channels used
dft_functional                 default #exchange correlation functional
xc_temperature                 0 #temperature for finite temperature functionals
calculation                    scf #test; scf; relax; nscf; get_wf; get_pchg
esolver_type                   ksdft #the energy solver: ksdft, sdft, ofdft, tddft, lj, dp
ntype                          1 #atom species number
nspin                          1 #1: single spin; 2: up and down spin; 4: noncollinear spin
kspacing                       0 0 0  #unit in 1/bohr, should be > 0, default is 0 which means read KPT file
min_dist_coef                  0.2 #factor related to the allowed minimum distance between two atoms
nbands                         0 #number of bands
nbands_sto                     256 #number of stochastic bands
nbands_istate                  5 #number of bands around Fermi level for get_pchg calulation
symmetry                       1 #the control of symmetry
init_vel                       0 #read velocity from STRU or not
symmetry_prec                  1e-06 #accuracy for symmetry
symmetry_autoclose             1 #whether to close symmetry automatically when error occurs in symmetry analysis
nelec                          0 #input number of electrons
nelec_delta                    0 #change in the number of total electrons
out_mul                        0 # mulliken  charge or not
noncolin                       0 #using non-collinear-spin
lspinorb                       0 #consider the spin-orbit interaction
kpar                           1 #devide all processors into kpar groups and k points will be distributed among each group
bndpar                         1 #devide all processors into bndpar groups and bands will be distributed among each group
out_freq_elec                  0 #the frequency ( >= 0) of electronic iter to output charge density and wavefunction. 0: output only when converged
dft_plus_dmft                  0 #true:DFT+DMFT; false: standard DFT calcullation(default)
rpa                            0 #true:generate output files used in rpa calculation; false:(default)
printe                         100 #Print out energy for each band for every printe steps
mem_saver                      0 #Only for nscf calculations. if set to 1, then a memory saving technique will be used for many k point calculations.
diago_proc                     1 #the number of procs used to do diagonalization
nbspline                       -1 #the order of B-spline basis
wannier_card                   none #input card for wannier functions
soc_lambda                     1 #The fraction of averaged SOC pseudopotential is given by (1-soc_lambda)
cal_force                      0 #if calculate the force at the end of the electronic iteration
out_freq_ion                   0 #the frequency ( >= 0 ) of ionic step to output charge density and wavefunction. 0: output only when ion steps are finished
device                         cpu #the computing device for ABACUS
precision                      double #the computing precision for ABACUS

#Parameters (2.PW)
ecutwfc                        60 ##energy cutoff for wave functions
ecutrho                        240 ##energy cutoff for charge density and potential
erf_ecut                       0 ##the value of the constant energy cutoff
erf_height                     0 ##the height of the energy step for reciprocal vectors
erf_sigma                      0.1 ##the width of the energy step for reciprocal vectors
fft_mode                       0 ##mode of FFTW
pw_diag_nmax                   50 #max iteration number for cg
diago_cg_prec                  1 #diago_cg_prec
pw_diag_thr                    0.01 #threshold for eigenvalues is cg electron iterations
scf_thr                        1e-07 #charge density error
scf_thr_type                   1 #type of the criterion of scf_thr, 1: reci drho for pw, 2: real drho for lcao
init_wfc                       atomic #start wave functions are from 'atomic', 'atomic+random', 'random' or 'file'
init_chg                       atomic #start charge is from 'atomic' or file
chg_extrap                     atomic #atomic; first-order; second-order; dm:coefficients of SIA
out_chg                        0 #>0 output charge density for selected electron steps
out_pot                        0 #output realspace potential
out_wfc_pw                     0 #output wave functions
out_wfc_r                      0 #output wave functions in realspace
out_dos                        0 #output energy and dos
out_band                       0 #output energy and band structure (with precision 8)
out_proj_band                  0 #output projected band structure
restart_save                   0 #print to disk every step for restart
restart_load                   0 #restart from disk
read_file_dir                  auto #directory of files for reading
nx                             0 #number of points along x axis for FFT grid
ny                             0 #number of points along y axis for FFT grid
nz                             0 #number of points along z axis for FFT grid
ndx                            0 #number of points along x axis for FFT smooth grid
ndy                            0 #number of points along y axis for FFT smooth grid
ndz                            0 #number of points along z axis for FFT smooth grid
cell_factor                    1.2 #used in the construction of the pseudopotential tables
pw_seed                        1 #random seed for initializing wave functions

#Parameters (3.Stochastic DFT)
method_sto                     2 #1: slow and save memory, 2: fast and waste memory
npart_sto                      1 #Reduce memory when calculating Stochastic DOS
nbands_sto                     256 #number of stochstic orbitals
nche_sto                       100 #Chebyshev expansion orders
emin_sto                       0 #trial energy to guess the lower bound of eigen energies of the Hamitonian operator
emax_sto                       0 #trial energy to guess the upper bound of eigen energies of the Hamitonian operator
seed_sto                       0 #the random seed to generate stochastic orbitals
initsto_ecut                   0 #maximum ecut to init stochastic bands
initsto_freq                   0 #frequency to generate new stochastic orbitals when running md
cal_cond                       0 #calculate electronic conductivities
cond_che_thr                   1e-08 #control the error of Chebyshev expansions for conductivities
cond_dw                        0.1 #frequency interval for conductivities
cond_wcut                      10 #cutoff frequency (omega) for conductivities
cond_dt                        0.02 #t interval to integrate Onsager coefficiencies
cond_dtbatch                   0 #exp(iH*dt*cond_dtbatch) is expanded with Chebyshev expansion.
cond_smear                     1 #Smearing method for conductivities
cond_fwhm                      0.4 #FWHM for conductivities
cond_nonlocal                  1 #Nonlocal effects for conductivities

#Parameters (4.Relaxation)
ks_solver                      cg #cg; dav; lapack; genelpa; scalapack_gvx; cusolver
scf_nmax                       100 ##number of electron iterations
relax_nmax                     1 #number of ion iteration steps
out_stru                       0 #output the structure files after each ion step
force_thr                      0.001 #force threshold, unit: Ry/Bohr
force_thr_ev                   0.0257112 #force threshold, unit: eV/Angstrom
force_thr_ev2                  0 #force invalid threshold, unit: eV/Angstrom
relax_cg_thr                   0.5 #threshold for switching from cg to bfgs, unit: eV/Angstrom
stress_thr                     0.5 #stress threshold
press1                         0 #target pressure, unit: KBar
press2                         0 #target pressure, unit: KBar
press3                         0 #target pressure, unit: KBar
relax_bfgs_w1                  0.01 #wolfe condition 1 for bfgs
relax_bfgs_w2                  0.5 #wolfe condition 2 for bfgs
relax_bfgs_rmax                0.8 #maximal trust radius, unit: Bohr
relax_bfgs_rmin                1e-05 #minimal trust radius, unit: Bohr
relax_bfgs_init                0.5 #initial trust radius, unit: Bohr
cal_stress                     0 #calculate the stress or not
fixed_axes                     None #which axes are fixed
fixed_ibrav                    0 #whether to preseve lattice type during relaxation
fixed_atoms                    0 #whether to preseve direct coordinates of atoms during relaxation
relax_method                   cg #bfgs; sd; cg; cg_bfgs;
relax_new                      1 #whether to use the new relaxation method
relax_scale_force              0.5 #controls the size of the first CG step if relax_new is true
out_level                      ie #ie(for electrons); i(for ions);
out_dm                         0 #>0 output density matrix
out_bandgap                    0 #if true, print out bandgap
use_paw                        0 #whether to use PAW in pw calculation
deepks_out_labels              0 #>0 compute descriptor for deepks
deepks_scf                     0 #>0 add V_delta to Hamiltonian
deepks_bandgap                 0 #>0 for bandgap label
deepks_out_unittest            0 #if set 1, prints intermediate quantities that shall be used for making unit test
deepks_model                    #file dir of traced pytorch model: 'model.ptg

#Parameters (5.LCAO)
basis_type                     pw #PW; LCAO in pw; LCAO
gamma_only                     0 #Only for localized orbitals set and gamma point. If set to 1, a fast algorithm is used
search_radius                  -1 #input search radius (Bohr)
search_pbc                     1 #input periodic boundary condition
lcao_ecut                      0 #energy cutoff for LCAO
lcao_dk                        0.01 #delta k for 1D integration in LCAO
lcao_dr                        0.01 #delta r for 1D integration in LCAO
lcao_rmax                      30 #max R for 1D two-center integration table
out_mat_hs                     0 #output H and S matrix (with precision 8)
out_mat_hs2                    0 #output H(R) and S(R) matrix
out_mat_dh                     0 #output of derivative of H(R) matrix
out_mat_xc                     0 #output exchange-correlation matrix in KS-orbital representation
out_interval                   1 #interval for printing H(R) and S(R) matrix during MD
out_app_flag                   1 #whether output r(R), H(R), S(R), T(R), and dH(R) matrices in an append manner during MD
out_mat_t                      0 #output T(R) matrix
out_element_info               0 #output (projected) wavefunction of each element
out_mat_r                      0 #output r(R) matrix
out_wfc_lcao                   0 #ouput LCAO wave functions, 0, no output 1: text, 2: binary
bx                             1 #division of an element grid in FFT grid along x
by                             1 #division of an element grid in FFT grid along y
bz                             1 #division of an element grid in FFT grid along z

#Parameters (6.Smearing)
smearing_method                gauss #type of smearing_method: gauss; fd; fixed; mp; mp2; mv
smearing_sigma                 0.015 #energy range for smearing

#Parameters (7.Charge Mixing)
mixing_type                    broyden #plain; pulay; broyden
mixing_beta                    0.8 #mixing parameter: 0 means no new charge
mixing_ndim                    8 #mixing dimension in pulay or broyden
mixing_restart                 0 #threshold to restart mixing during SCF
mixing_gg0                     1 #mixing parameter in kerker
mixing_beta_mag                -10 #mixing parameter for magnetic density
mixing_gg0_mag                 0 #mixing parameter in kerker
mixing_gg0_min                 0.1 #the minimum kerker coefficient
mixing_angle                   -10 #angle mixing parameter for non-colinear calculations
mixing_tau                     0 #whether to mix tau in mGGA calculation
mixing_dftu                    0 #whether to mix locale in DFT+U calculation
mixing_dmr                     0 #whether to mix real-space density matrix

#Parameters (8.DOS)
dos_emin_ev                    -15 #minimal range for dos
dos_emax_ev                    15 #maximal range for dos
dos_edelta_ev                  0.01 #delta energy for dos
dos_scale                      0.01 #scale dos range by
dos_sigma                      0.07 #gauss b coefficeinet(default=0.07)
dos_nche                       100 #orders of Chebyshev expansions for dos

#Parameters (9.Molecular dynamics)
md_type                        nvt #choose ensemble
md_thermostat                  nhc #choose thermostat
md_nstep                       10 #md steps
md_dt                          1 #time step
md_tchain                      1 #number of Nose-Hoover chains
md_tfirst                      -1 #temperature first
md_tlast                       -1 #temperature last
md_dumpfreq                    1 #The period to dump MD information
md_restartfreq                 5 #The period to output MD restart information
md_seed                        -1 #random seed for MD
md_prec_level                  0 #precision level for vc-md
ref_cell_factor                1 #construct a reference cell bigger than the initial cell
md_restart                     0 #whether restart
lj_rcut                        8.5 #cutoff radius of LJ potential
lj_epsilon                     0.01032 #the value of epsilon for LJ potential
lj_sigma                       3.405 #the value of sigma for LJ potential
pot_file                       graph.pb #the filename of potential files for CMD such as DP
msst_direction                 2 #the direction of shock wave
msst_vel                       0 #the velocity of shock wave
msst_vis                       0 #artificial viscosity
msst_tscale                    0.01 #reduction in initial temperature
msst_qmass                     -1 #mass of thermostat
md_tfreq                       0 #oscillation frequency, used to determine qmass of NHC
md_damp                        1 #damping parameter (time units) used to add force in Langevin method
md_nraise                      1 #parameters used when md_type=nvt
cal_syns                       0 #calculate asynchronous overlap matrix to output for Hefei-NAMD
dmax                           0.01 #maximum displacement of all atoms in one step (bohr)
md_tolerance                   100 #tolerance for velocity rescaling (K)
md_pmode                       iso #NPT ensemble mode: iso, aniso, tri
md_pcouple                     none #whether couple different components: xyz, xy, yz, xz, none
md_pchain                      1 #num of thermostats coupled with barostat
md_pfirst                      -1 #initial target pressure
md_plast                       -1 #final target pressure
md_pfreq                       0 #oscillation frequency, used to determine qmass of thermostats coupled with barostat
dump_force                     1 #output atomic forces into the file MD_dump or not
dump_vel                       1 #output atomic velocities into the file MD_dump or not
dump_virial                    1 #output lattice virial into the file MD_dump or not

#Parameters (10.Electric field and dipole correction)
efield_flag                    0 #add electric field
dip_cor_flag                   0 #dipole correction
efield_dir                     2 #the direction of the electric field or dipole correction
efield_pos_max                 -1 #position of the maximum of the saw-like potential along crystal axis efield_dir
efield_pos_dec                 -1 #zone in the unit cell where the saw-like potential decreases
efield_amp                     0 #amplitude of the electric field

#Parameters (11.Gate field)
gate_flag                      0 #compensating charge or not
zgate                          0.5 #position of charged plate
relax                          0 #allow relaxation along the specific direction
block                          0 #add a block potential or not
block_down                     0.45 #low bound of the block
block_up                       0.55 #high bound of the block
block_height                   0.1 #height of the block

#Parameters (12.Test)
out_alllog                     0 #output information for each processor, when parallel
nurse                          0 #for coders
colour                         0 #for coders, make their live colourful
t_in_h                         1 #calculate the kinetic energy or not
vl_in_h                        1 #calculate the local potential or not
vnl_in_h                       1 #calculate the nonlocal potential or not
vh_in_h                        1 #calculate the hartree potential or not
vion_in_h                      1 #calculate the local ionic potential or not
test_force                     0 #test the force
test_stress                    0 #test the force
test_skip_ewald                0 #skip ewald energy

#Parameters (13.vdW Correction)
vdw_method                     none #the method of calculating vdw (none ; d2 ; d3_0 ; d3_bj
vdw_s6                         default #scale parameter of d2/d3_0/d3_bj
vdw_s8                         default #scale parameter of d3_0/d3_bj
vdw_a1                         default #damping parameter of d3_0/d3_bj
vdw_a2                         default #damping parameter of d3_bj
vdw_d                          20 #damping parameter of d2
vdw_abc                        0 #third-order term?
vdw_C6_file                    default #filename of C6
vdw_C6_unit                    Jnm6/mol #unit of C6, Jnm6/mol or eVA6
vdw_R0_file                    default #filename of R0
vdw_R0_unit                    A #unit of R0, A or Bohr
vdw_cutoff_type                radius #expression model of periodic structure, radius or period
vdw_cutoff_radius              default #radius cutoff for periodic structure
vdw_radius_unit                Bohr #unit of radius cutoff for periodic structure
vdw_cn_thr                     40 #radius cutoff for cn
vdw_cn_thr_unit                Bohr #unit of cn_thr, Bohr or Angstrom
vdw_cutoff_period   3 3 3 #periods of periodic structure

#Parameters (14.exx)
exx_hybrid_alpha               default #fraction of Fock exchange in hybrid functionals
exx_hse_omega                  0.11 #range-separation parameter in HSE functional
exx_separate_loop              1 #if 1, a two-step method is employed, else it will start with a GGA-Loop, and then Hybrid-Loop
exx_hybrid_step                100 #the maximal electronic iteration number in the evaluation of Fock exchange
exx_mixing_beta                1 #mixing_beta for outer-loop when exx_separate_loop=1
exx_lambda                     0.3 #used to compensate for divergence points at G=0 in the evaluation of Fock exchange using lcao_in_pw method
exx_real_number                0 #exx calculated in real or complex
exx_pca_threshold              0.0001 #threshold to screen on-site ABFs in exx
exx_c_threshold                0.0001 #threshold to screen C matrix in exx
exx_v_threshold                0.1 #threshold to screen C matrix in exx
exx_dm_threshold               0.0001 #threshold to screen density matrix in exx
exx_cauchy_threshold           1e-07 #threshold to screen exx using Cauchy-Schwartz inequality
exx_c_grad_threshold           0.0001 #threshold to screen nabla C matrix in exx
exx_v_grad_threshold           0.1 #threshold to screen nabla V matrix in exx
exx_cauchy_force_threshold     1e-07 #threshold to screen exx force using Cauchy-Schwartz inequality
exx_cauchy_stress_threshold    1e-07 #threshold to screen exx stress using Cauchy-Schwartz inequality
exx_ccp_rmesh_times            default #how many times larger the radial mesh required for calculating Columb potential is to that of atomic orbitals
exx_opt_orb_lmax               0 #the maximum l of the spherical Bessel functions for opt ABFs
exx_opt_orb_ecut               0 #the cut-off of plane wave expansion for opt ABFs
exx_opt_orb_tolerence          0 #the threshold when solving for the zeros of spherical Bessel functions for opt ABFs

#Parameters (16.tddft)
td_force_dt                    0.02 #time of force change
td_vext                        0 #add extern potential or not
td_vext_dire                   1 #extern potential direction
out_dipole                     0 #output dipole or not
out_efield                     0 #output dipole or not
out_current                    0 #output current or not
ocp                            0 #change occupation or not
ocp_set                         #set occupation

#Parameters (17.berry_wannier)
berry_phase                    0 #calculate berry phase or not
gdir                           3 #calculate the polarization in the direction of the lattice vector
towannier90                    0 #use wannier90 code interface or not
nnkpfile                       seedname.nnkp #the wannier90 code nnkp file name
wannier_spin                   up #calculate spin in wannier90 code interface
wannier_method                 1 #different implementation methods under Lcao basis set
out_wannier_mmn                1 #output .mmn file or not
out_wannier_amn                1 #output .amn file or not
out_wannier_unk                0 #output UNK. file or not
out_wannier_eig                1 #output .eig file or not
out_wannier_wvfn_formatted     1 #output UNK. file in text format or in binary format

#Parameters (18.implicit_solvation)
imp_sol                        0 #calculate implicit solvation correction or not
eb_k                           80 #the relative permittivity of the bulk solvent
tau                            1.0798e-05 #the effective surface tension parameter
sigma_k                        0.6 # the width of the diffuse cavity
nc_k                           0.00037 # the cut-off charge density

#Parameters (19.orbital free density functional theory)
of_kinetic                     wt #kinetic energy functional, such as tf, vw, wt
of_method                      tn #optimization method used in OFDFT, including cg1, cg2, tn (default)
of_conv                        energy #the convergence criterion, potential, energy (default), or both
of_tole                        1e-06 #tolerance of the energy change (in Ry) for determining the convergence, default=2e-6 Ry
of_tolp                        1e-05 #tolerance of potential for determining the convergence, default=1e-5 in a.u.
of_tf_weight                   1 #weight of TF KEDF
of_vw_weight                   1 #weight of vW KEDF
of_wt_alpha                    0.833333 #parameter alpha of WT KEDF
of_wt_beta                     0.833333 #parameter beta of WT KEDF
of_wt_rho0                     0 #the average density of system, used in WT KEDF, in Bohr^-3
of_hold_rho0                   0 #If set to 1, the rho0 will be fixed even if the volume of system has changed, it will be set to 1 automaticly if of_wt_rho0 is not zero
of_lkt_a                       1.3 #parameter a of LKT KEDF
of_full_pw                     1 #If set to 1, ecut will be ignored when collect planewaves, so that all planewaves will be used
of_full_pw_dim                 0 #If of_full_pw = true, dimention of FFT is testricted to be (0) either odd or even; (1) odd only; (2) even only
of_read_kernel                 0 #If set to 1, the kernel of WT KEDF will be filled from file of_kernel_file, not from formula. Only usable for WT KEDF
of_kernel_file                 WTkernel.txt #The name of WT kernel file.

#Parameters (20.dft+u)
dft_plus_u                     0 #1/2:new/old DFT+U correction method; 0: standard DFT calcullation(default)
yukawa_lambda                  -1 #default:0.0
yukawa_potential               0 #default: false
omc                            0 #the mode of occupation matrix control
onsite_radius                  0 #radius of the sphere for onsite projection (Bohr)
hubbard_u           0 #Hubbard Coulomb interaction parameter U(ev)
orbital_corr        -1 #which correlated orbitals need corrected ; d:2 ,f:3, do not need correction:-1

#Parameters (21.spherical bessel)
bessel_nao_ecut                60.000000 #energy cutoff for spherical bessel functions(Ry)
bessel_nao_tolerence           1e-12 #tolerence for spherical bessel root
bessel_nao_rcut                6 #radial cutoff for spherical bessel functions(a.u.)
bessel_nao_smooth              1 #spherical bessel smooth or not
bessel_nao_sigma               0.1 #spherical bessel smearing_sigma
bessel_descriptor_lmax         2 #lmax used in generating spherical bessel functions
bessel_descriptor_ecut         60.000000 #energy cutoff for spherical bessel functions(Ry)
bessel_descriptor_tolerence    1e-12 #tolerence for spherical bessel root
bessel_descriptor_rcut         6 #radial cutoff for spherical bessel functions(a.u.)
bessel_descriptor_smooth       1 #spherical bessel smooth or not
bessel_descriptor_sigma        0.1 #spherical bessel smearing_sigma

#Parameters (22.non-collinear spin-constrained DFT)
sc_mag_switch                  0 #0: no spin-constrained DFT; 1: constrain atomic magnetization
decay_grad_switch              0 #switch to control gradient break condition
sc_thr                         1e-06 #Convergence criterion of spin-constrained iteration (RMS) in uB
nsc                            100 #Maximal number of spin-constrained iteration
nsc_min                        2 #Minimum number of spin-constrained iteration
sc_scf_nmin                    2 #Minimum number of outer scf loop before initializing lambda loop
alpha_trial                    0.01 #Initial trial step size for lambda in eV/uB^2
sccut                          3 #Maximal step size for lambda in eV/uB
sc_file                        none #file name for parameters used in non-collinear spin-constrained DFT (json format)

#Parameters (23.Quasiatomic Orbital analysis)
qo_switch                      0 #0: no QO analysis; 1: QO analysis
qo_basis                       szv #type of QO basis function: hydrogen: hydrogen-like basis, pswfc: read basis from pseudopotential
qo_thr                         1e-06 #accuracy for evaluating cutoff radius of QO basis function
"""
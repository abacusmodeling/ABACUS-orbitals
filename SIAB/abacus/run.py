'''
Concepts
--------
this module defines different type of workflow of ABACUS calculation.
'''
import numpy as np
import SIAB.io.read_output as read_output
import SIAB.data.interface as db
from scipy.optimize import curve_fit
from SIAB.abacus.utils import is_duplicate
from SIAB.supercomputing.op import submit as envsub
from SIAB.supercomputing.op import op as envop
from SIAB.abacus.io import BLSCAN_WARNMSG, DEFAULT_BOND_LENGTH,\
    archive, configure

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
    element = general["element"]
    folders = []
    if general.get("skip_abacus", False):
        # Skipping calculations as per the configuration to optimize performance
        print("INFO: required by orbital generation setting the 'optimizer' = 'none'/'restart', skip abacus.", flush=True)
        return [[f"{element}-virtual-folder"]]

    for isp, shape in enumerate(structures):
        shape, bond_lengths = shape
        folders_istructure = []
        """abacus_driver can be created iteratively in this layer, and feed in following functions"""
        if bond_lengths == "auto": print(BLSCAN_WARNMSG, flush=True)
        # deal with "auto" keyword
        if bond_lengths == "auto":
            bond_lengths = "default" if element in DEFAULT_BOND_LENGTH.get(shape, {}) else "scan"
        
        #########################
        # Bond length scan task #
        #########################
        if (bond_lengths == "scan" and shape != "monomer"):
            """search bond lengths"""
            if bond_lengths == "default": 
                print("WARNING: default bond length only support dimer/trimer. Now fall back to \"scan\"", flush=True)
            folders_istructure = blscan(general=general,
                                        calculation_setting=calculation_settings[isp],
                                        env_settings=env_settings,
                                        reference_shape=shape,
                                        nstep_bidirection=5,
                                        stepsize=[0.2, 0.5],
                                        ener_thr=1.5,
                                        test=test)
        
        ################
        # Routine task #
        ################
        else:
            bond_lengths = bond_lengths if shape != "monomer" else [0.0]
            bond_lengths = DEFAULT_BOND_LENGTH.get(shape, {})[element] if bond_lengths == "default" else bond_lengths
            assert isinstance(bond_lengths, list), "bond_lengths should be a list"
            assert all([isinstance(bond_length, float) for bond_length in bond_lengths]), "bond_lengths should be a list of floats"
            folders_istructure = normal(general=general,
                                        reference_shape=shape,
                                        bond_lengths=bond_lengths,
                                        calculation_setting=calculation_settings[isp],
                                        env_settings=env_settings,
                                        test=test)
            
        folders.append(folders_istructure)
    return folders

# in the following, defines different tasks
# -------------------------------------------#
# TASK KIND1 - blscan                        #
# DESCRIPTION: search bond lengths           #
# -------------------------------------------#
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
    out = []
    """read energies"""
    use_jy = calculation_setting.get("basis_type", "pw") == "jy"
    rcuts = calculation_setting.get("bessel_nao_rcut")
    if use_jy:
        for rcut in rcuts:
            f_ = [folder for folder in folders if folder.endswith(str(rcut) + "au")]
            bond_lengths = [float(folder.split("-")[-1]) for folder in f_]
            energies = [read_output.read_energy(folder=folder,
                                                suffix=folder) for folder in f_]
            natoms = [read_output.read_natom(folder=folder,
                                            suffix=folder) for folder in f_]
            energies = [energy/natom for energy, natom in zip(energies, natoms)]
            """fitting morse potential"""
            De, a, re, e0 = blscan_fitmorse(bond_lengths, energies)
            """search bond lengths"""
            bond_lengths = blscan_returnbls(bl0=re,
                                            ener0=e0,
                                            bond_lengths=bond_lengths,
                                            energies=energies,
                                            ener_thr=ener_thr)
            f_ = [folder for folder in f_ for bond_length in bond_lengths if "%3.2f"%bond_length in folder]
            out.extend(f_)
        return out
    
    bond_lengths = [float(folder.split("-")[-1]) for folder in folders]
    energies = [read_output.read_energy(folder=folder,
                                        suffix=folder) for folder in folders]
    natoms = [read_output.read_natom(folder=folder,
                                    suffix=folder) for folder in folders]
    energies = [energy/natom for energy, natom in zip(energies, natoms)]
    
    """fitting morse potential"""
    De, a, re, e0 = blscan_fitmorse(bond_lengths, energies)

    """search bond lengths"""
    bond_lengths = blscan_returnbls(bl0=re,
                                    ener0=e0,
                                    bond_lengths=bond_lengths,
                                    energies=energies,
                                    ener_thr=ener_thr)

    return [folder for folder in folders for bond_length in bond_lengths if "%3.2f"%bond_length in folder]

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
    assert idx_min > 1, "There are fewer than 2 points on the left side of the minimum energy point, which indicates unreasonable bond length sampling."
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
        print("\nSummary of bond lengths and energies per atom:".upper(), flush=True)
        print("| Bond length (Angstrom) |   Energy (eV)   | Relative Energy (eV) |", flush=True)
        print("|------------------------|-----------------|----------------------|", flush=True)
        for bl, e, de in zip(bond_lengths, energies, delta_energies):
            line = "|%24.2f|%17.10f|%22.10f|"%(bl, e, de)
            print(line, flush=True)

        raise ValueError("""WARNING: No bond length found with energy higher than %4.2f eV in bond length
search in left direction (bond length decrease direction), this is absolutely unacceptable compared with 
the right direction which may because of low dissociation energy. Exit."""%ener_thr)

    indices = [i_emax_l, (i_emax_l+i_emin)//2, i_emin, (i_emax_r+i_emin)//2, i_emax_r]
    print("\nSummary of bond lengths and energies per atom:".upper(), flush=True)
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
            "fpseudo": general["pseudo_name"], "lattice_constant": 30.0, "nspin": calculation_setting["nspin"],
            "mass": 1.0}
        # SIAB-v3.0 refactor here, change the configure() to generator
        for folder in configure(input_setting=calculation_setting,
                                stru_setting=stru_setting):
            folders.append(folder) if "monomer" not in folder else None
            # check if the calculation is duplicate, if so, skip
            if is_duplicate(folder, calculation_setting):
                print("ABACUS calculation on reference structure %s with bond length %s is skipped."%(reference_shape, bond_length), flush=True)
                envop("rm", "INPUT-%s KPT-%s STRU-%s INPUTw"%(folder, folder, folder), env="local")
                continue
            # else...
            archive(footer=folder)
            print("""
# ----------------------------------------------- #
# Run ABACUS calculation on reference structure.  #
# Reference structure: %10s                 #
# Bond length: %4.2f                               #
# ----------------------------------------------- #
"""%(reference_shape, bond_length), flush=True)
    
    # ========= HERE DIFFERENT PARALLELIZATION ON ABACUS RUN CAN BE IMPLEMENTED ===========
    # presently it is only run in serial...
            _jtg = envsub(folder=folder,
                          module_load_command=env_settings["environment"],
                          mpi_command=env_settings["mpi_command"],
                          program_command=env_settings["abacus_command"],
                          test=test)
    # =====================================================================================
    """wait for all jobs to finish"""
    return folders

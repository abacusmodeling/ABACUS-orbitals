import SIAB.interface.cmd_wrapper as cmdwrp
import SIAB.io.file_manage as file_manage
def _submit(folder: str = "", 
             module_load_command: str = "",
             mpi_command: str = "",
             abacus_command: str = "",
             rcuts: list = [6],
             env: str = "local",
             test: bool = False) -> str:
    
    """submit ABACUS job"""
    jtg = "%s\n"%module_load_command
    jtg += "echo \"present directory: \" `pwd`;\n"
    jtg += "export OMP_NUM_THREADS=1\n"
    jtg += "echo \"OMP_NUM_THREADS:\" $OMP_NUM_THREADS\n"
    jtg += "folder=%s\n"%folder
    jtg += "abacus_command='%s'\n"%(abacus_command)
    jtg += "mpi_command='%s'\n"%(mpi_command)
    jtg += "echo \"run with command: $mpi_command $abacus_command\"\n"
    jtg += "stdbuf -oL $mpi_command $abacus_command"

    if file_manage.skip(folder, rcuts):
        print("skip folder %s"%folder)
        return "skip"
    else:
        if not test:
            hpc_settings = {"shell": True, "text": True, "timeout": 72000}
            cmdwrp.run(command=jtg, env=env, hpc_settings=hpc_settings)

    return jtg

import os
import SIAB.interface.abacus as abacus
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
        folder = abacus.generation(input_setting=calculation_setting,
                                   stru_setting=stru_setting)
        file_manage.archive(footer=folder)
        folders.append(folder)
        """check-in folder and run ABACUS"""
        os.chdir(folder)
        print("""Run ABACUS calculation on reference structure.
Reference structure: %s
Bond length: %s"""%(reference_shape, bond_length))
        _jtg = _submit(folder=folder, 
                       module_load_command=env_settings["environment"],
                       mpi_command=env_settings["mpi_command"],
                       abacus_command=env_settings["abacus_command"],
                       rcuts=calculation_setting["bessel_nao_rcut"],
                       test=test)
        
        os.chdir("../")
    """wait for all jobs to finish"""
    return folders

import numpy as np
from scipy.optimize import curve_fit
def _morse_potential_fitting(bond_lengths: list,
                              energies: list):
    """fitting morse potential, return D_e, a, r_e, e_0 in the equation below:
    
    V(r) = D_e * (1-exp(-a(r-r_e)))^2 + e_0
    
    Use scipy.optimize.curve_fit to fit the parameters
    """
    def morse_potential(r, De, a, re, e0=0.0):
        return De * (1.0 - np.exp(-a*(r-re)))**2.0 + e0
    
    popt, pcov = curve_fit(morse_potential, bond_lengths, energies, bounds=(0, np.inf))
    if pcov is None:
        raise ValueError("fitting failed.")
    elif np.any(np.diag(pcov) < 0):
        raise ValueError("fitting failed.")
    elif np.any(np.diag(pcov) > 1e5):
        print("WARNING: fitting parameters are not accurate.")

    return popt[0], popt[1], popt[2], popt[3]

def _init_blrange(bl0: float, stepsize: list, nstep_bidirection: int = 5):

    blmin = bl0 - stepsize[0]*nstep_bidirection
    blmax = bl0 + stepsize[1]*nstep_bidirection
    print("Searching bond lengths from %4.2f to %4.2f Angstrom, with stepsize %s."%(blmin, blmax, stepsize))
    bond_lengths = np.linspace(blmin, bl0, nstep_bidirection).tolist()
    bond_lengths.extend(np.linspace(bl0, blmax, nstep_bidirection).tolist())
    return [round(bl, 2) for bl in bond_lengths]

def _summarize_blrange(bl0: float, 
                        ener0: float, 
                        bond_lengths: list, 
                        energies: list, 
                        ener_thr: float = 1.0):
    """Get the range of bond lengths corresponding to energies lower than ener_thr"""
    blmin = min(bond_lengths)
    blmax = max(bond_lengths)
    if bl0 < blmin or bl0 > blmax:
        raise ValueError("bl0 is not in the range of bond lengths.")

    # Sort bond lengths and energies according to bond lengths
    bond_lengths, energies = zip(*sorted(zip(bond_lengths, energies)))

    # Split bond lengths and energies into two parts: lower and higher than bl0
    bl_lower = [bl for bl in bond_lengths if bl < bl0]
    ener_lower = [e for e, bl in zip(energies, bond_lengths) if bl < bl0]

    # Find energies <= ener_thr + ener0 for the lower part
    bl_lower = [bl for bl, e in zip(bl_lower, ener_lower) if e <= ener_thr + ener0]
    ener_lower = [e for e in ener_lower if e <= ener_thr + ener0]
    # check if the first energy is approximately equal to ener0 + ener_thr
    
    # Split bond lengths and energies into two parts: lower and higher than bl0
    bl_higher = [bl for bl in bond_lengths if bl > bl0]
    ener_higher = [e for e, bl in zip(energies, bond_lengths) if bl > bl0]

    # Find energies <= ener_thr + ener0 for the higher part
    bl_higher = [bl for bl, e in zip(bl_higher, ener_higher) if e <= ener_thr + ener0]
    ener_higher = [e for e in ener_higher if e <= ener_thr + ener0]

    # Select representative bond lengths
    bond_lengths = [
        bl_lower[0],
        bl_lower[len(bl_lower) // 2],
        bl_lower[-1],
        bl_higher[len(bl_higher) // 2],
        bl_higher[-1]
    ]
    energies = [
        ener_lower[0],
        ener_lower[len(ener_lower) // 2],
        ener_lower[-1],
        ener_higher[len(ener_higher) // 2],
        ener_higher[-1]
    ]

    return [round(bl, 2) for bl in bond_lengths]

import numpy as np
import SIAB.io.read_output as read_output
import SIAB.data.interface as db
def blscan(general: dict,                  # general settings
           calculation_setting: dict,      # calculation setting, for setting up INPUT file
           env_settings: dict,             # calculation environment settings
           reference_shape: str,           # reference shape, always to be dimer
           nstep_bidirection: int = 5,     # number of steps for searching bond lengths per direction
           stepsize: list = [0.2, 0.5],    # stepsize for searching bond lengths, unit in angstrom
           ener_thr: float = 1.0,          # energy threshold for searching bond lengths
           test: bool = True):
    
    """search bond lengths for reference shape, for the case bond lengths is specified as auto
    if this is run, skip the calling of run_abacus function"""
    bond_lengths = _init_blrange(bl0=db.get_radius(general["element"]) * 2.0,
                                  stepsize=stepsize,
                                  nstep_bidirection=nstep_bidirection)
    """generate folders"""
    folders = []
    for bond_length in bond_lengths:
        stru_setting = {
            "shape": reference_shape,
            "bond_length": bond_length,
            "element": general["element"],
            "fpseudo": general["pseudo_name"],
            "lattice_constant": 20.0
        }
        folder = abacus.generation(input_setting=calculation_setting,
                                   stru_setting=stru_setting)
        folders.append(folder)
        file_manage.archive(footer=folder)
        """check-in folder and run ABACUS"""
        os.chdir(folder)
        print("""Run ABACUS calculation on reference structure.
Reference structure: %s
Bond length: %s"""%(reference_shape, bond_length))
        rcuts = calculation_setting["bessel_nao_rcut"]
        _jtg = _submit(folder=folder,
                       module_load_command=env_settings["environment"],
                       mpi_command=env_settings["mpi_command"],
                       abacus_command=env_settings["abacus_command"],
                       rcuts=rcuts,
                       test=test)
        os.chdir("../")
    """wait for all jobs to finish"""
    """read energies"""
    bond_lengths = [float(folder.split("-")[-1]) for folder in folders]
    energies = [read_output.read_energy(folder=folder) for folder in folders]
    """fitting morse potential"""
    De, a, re, e0 = _morse_potential_fitting(bond_lengths, energies)
    print("""Fitting morse potential for reference structure %s.
Equilibrium bond length: %s Angstrom
Dissociation energy: %s eV"""%(reference_shape, re, De))
    """search bond lengths"""
    bond_lengths = _summarize_blrange(bl0=re,
                                      ener0=e0,
                                      bond_lengths=bond_lengths,
                                      energies=energies,
                                      ener_thr=ener_thr)
    folders_to_use = [folder for folder in folders for bond_length in bond_lengths if str(bond_length) in folder]
    return folders_to_use

def iterate(general: dict,
            reference_shapes: list,
            bond_lengths: list,
            calculation_settings: list,
            env_settings: tuple,
            test: bool = False):
    """iterate over all settings and submit jobs"""
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
                                        ener_thr=1.0,
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

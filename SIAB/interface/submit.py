import SIAB.interface.cmd_wrapper as cmdwrp
import SIAB.io.restart as sirst
def _submit(folder: str = "", 
            module_load_command: str = "",
            mpi_command: str = "",
            abacus_command: str = "",
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

    if sirst.abacus_skip(folder=folder):
        print("=> <OVERLAP_Q>, <OVERLAP_Sq> and <OVERLAP_V> storing files detected, \nskip folder %s"%folder)
        return "skip"
    else:
        os.chdir(folder)
        if not test:
            hpc_settings = {"shell": True, "text": True, "timeout": 72000}
            cmdwrp.run(command=jtg, env=env, hpc_settings=hpc_settings)
        os.chdir("../")
    return jtg

import os
import SIAB.interface.abacus as sia
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
        folder = sia.generation(input_setting=calculation_setting,
                                stru_setting=stru_setting)
        sirst.abacus_inputs_archive(footer=folder)
        folders.append(folder)
        print("""Run ABACUS calculation on reference structure.
Reference structure: %s
Bond length: %s"""%(reference_shape, bond_length))
        # need a better design here
        _jtg = _submit(folder=folder,
                       module_load_command=env_settings["environment"],
                       mpi_command=env_settings["mpi_command"],
                       abacus_command=env_settings["abacus_command"],
                       test=test)
        
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
    
    popt, pcov = curve_fit(f=morse_potential, 
                           xdata=bond_lengths, 
                           ydata=energies,
                           p0=[-100, 1.0, 2.0, 0.0])
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
    left = np.linspace(blmin, bl0, nstep_bidirection).tolist()
    right = np.linspace(bl0, blmax, nstep_bidirection).tolist()
    bond_lengths = left + right[1:]
    return [round(bl, 2) for bl in bond_lengths]

def _summarize_blrange(bl0: float, 
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
        raise ValueError("No bond length found with energy higher than %4.2f eV."%ener_thr)
    if i_emax_l == -1:
        print("WARNING: No bond length found with energy higher than %4.2f eV."%ener_thr)

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

import numpy as np
import SIAB.io.read_output as read_output
import SIAB.data.interface as db
def blscan(general: dict,                  # general settings
           calculation_setting: dict,      # calculation setting, for setting up INPUT file
           env_settings: dict,             # calculation environment settings
           reference_shape: str,           # reference shape, always to be dimer
           nstep_bidirection: int = 5,     # number of steps for searching bond lengths per direction
           stepsize: list = [0.2, 0.5],    # stepsize for searching bond lengths, unit in angstrom
           ener_thr: float = 1.5,          # energy threshold for searching bond lengths
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
        folder = sia.generation(input_setting=calculation_setting,
                                stru_setting=stru_setting)
        folders.append(folder)
        sirst.abacus_inputs_archive(footer=folder)
        """check-in folder and run ABACUS"""
        print("""Run ABACUS calculation on reference structure.
Reference structure: %s
Bond length: %s"""%(reference_shape, bond_length))
        _jtg = _submit(folder=folder,
                       module_load_command=env_settings["environment"],
                       mpi_command=env_settings["mpi_command"],
                       abacus_command=env_settings["abacus_command"],
                       test=test)
    """wait for all jobs to finish"""
    """read energies"""
    bond_lengths = [float(folder.split("-")[-1]) for folder in folders]
    energies = [read_output.read_energy(folder=folder,
                                        suffix=folder) for folder in folders]
    """fitting morse potential"""
    De, a, re, e0 = _morse_potential_fitting(bond_lengths, energies)
    print("Morse potential fitting results:")

    print("Fitting morse potential for reference structure %s"%(reference_shape))
    print("%6s: %15.10f %10s (Bond dissociation energy)"%("D_e", De, "eV"))
    print("%6s: %15.10f %10s (Morse potential parameter)"%("a", a, ""))
    print("%6s: %15.10f %10s (Equilibrium bond length)"%("r_e", re, "Angstrom"))
    print("%6s: %15.10f %10s (Zero point energy)"%("e_0", e0, "eV"))

    """search bond lengths"""
    bond_lengths = _summarize_blrange(bl0=re,
                                      ener0=e0,
                                      bond_lengths=bond_lengths,
                                      energies=energies,
                                      ener_thr=ener_thr)

    folders_to_use = [folder for folder in folders for bond_length in bond_lengths if "%3.2f"%bond_length in folder]
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

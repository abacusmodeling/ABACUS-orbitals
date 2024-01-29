"""
DESCRIPTION
----
This is a refactored version of PTG_dpsi method numerical atomic orbital generator,

Contains two functions:
1. generate numerical atomic orbitals taking almost all bands as reference, to
   reproduce all band structures as much as possible, for ABACUS basis_type lcao
   calculation.
2. keep the construction picture of numerical atomic orbitals, but only take some
   of the bands as reference, to reproduce the band structures of interest, for
   wannierize and kpoint extrapolation.

REFACTOR CHECKLIST
----
get_fileString              -> DEPRECATED
get_string_linehead         -> DEPRECATED
get_nRows_linehead          -> DEPRECATED
get_array_linehead          -> DEPRECATED
strs_to_ints                -> DEPRECATED
strs_to_floats              -> DEPRECATED
str_to_bool                 -> DEPRECATED
strs_to_bools               -> DEPRECATED
Search_Num_nearStr          -> DEPRECATED
parse_arguments             -> main
orbConf_to_list             -> database.orbital_configration2list
get_input_STRU              -> abacus.STRU
get_input_KPOINTS           -> abacus.KPOINTS
get_input_INPUTw            -> DEPRECATED
get_input_INPUTs            -> DEPRECATED
get_input_INPUT             -> abacus.INPUT
write_string_tofile         -> DEPRECATED
set_pw_datadir              -> DEPRECATED
pw_calculation              -> run_abacus
define_global_var           -> database.PERIODIC_TABLE_TOINDEX
                               database.PERIODIC_TABLE_TOSYMBOL
                               database.unit_conversion
                               read_input.default
                               read_input.wash
"""
import SIAB.io.read_input as ir
import os
def initialize(version: str = "0.1.0",
               fname: str = "./SIAB_INPUT", 
               pseudopotential_check: bool = True):

    """read input and wash it"""
    fname = fname.strip().replace("\\", "/")
    if version == "0.1.0":
        user_settings = ir.wash(ir.parse(fname=fname))
    elif version == "0.2.0":
        raise NotImplementedError("Version %s not implemented."%version)
    else:
        raise NotImplementedError("Version %s not implemented."%version)
    """ensure the existence of pseudopotential file"""
    fpseudo = user_settings["Pseudo_dir"]+"/"+user_settings["Pseudo_name"]
    if not os.path.exists(fpseudo) and pseudopotential_check: # check the existence of pseudopotential file
        raise FileNotFoundError(
            "Pseudopotential file %s not found"%fpseudo)
    
    reference_shapes, bond_lengths, calculation_settings = ir.unpack_siab_settings(user_settings)
    return user_settings, reference_shapes, bond_lengths, calculation_settings

import SIAB.interface.cmd_wrapper as cmdwrp
def checkpoint(user_settings: dict, 
               rcut: float, 
               orbital_config: str,
               env: str = "local"):
    """After optimization of numerical orbitals' coefficients,
        move generated orbitals to the folder named as:
        [element]_gga_[Ecut]Ry_[Rcut]au_[orbital_config]
    
    Design:
        all information should be included in user_settings
        rather than externally defined additionally.
    
    Args:
        user_settings (dict): user settings
        rcut (float): cutoff radius
        orbital_config (str): orbital configuration, e.g. 1s1p
    
    Returns:
        None
    """
    folder = "_".join([
        user_settings["element"], 
        str(user_settings["Ecut"]) + "Ry", 
        str(rcut)+"au", 
        orbital_config])
    """backup input file, unlike the original version, we fix it must be named as SIAB_INPUT"""
    cmdwrp.op("cp", "SIAB_INPUT", "%s/SIAB_INPUT"%folder, env=env)
    """move spillage.dat"""
    cmdwrp.op("mv", "spillage.dat", "%s/spillage.dat"%folder, env=env)
    """move ORBITAL_PLOTU.dat and ORBITAL_RESULTS.txt"""
    cmdwrp.op("mv", "ORBITAL_PLOTU.dat", "%s/ORBITAL_PLOTU.dat"%folder, env=env)
    cmdwrp.op("mv", "ORBITAL_RESULTS.txt", "%s/ORBITAL_RESULTS.txt"%folder, env=env)
    """move ORBITAL_[element]U.dat to [element]_gga_[Ecut]Ry_[Rcut]au.orb"""
    forb = "%s_gga_%sRy_%sau_%s.orb"%(
        user_settings["element"], str(user_settings["Ecut"]), str(rcut), orbital_config)
    cmdwrp.op("cp", "ORBITAL_%sU.dat"%user_settings["element"], "%s/%s"%(folder, forb), env=env)
    print("Orbital file %s generated."%forb)
    """and directly move it to the folder"""
    cmdwrp.op("mv", "ORBITAL_%sU.dat"%user_settings["element"], "%s/ORBITAL_%sU.dat"%(
        folder, user_settings["element"]), env=env)

import SIAB.interface.submit as submit
def abacus(reference_shapes: list,
           bond_lengths: list,
           calculation_settings: list,
           user_settings: dict,
           test: bool = True):
    """abacus interface"""
    return submit.iterate(reference_shapes=reference_shapes,
                          bond_lengths=bond_lengths,
                          calculation_settings=calculation_settings,
                          user_settings=user_settings,
                          test=test)

import SIAB.spillage as spill
def spillage(folders: list,
             user_settings: dict):
    pass

def driver(fname, test: bool = True):
    # read input
    user_settings, reference_shapes, bond_lengths, calculation_settings = initialize(fname=fname, 
                                                                                     pseudopotential_check=False)

    """ABACUS corresponding refactor has done supporting multiple bessel_nao_rcut input"""
    folders = abacus(reference_shapes=reference_shapes,
                     bond_lengths=bond_lengths,
                     calculation_settings=calculation_settings,
                     user_settings=user_settings,
                     test=test)
    """then call optimizer"""
    spillage(folders=folders,
             user_settings=user_settings)
    
    return folders

import argparse
def main(cmdline_mode: bool = True):
    welcome = """
Starting new version of Systematically Improvable Atomic-orbital Basis (SIAB) method
for generating numerical atomic orbitals (NAOs) for Linar Combinations of Atomic 
Orbitals (LCAO) based electronic structure calculations.

This version is refactored from PTG_dpsi, by ABACUS-AISI developers.
    """
    print(welcome)
    if cmdline_mode:
        parser = argparse.ArgumentParser(description=welcome)
        parser.add_argument(
            "-i", "--input", 
            type=str, 
            default="./SIAB_INPUT", 
            help="input script, default is SIAB_INPUT")
        parser.add_argument(
            "-t", "--test", 
            default=False,
            help="test mode, default is False")
        args = parser.parse_args()

        driver(args.input, args.test)
    else:
        driver("./SIAB_INPUT", False)

if __name__ == '__main__':

    main(cmdline_mode=False)
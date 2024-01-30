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
# interface to initialize
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

    return ir.unpack_siab_input(user_settings)
# interface to abacus
import SIAB.interface.submit as submit
def abacus(general: dict,
           reference_shapes: list,
           bond_lengths: list,
           calculation_settings: list,
           env_settings: tuple,
           test: bool = True):
    """abacus interface"""
    return submit.iterate(general=general,
                          reference_shapes=reference_shapes,
                          bond_lengths=bond_lengths,
                          calculation_settings=calculation_settings,
                          env_settings=env_settings,
                          test=test)
# interface to Spillage optimization
import SIAB.spillage as spill
def spillage(folders: list,
             siab_settings: dict):
    pass

def driver(fname, test: bool = True):
    # read input
    reference_shapes, bond_lengths, calculation_settings,\
    siab_settings, env_settings, general = initialize(fname=fname, 
                                                      pseudopotential_check=False)

    """ABACUS corresponding refactor has done supporting multiple bessel_nao_rcut input"""
    folders = abacus(general=general,
                     reference_shapes=reference_shapes,
                     bond_lengths=bond_lengths,
                     calculation_settings=calculation_settings,
                     env_settings=env_settings,
                     test=test)
    """then call optimizer"""
    spillage(folders=folders,
             siab_settings=siab_settings)
    
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
            action="store",
            help="input script, default is SIAB_INPUT")
        parser.add_argument(
            "-t", "--test",
            type=bool, 
            default=False,
            action="store",
            help="test mode, default is False")
        # add --help
        parser.add_argument(
            "-h", "--help", 
            action="help", 
            help="show this help message and exit")
        args = parser.parse_args()

        outs = driver(args.input, args.test)
    else:
        outs = driver("./SIAB_INPUT", False)
    print(outs)

if __name__ == '__main__':

    main(cmdline_mode=False)
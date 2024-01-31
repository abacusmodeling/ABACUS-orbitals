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
import argparse
def initialize(command_line: bool = True):
    welcome = """
Starting new version of Systematically Improvable Atomic-orbital Basis (SIAB) method
for generating numerical atomic orbitals (NAOs) for Linar Combinations of Atomic 
Orbitals (LCAO) based electronic structure calculations.

This version is refactored from PTG_dpsi, by ABACUS-AISI developers.
    """
    print(welcome)
    placeholder_1 = ""
    placeholder_2 = ""
    if command_line:
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

        placeholder_1 = args.input
        placeholder_2 = args.test
    else:
        placeholder_1 = "./SIAB_INPUT"
        placeholder_2 = False

    return  placeholder_1, placeholder_2 

import SIAB.driver.front as sdf
def run(fname: str, version: str = "0.1.0", test: bool = True):
    # read input
    reference_shapes, bond_lengths, calculation_settings,\
    siab_settings, env_settings, general = sdf.initialize(fname=fname,
                                                          version=version, 
                                                          pseudopotential_check=False)

    """ABACUS corresponding refactor has done supporting multiple bessel_nao_rcut input"""
    folders = sdf.abacus(general=general,
                         reference_shapes=reference_shapes,
                         bond_lengths=bond_lengths,
                         calculation_settings=calculation_settings,
                         env_settings=env_settings,
                         test=test)
    """then call optimizer"""
    sdf.spillage(folders=folders,
                 calculation_setting=calculation_settings[0],
                 siab_settings=siab_settings)
    
    return "seems everything is fine"

def finalize(outs: str):
    print(outs)

def main(command_line: bool = True):

    fname, test = initialize(command_line=command_line)
    outs = run(fname=fname, version="0.1.0", test=test)
    finalize(outs=outs)

if __name__ == '__main__':

    main(command_line=False)
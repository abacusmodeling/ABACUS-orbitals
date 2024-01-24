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
import read_input as ir
import abacus as abacus
import os
def initialize(fname: str = "./SIAB_INPUT", pseudopotential_check: bool = True):

    """read input and wash it"""
    fname = fname.strip().replace("\\", "/")
    user_settings = ir.wash(ir.parse(fname=fname))
    """ensure the existence of pseudopotential file"""
    fpseudo = user_settings["Pseudo_dir"]+"/"+user_settings["Pseudo_name"]
    if not os.path.exists(fpseudo) and pseudopotential_check: # check the existence of pseudopotential file
        raise FileNotFoundError(
            "Pseudopotential file %s not found"%fpseudo)

    return user_settings

def unpack_siab_settings(user_settings: dict):
    """unpack SIAB_INPUT settings for easy generation of ABACUS input files"""
    bond_lengths = [[
            bond_length for bond_length in user_settings[key][4:]
        ] for key in user_settings.keys() if key.startswith("STRU")
    ]
    reference_shape = [
        user_settings[key][0] for key in user_settings.keys() if key.startswith("STRU")
    ]
    readin_calc_settings = {
        keywords_translate(key): user_settings[key] for key in user_settings.keys() if key in [
            "Ecut", "Rcut", "Pseudo_dir", "sigma"
        ]}
    readin_calc_settings["nbands"] = [
        int(user_settings[key][1]) for key in user_settings.keys() if key.startswith("STRU")
        ]
    readin_calc_settings["lmaxmax"] = max([
        int(user_settings[key][2]) for key in user_settings.keys() if key.startswith("STRU")
        ])
    readin_calc_settings["nspin"] = [
        int(user_settings[key][3]) for key in user_settings.keys() if key.startswith("STRU")
        ]
    
    calculation_settings = [{} for _ in range(len(reference_shape))]
    for key, value in readin_calc_settings.items():
        if key != "bessel_nao_rcut":
            if isinstance(value, list):
                for i, val in enumerate(value):
                    calculation_settings[i][key] = val
            else:
                for settings in calculation_settings:
                    settings[key] = value
        else:
            bessel_nao_rcut = " ".join([str(v) for v in value])
            for settings in calculation_settings:
                settings[key] = bessel_nao_rcut

    return reference_shape, bond_lengths, calculation_settings

def keywords_translate(keyword: str):

    if keyword == "Ecut":
        return "ecutwfc"
    elif keyword == "Rcut":
        return "bessel_nao_rcut"
    elif keyword == "Pseudo_dir":
        return "pseudo_dir"
    elif keyword == "sigma":
        return "smearing_sigma"
    else:
        return keyword

import cmd_wrapper as cmdwrp
def archive(footer: str = "", env: str = "local"):

    """archive the results"""
    headers = ["INPUT", "STRU", "KPT"]
    if footer != "":
        cmdwrp.op("mkdir", footer, additional_args=["-p"], env=env)
        for header in headers:
            if header == "INPUT":
                cmdwrp.op("mv", "%s-%s"%(header, footer), "%s/INPUT"%(footer), env=env)
            else:
                cmdwrp.op("mv", "%s-%s"%(header, footer), "%s/"%(footer), env=env)
    else:
        raise ValueError("footer is not specified")

def run_abacus(reference_shapes: list,
               bond_lengths: list,
               calculation_settings: list,
               user_settings: dict,
               test: bool = True):
    """iteratively run ABACUS calculation on reference structures
    To let optimizer be easy to find output, return names of folders"""
    nstructure_ref = 0
    if len(reference_shapes)*len(bond_lengths)*len(calculation_settings) != len(reference_shapes)**3:
        raise ValueError("number of settings not aligned.")
    else:
        nstructure_ref = len(reference_shapes)
    
    folders = []
    for i in range(nstructure_ref):
        folders_istructure = []
        for bond_length in bond_lengths[i]:
            stru_settings = {
                "shape": reference_shapes[i],
                "bond_length": bond_length,
                "element": user_settings["element"],
                "fpseudo": user_settings["Pseudo_name"],
                "lattice_constant": 20.0
            }
            folder = abacus.generation(input_settings=calculation_settings[i],
                                       stru_settings=stru_settings)
            folders_istructure.append(folder)
            archive(footer=folder)
            """check-in folder and run ABACUS"""
            os.chdir(folder)
            print("""Run ABACUS calculation on reference structure.
Reference structure: %s
Bond length: %s"""%(reference_shapes[i], bond_length))
            _jtg = abacus.submit(folder=folder, 
                                 module_load_command=user_settings["EXE_env"],
                                 mpi_command=user_settings["EXE_mpi"],
                                 abacus_command=user_settings["EXE_pw"],
                                 rcuts=user_settings["Rcut"],
                                 test=test)
            
            os.chdir("../")
        folders.append(folders_istructure)
    """wait for all jobs to finish"""
    
    return folders

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

def driver(fname, test: bool = True):
    # read input
    user_settings = initialize(fname, False)
    #print(user_settings)
    # in new SIAB version, we calculate SCF on each bong length, each kind of reference structures
    # for only once. Parse the orbital overlap files named as "orb_matrix_rcutXderivY.dat", X is 
    # according to bessel_nao_rcut the user setting from SIAB_INPUT, Y is always to be 0 and 1 for
    # PTG_dpsi method
    reference_shapes, bond_lengths, calculation_settings = unpack_siab_settings(user_settings) 

    """ABACUS corresponding refactor has done supporting multiple bessel_nao_rcut input"""
    folders = run_abacus(reference_shapes=reference_shapes,
                         bond_lengths=bond_lengths,
                         calculation_settings=calculation_settings,
                         user_settings=user_settings,
                         test=test)
    return folders
    """then call optimizer"""
    



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
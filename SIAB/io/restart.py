"""a module controlling how to restart a job"""
import SIAB.interface.cmd_wrapper as cmdwrp
def abacus_inputs_archive(footer: str = "", env: str = "local"):

    """archive the results"""
    headers = ["INPUT", "STRU", "KPT"]
    if footer != "":
        cmdwrp.op("mkdir", footer, additional_args=["-p"], env=env)
        for header in headers:
            if header == "INPUT":
                cmdwrp.op("mv", "%s-%s"%(header, footer), "%s/INPUT"%(footer), env=env)
            else:
                cmdwrp.op("mv", "%s-%s"%(header, footer), "%s/"%(footer), env=env)
        cmdwrp.op("mv", "INPUTw", "%s/INPUTw"%(footer), env=env)
    else:
        raise ValueError("footer is not specified")

import os
import re
def abacus_skip(folder: str):
    """check if the abacus calculation is skipped"""
    if not os.path.isdir(folder):
        return False
    files = os.listdir(folder)

    ov_qsv = []
    nv_qsv = []

    ov_pattern = r"^(orb_matrix\.)([01])(\.dat)$"
    nv_pattern = r"^(orb_matrix_rcut)([0-9]+)(deriv)([01])(\.dat)$"
    for file in files:
        if re.match(ov_pattern, file):
            ov_qsv.append(file)
        elif re.match(nv_pattern, file):
            nv_qsv.append(file)
        else:
            continue
    if len(ov_qsv) == 2:
        header = "orb_matrix"
        if header+".0.dat" in ov_qsv and header+".1.dat" in ov_qsv:
            return True
        else:
            return False
    elif len(ov_qsv) != 0:
        return False
    elif len(nv_qsv) != 0:
        header = "orb_matrix_rcut"
        for file in nv_qsv:
            _match = re.match(nv_pattern, file)
            rcut = _match.group(2)
            deriv = int(_match.group(4))
            counter = "%s%sderiv%s.dat"%(header, rcut, abs(deriv-1))
            if counter not in nv_qsv:
                return False
        return True
    else:
        return False
    
import SIAB.data.interface as sdi
def checkpoint(src: str,
               dst: str,
               user_settings: dict, 
               rcut: float, 
               orbital_config: str,
               env: str = "local"):
    """After optimization of numerical orbitals' coefficients,
        move generated orbitals to the folder named as:
        [element]_gga_[Ecut]Ry_[Rcut]au_[orbital_config]

        ONCE ONE OPTIMIZATION TASK COMPLETES, CALL THIS FUNCTION.
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
    # first check if the folder exists, if not, create it
    if not os.path.isdir(dst):
        cmdwrp.op("mkdir", dst, additional_args=["-p"], env=env)
    """backup input file, unlike the original version, we fix it must be named as SIAB_INPUT"""
    cmdwrp.op("cp", "%s/SIAB_INPUT"%src, "%s/SIAB_INPUT"%dst, env=env)
    """move spillage.dat"""
    cmdwrp.op("mv", "%s/spillage.dat"%src, "%s/spillage.dat"%dst, env=env)
    """move ORBITAL_PLOTU.dat and ORBITAL_RESULTS.txt"""
    cmdwrp.op("mv", "%s/ORBITAL_PLOTU.dat"%src, "%s/ORBITAL_PLOTU.dat"%dst, env=env)
    cmdwrp.op("mv", "%s/ORBITAL_RESULTS.txt"%src, "%s/ORBITAL_RESULTS.txt"%dst, env=env)
    """move ORBITAL_[element]U.dat to [element]_gga_[Ecut]Ry_[Rcut]au.orb"""
    forb = "%s_gga_%sRy_%sau_%s.orb"%(
        user_settings["element"], str(user_settings["Ecut"]), str(rcut), orbital_config)
    index = sdi.PERIODIC_TABLE_TOINDEX[user_settings["element"]]
    cmdwrp.op("cp", "%s/ORBITAL_%sU.dat"%(src, index), "%s/%s"%(dst, forb), env=env)
    print("Orbital file %s generated."%forb)
    """and directly move it to the folder"""
    cmdwrp.op("mv", "%s/ORBITAL_%sU.dat"%(src, index), "%s/ORBITAL_%sU.dat"%(dst, index), env=env)

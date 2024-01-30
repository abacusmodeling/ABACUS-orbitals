"""a module controlling how to restart a job"""

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

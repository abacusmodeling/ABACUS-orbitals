import SIAB.interface.cmd_wrapper as cmdwrp
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

import os
def skip(folder: str = "",
         rcuts: list = [6],
         derivs: list = [0, 1]):
    """compatible with old ABACUS version generated orb_matrix.0.dat and orb_matrix.1.dat"""
    fname_pattern = r"^(orb_matrix.)(rcut)?([0-9]+)?(deriv)?([01]{1})(.dat)$"

    if not os.path.isdir(folder):
        return False
    files = os.listdir(folder)
    if len(rcuts) == 0:
        """the old version, only check if there are those two files"""
        if "orb_matrix.0.dat" in files and "orb_matrix.1.dat" in files:
            return True
        else:
            return False
    else:
        """the new version, check if all rcut values in rcuts are calculated"""
        for rcut in rcuts:
            for deriv in derivs:
                fname = "orb_matrix.rcut%dderiv%d.dat"%(rcut, deriv)
                if fname not in files:
                    return False
        return True

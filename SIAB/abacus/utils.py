'''
functionalities related to ABACUS
'''
import os
from SIAB.abacus.io import read_INPUT

##############################################
#           general information              #
##############################################
def version_compare(version_1: str, version_2: str) -> bool:
    """compare two version strings, return True if version_1 <= version_2"""
    version_1 = version_1.split(".")
    version_2 = version_2.split(".")
    for i in range(len(version_1)):
        if int(version_1[i]) < int(version_2[i]):
            return True
        elif int(version_1[i]) > int(version_2[i]):
            return False
        else:
            continue
    return True

def is_duplicate(folder: str, abacus_setting: dict):
    """check if the abacus calculation can be safely (really?)
    skipped"""
    # STAGE1: existence of folder
    if not os.path.isdir(folder):
        return False
    files = os.listdir(folder)
    print("DUPLICATE CHECK-1 pass: folder %s exists"%folder, flush=True)
    # STAGE2: existence of INPUT files   
    for fcplsry in ["INPUT", "INPUTw"]:
        if fcplsry not in files:
            return False
    print("DUPLICATE CHECK-2 pass: INPUT and INPUTw exist", flush=True)
    # STAGE3: correspondence of INPUT settings
    for key in ["bessel_nao_rcut", "lmaxmax"]:
        if key not in abacus_setting.keys():
            raise ValueError("NECESSARY KEYWORD %s is not specified"%key)
    original = read_INPUT(folder)

    check_keys = [k for k in abacus_setting.keys() if k not in ["orbital_dir", "bessel_nao_rcut"]]
    check_keys = list(abacus_setting.keys())\
        if abacus_setting.get("basis_type", "pw") == "pw" else check_keys
    for key in check_keys:
        value = abacus_setting[key]
        if isinstance(value, list):
            value = " ".join([str(v) for v in value])
        else:
            value = str(value)
        value_ = original.get(key, None)
        # for jy, it is different here. Because the forb is no where to store, all orbitals
        # involved are temporarily stored in the value of key "orbital_dir". Thus the following
        # will fail for jy for two keys: orbital_dir and bessel_nao_rcut, the latter is because
        # for jy, one SCF can only have one rcut.
        if value_ != value:
            print("KEYWORD \"%s\" has different values. Original: %s, new: %s\nDifference \
                  detected, start a new job."%(key, value_, value), flush=True)
            return False
    
    # for jy, the following will also fail, because jy will not print such matrix, instead, 
    # there will only be several matrices such as T(k), S(k), H(k) and wavefunction file.    
    print("DUPLICATE CHECK-3 pass: INPUT settings are consistent", flush=True)

    # STAGE4: existence of crucial output files
    rcuts = abacus_setting["bessel_nao_rcut"]
    rcuts = [rcuts] if not isinstance(rcuts, list) else rcuts
    print(original.get("bessel_nao_rcut"))
    if abacus_setting.get("basis_type", "pw") != "pw" and \
        float(original.get("bessel_nao_rcut", 0)) in [float(rcut) for rcut in rcuts]:
        print("DUPLICATE CHECK-4 pass: realspace cutoff matches (file integrities not checked)", 
              flush=True)
        return True
    
    if len(rcuts) == 1:
        if "orb_matrix.0.dat" not in files:
            return False
        if "orb_matrix.1.dat" not in files:
            return False
    else:
        for rcut in rcuts:
            if "orb_matrix_rcut%sderiv0.dat"%rcut not in files:
                return False
            if "orb_matrix_rcut%sderiv1.dat"%rcut not in files:
                return False
    print("DUPLICATE CHECK-4 pass: crucial output files exist", flush=True)
    return True

"""This module is for making compatible with previous version of SIAB, which needs
the "INPUT.json" file for performing optimization task. """

def to_bc():
    """this function is for dumping the old "INPUT.json" file for performing optimization
    task, read by opt_orb_pytorch_dpsi/main.py
    
    here named the dict object as to_bc_settings, it has the structure as below:
    {
        "files": {
            "origin": [],
            "linear": []
        },
        "parameters": {},
        "weights": {},
        "sphebes_coeffs": {},
        "V": {}
    }
    """

def bc_parameters(element: str,
                  orbital_config: list,
                  bessel_nao_rcut: float,
                  bessel_nao_ecut: float,
                  dr: float = 0.01,
                  opt_maxsteps: int = 1000,
                  lr: float = 0.03,
                  calc_kinetic_ener: bool = False,
                  calc_smooth: bool = True
                  ):
    """this function is for dumping the old "INPUT.json" file for performing optimization
    parameters for SINGLE ELEMENT, to get identical "info" section contents, call merge_param
    function and feed list of what this function returns for elements"""

    keys = ["Nt_all", "Nu", "Rcut", "dr", "Ecut", "lr", "cal_T", "cal_smooth", "max_steps"]

    return dict(zip(keys, [element, orbital_config, 
                           bessel_nao_rcut, dr, bessel_nao_ecut,
                            lr, calc_kinetic_ener, calc_smooth, opt_maxsteps]
                    ))

def merge_bcparam(bcparams: list):
    """this function is for merging all parameters of each element packed in dicts organized
    in one list as input, into one dict. 
    Except the "Nt_all", which would be a list of element after merging, and lr, cal_T, 
     cal_smooth and max_steps whose values would be shared by all elements and will still be
    scalar values, all other keys' values would be dicts with element as key and value 
    as the corresponding value of that element."""

    keys_to_merge = ["Nu", "Rcut", "dr", "Ecut"]
    """check if all dicts in bcparams have the same values for lr, cal_T, cal_smooth and max_steps"""
    for key in ["lr", "cal_T", "cal_smooth", "max_steps"]:
        for bcparam in bcparams:
            if bcparam[key] != bcparams[0][key]:
                raise ValueError(f"bcparams have different values for {key}")
    """merge the dicts in bcparams"""
    merged = {
        "Nt_all": [bcparam["Nt_all"] for bcparam in bcparams],
        "lr": bcparams[0]["lr"],
        "cal_T": bcparams[0]["cal_T"],
        "cal_smooth": bcparams[0]["cal_smooth"],
        "max_steps": bcparams[0]["max_steps"]
    }
    for key in keys_to_merge:
        merged[key] = dict(zip(merged["Nt_all"], [bcparam[key] for bcparam in bcparams]))
    return merged

def bc_weights():
    pass
"""This module is for making compatible with previous version of PTG_dpsi, 
The optimizer needs the "INPUT.json" file for performing optimization task. """

"""variables appear in function prepare_SIAB_INPUT()
refSTRU_Level: ["STRU1", "STRU2"]
orbConf_Level: [
#   element1, element2, ...
    [2s2p1d, 4s2p2d1f, ...], # level1
    [2s2p1d, 4s2p2d1f, ...], # level2
    ...
]
refBands_Level: [
# bl: bond length
#   bl1, bl2, ...
    [18, 18, 18, 18, 18], # level1
    [18, 18, 18, 18, 18], # level2
    [27, 27, 27]          # level3
]
BL_STRU: {
    "STRU1": [bl1, bl2, ...],
    "STRU2": [bl1, bl2, ...]
}
nBL_STRU: {
    "STRU1": 5,
    "STRU2": 3
}
fixPre_Level: [
    "none", "fix", "fix", ... 
]
"""

def to_oldversion():
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

def ov_parameters(element: str,
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

def merge_ovparam(params: list):
    """this function is for merging all parameters of each element packed in dicts organized
    in one list as input, into one dict. 
    Except the "Nt_all", which would be a list of element after merging, and lr, cal_T, 
     cal_smooth and max_steps whose values would be shared by all elements and will still be
    scalar values, all other keys' values would be dicts with element as key and value 
    as the corresponding value of that element."""
    """this function corresponds to original version SIAB.py line 411 to 418"""
    keys_to_merge = ["Nu", "Rcut", "dr", "Ecut"]
    """check if all dicts in params have the same values for lr, cal_T, cal_smooth and max_steps"""
    for key in ["lr", "cal_T", "cal_smooth", "max_steps"]:
        for param in params:
            if param[key] != params[0][key]:
                raise ValueError(f"params have different values for {key}")
    """merge the dicts in params"""
    merged = {
        "Nt_all": [param["Nt_all"] for param in params],
        "lr": params[0]["lr"],
        "cal_T": params[0]["cal_T"],
        "cal_smooth": params[0]["cal_smooth"],
        "max_steps": params[0]["max_steps"]
    }
    for key in keys_to_merge:
        merged[key] = dict(zip(merged["Nt_all"], [param[key] for param in params]))
    return merged

def ov_weights():
    pass

def ov_ovlps():
    pass

def ov_reference_states():
    pass
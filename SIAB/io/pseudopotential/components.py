"""one cannot directly call functions defined in kernel module"""

import SIAB.io.pseudopotential.kernel as kernel

def parse(fname: str):
    """from API, it is important to directly call functions defined in kernel,
    therefore set up this interface function"""
    import os
    if os.path.basename(fname).upper().endswith(".UPF"):
        return kernel.upf(fname=fname)
    elif os.path.basename(fname).lower().startswith("vwr."):
        return kernel.vwr(fname=fname)
    else:
        raise TypeError("Unknown pseudopotential format")
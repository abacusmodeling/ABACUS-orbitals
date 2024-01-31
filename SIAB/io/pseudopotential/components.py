"""one cannot directly call functions defined in kernel module"""

import SIAB.io.pseudopotential.kernel as kernel

def parse(fname: str):
    """from API, it is important to directly call functions defined in kernel,
    therefore set up this interface function"""
    return kernel.parse(fname=fname)
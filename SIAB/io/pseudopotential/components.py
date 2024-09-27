"""one cannot directly call functions defined in kernel module"""

import SIAB.io.pseudopotential.kernel as kernel

def parse(fname: str):
    """from API, it is important to directly call functions defined in kernel,
    therefore set up this interface function.
    
    The function will return a dictionary contains information."""
    import os
    error_msg = """ERROR: unknown pseudopotential format. The format of pseudopotential
is determined by the name of the file and the logic is quite simple, if it is the one
with UPF-standard format, the file name should end with .UPF, if it is the one with
vwr format, the file name should start with vwr. Thus for the case you exactly know
the format of pseudopotential you provided, please rename the file and retry.
Thanks for understanding.
Quit with TypeError raised..."""
    if os.path.basename(fname).upper().endswith(".UPF"):
        # Add error-catching for UPF format at Aug 6, 2024
        return kernel.upf(fname=fname)
    elif os.path.basename(fname).lower().startswith("vwr."):
        # presently without error-catching for vwr format
        return kernel.vwr(fname=fname)
    else:
        print(error_msg, flush=True)
        raise TypeError("ERROR: Please read the error message above.")
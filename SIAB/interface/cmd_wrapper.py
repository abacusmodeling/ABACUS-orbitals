"""this module is for converting common Liunx command to the version
compatible with High Performance Computing (HPC) system or say for
Supercomputer."""

import subprocess
import sys
import os

def run(command: str,
        env: str = "local",
        additional_args: list = None,
        hpc_settings: dict = None):
    
    if additional_args is not None:
        command = " ".join([command, *additional_args])
    if hpc_settings is None:
        hpc_settings = {
            "stdin": subprocess.DEVNULL,
            "shell": True,
            "text": True,
            "timeout": 60
        }
    
    """run command in different environment"""
    if env == "local":
        value = os.system(command)
    elif env == "hpc":
        sys.stdout.flush()
        value = subprocess.run(command, **hpc_settings)
        sys.stdout.flush()
    return value

def op(operation: str,
       src: str,
       dst: str = "",
       env: str = "local",
       additional_args: list = None,
       hpc_settings: dict = None):
    
    if additional_args is None:
        additional_args = []
    additional_args = [operation, *additional_args]
    return run(command=" ".join([*additional_args, src, dst]).replace("  ", " "),
               env=env,
               hpc_settings=hpc_settings)


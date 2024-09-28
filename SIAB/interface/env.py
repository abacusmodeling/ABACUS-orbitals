"""this module is for converting common Liunx command to the version
compatible with High Performance Computing (HPC) system or say for
Supercomputer."""
import os
import subprocess
import sys

##############################################
#                                            #
##############################################
def submit(folder: str = "", 
           module_load_command: str = "",
           mpi_command: str = "",
           program_command: str = "",
           env: str = "local",
           test: bool = False) -> str:
    
    """general submit function with compatibility of HPC systems"""
    jtg = "%s\n"%module_load_command
    jtg += "echo \"present directory: \" `pwd`;\n"
    jtg += "export OMP_NUM_THREADS=1\n"
    jtg += "echo \"OMP_NUM_THREADS:\" $OMP_NUM_THREADS\n"
    jtg += "folder=%s\n"%folder
    jtg += "program_command='%s'\n"%(program_command)
    jtg += "mpi_command='%s'\n"%(mpi_command)
    jtg += "echo \"run with command: $mpi_command $program_command\"\n"
    jtg += "stdbuf -oL $mpi_command $program_command"

    os.chdir(folder)
    if not test:
        hpc_settings = {"shell": True, "text": True, "timeout": 72000}
        run(command=jtg, env=env, hpc_settings=hpc_settings)
    os.chdir("../")
    return jtg

##############################################
#        basic wrapped linux commands        #
##############################################

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


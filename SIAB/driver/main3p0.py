from SIAB.io.param import read
from SIAB.supercomputing.op import submit
from SIAB.abacus.api import build_abacus_jobs
from SIAB.orb.orb import cascade_gen
def init(fn):
     """
     initialize the ABACUS-ORBGEN workflow by reading the input file

     Parameters
     ----------
     fn: str
         input filename
     """
     return read(fn)

def rundft(elem,
           rcuts,
           dftparam,
           geoms,
           spill_guess,
           compparam):
     jobs = build_abacus_jobs(elem, rcuts, dftparam, geoms, spill_guess)
     for job in jobs:
         _ = submit(job, 
                    compparam.get('environment', ''),
                    compparam.get('mpi_command', ''),
                    compparam.get('abacus_command', 'abacus'))
     # to find these folders by name, call function
     # in SIAB.io.convention the dft_folder
     # dft_folder(elem, proto, pert, rcut = None)
     return jobs

def spillage(elem, rcuts, ecut, primtive_type, mode, scheme, jobs):
     pass
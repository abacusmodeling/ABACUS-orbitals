from SIAB.io.param import read, orb_link_geom
from SIAB.supercomputing.op import submit
from SIAB.abacus.api import build_abacus_jobs, job_done
from SIAB.io.convention import dft_folder
from SIAB.orb.api import orb_cascade

def init(fn):
     """
     initialize the ABACUS-ORBGEN workflow by reading the input file

     Parameters
     ----------
     fn: str
         input filename
     """
     glbparams, dftparams, spillparams, compute = read(fn)
     # if fit_basis is jy, then set basis_type in dftparams to lcao explicitly
     if spillparams.get('fit_basis', 'jy') == 'jy':
          dftparams['basis_type'] = 'lcao'
     else:
          dftparams['basis_type'] = 'pw'
     # link the geometries
     for orb in spillparams['orbitals']:
          orb['geoms'] = orb_link_geom(orb['geoms'], spillparams['geoms'])
          
     return glbparams, dftparams, spillparams, compute

def rundft(elem,
           rcuts,
           dftparam,
           geoms,
           spill_guess,
           compparam):
     jobs = build_abacus_jobs(elem, rcuts, dftparam, geoms, spill_guess)
     # there should be a check here, to avoid rerun completed jobs
     for job in jobs:
         if job_done(job):
             print(f'{job} has been done, skip')
             continue
         _ = submit(job, 
                    compparam.get('environment', ''),
                    compparam.get('mpi_command', ''),
                    compparam.get('abacus_command', 'abacus'))
     # to find these folders by name, call function
     # in SIAB.io.convention the dft_folder
     # dft_folder(elem, proto, pert, rcut = None)
     return jobs

def _spilltasks(elem, 
                rcuts, 
                scheme, 
                basis_type = 'jy'):
     '''
     
     Parameters
     ----------
     elem: str
         element symbol
     ecut: float
         the kinetic energy cutoff of planewave and spherical wave
     rcuts: list[float]
         the cutoff radius of orbital to generate
     primitive_type: str
          the type of primitive jy (spherical wave) basis set
     scheme: list
          the scheme of how to generate the orbitals
     jobs: list[str]
          the folders of the dft calculations
     '''
     convert_ = {'nzeta': 'nzeta', 'nbands': 'nbnds', 
                 'checkpoint': 'iorb_frozen', 'geoms': 'folders'}
     for rcut in rcuts:
          template = scheme.copy()
          orbitals = [{convert_[k]: v for k, v in orb.items()} for orb in template]
          additional = {} if basis_type != 'jy' else {'rcut': rcut}
          for orb in orbitals:
               geoms_orb = [{'elem': elem, 'proto': f['proto'], 'pert': pertmag} 
                            for f in orb['folders'] for pertmag in f['pertmags']]
               orb['folders'] = [dft_folder(**(geom|additional)) for geom in geoms_orb]
          yield rcut, orbitals

def spillage(elem, 
             ecut, 
             rcuts, 
             primitive_type,  
             scheme, 
             basis_type = 'jy',
             **kwargs):
     options = {'maxiter': kwargs.get('max_steps', 3000), 
                'disp': kwargs.get('spill_verbo', False),
                'ftol': kwargs.get('ftol', 0),
                'gtol': kwargs.get('gtol', 1e-6),
                'maxcor': 20}
     for rcut, task in _spilltasks(elem, rcuts, scheme, basis_type):
          initializer = {} if basis_type != 'jy' else {'rcut': rcut}
          cascade = orb_cascade(elem, 
                                rcut, 
                                ecut, 
                                primitive_type,
                                dft_folder(elem, 'monomer', 0, **initializer),
                                task,
                                basis_type)
          cascade.opt(immediplot='.', 
                      diagnosis=True, 
                      options=options, 
                      nthreads=kwargs.get('nthreads_rcut', 1))
          # cascade.plot('.')
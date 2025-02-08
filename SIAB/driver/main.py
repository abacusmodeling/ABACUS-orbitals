from SIAB.io.param import read, orb_link_geom
from SIAB.supercomputing.op import submit
from SIAB.abacus.api import build_abacus_jobs, job_done
from SIAB.io.convention import dft_folder
from SIAB.orb.api import GetOrbCascadeInstance
from SIAB.abacus.blscan import jobfilter
from SIAB.spillage.util import _spillparam

def init(fn):
     '''
     initialize the ABACUS-ORBGEN workflow by reading the input file

     Parameters
     ----------
     fn: str
         input filename
     '''
     glbparams, dftparams, spillparams, compute = read(fn)
     
     # if fit_basis is jy, then set basis_type in dftparams to lcao explicitly
     dftparams['basis_type'] = 'lcao' if spillparams.get('fit_basis', 'jy') == 'jy' else 'pw'
     
     # link the geometries
     for orb in spillparams['orbitals']:
          orb['geoms'] = orb_link_geom(orb['geoms'], spillparams['geoms'])
          
     return glbparams, dftparams, spillparams, compute

def rundft(atomspecies,
           rcuts,
           dftparam,
           geoms,
           spillguess,
           compparam,
           **kwargs):
     '''
     run the ABACUS DFT calculations to generate the reference wavefunctions
     
     Parameters
     ----------
     elem: str
         element symbol
     rcuts: list[float]
          the cutoff radius of orbital to generate. this parameter will affect
          in two aspects: 1. in the ABACUS INPUT parameter, 2. the definition
          of the orbital to generate
     dftparam: dict
          other ABACUS INPUT parameters
     geoms: dict
          the geometries that ABACUS perform the DFT calculations on
     spillguess: str|None
          the initial guess of the spillage optimization, now it must be `atomic`
     compparam: dict
          the computational parameters, including `abacus_command`, `environment`,
          `mpi_command`
     ecutjy: float, optional
          the kinetic energy cutoff of the underlying jy, if not set, will use the
          value of ecutwfc in dftparam
     
     Returns
     -------
     jobs: list[str]
          the job names of the ABACUS calculations
     '''
     # placing the build_abacus_jobs ahead of the check of `abacus_command`,
     # supporting the case that only generate the jy orbitals, without running
     # all the dft calculations
     jobs = build_abacus_jobs(elem=atomspecies[0]['elem'], 
                              rcuts=rcuts, 
                              dftparams=dftparam, 
                              geoms=geoms, 
                              spill_guess=spillguess, 
                              **kwargs)
     
     # then run ABACUS
     abacus_command = compparam.get('abacus_command', 'abacus')
     if abacus_command is None:
          print('abacus command is not found, workflow terminated.', flush=True)
          exit()
     for job in jobs:
         if job_done(job):
             print(f'{job} has been done, skip', flush=True)
             continue
         _ = submit(job, 
                    compparam.get('environment', ''),
                    compparam.get('mpi_command', ''),
                    abacus_command)
     # to find these folders by name, call function
     # in SIAB.io.convention the dft_folder
     # dft_folder(elem, proto, pert, rcut = None)
     return jobs

def _spilltasks(elem, 
                rcuts, 
                scheme, 
                dft_root = '.',
                run_mode = 'jy'):
     '''
     
     Parameters
     ----------
     elem: str
         element symbol
     rcuts: list[float]
         the cutoff radius of orbital to generate
     scheme: list
          the scheme of how to generate the orbitals
     dft_root: str
          the root folder of the dft calculations
     run_mode: str
          the mode to execute spillage optimization, default is jy, also can be pw
     
     Generate
     --------
     rcut: float
          the cutoff radius of the orbitals
     orbitals: list[dict]
          the orbitals to optimize
     '''
     convert_ = {'nzeta': 'nzeta', 'nbands': 'nbnds', 
                 'checkpoint': 'iorb_frozen', 'geoms': 'folders'}
     for rcut in rcuts:
          template = scheme.copy()
          orbitals = [{convert_.get(k, k): v for k, v in orb.items()} for orb in template]
          additional = {} if run_mode != 'jy' else {'rcut': rcut}
          for orb in orbitals:
               geoms_orb = [{'elem': elem, 'proto': f['proto'], 'pert': pertmag} 
                            for f in orb['folders'] 
                            for pertmag in jobfilter(dft_root, elem, f['proto'],
                                                     'stretch', f['pertmags'],
                                                     additional.get('rcut'),
                                                     5, 1.5)]
               orb['folders'] = [dft_folder(**(geom|additional)) for geom in geoms_orb]
          yield rcut, orbitals

def spillage(elem, 
             ecut, 
             rcuts, 
             primitive_type,  
             scheme, 
             dft_root = '.',
             run_mode = 'jy',
             **kwargs):
     '''
     Run the spillage optimization

     Parameters
     ----------
     elem: str
         element symbol
     ecut: float
          the kinetic energy cutoff of the underlying jy
     rcuts: list[float]
          the cutoff radius of orbital to generate
     primitive_type: str
          the type of jy, can be `reduced` or `normalized`
     scheme: list
          the scheme of how to generate the orbitals
     dft_root: str
          the root folder of the dft calculations
     run_mode: str
          the mode to execute spillage optimization, default is jy, also can be pw
     kwargs: dict
          additional parameters, including `max_steps`, `verbose`, `ftol`, `gtol`, `nthreads_rcut`
     '''
     optimizer, options = _spillparam(kwargs)
     for rcut, task in _spilltasks(elem, rcuts, scheme, dft_root, run_mode):
          initializer = {} if run_mode != 'jy' else {'rcut': rcut}
          cascade = GetOrbCascadeInstance(elem=elem, 
                                          rcut=rcut, 
                                          ecut=ecut, 
                                          primitive_type=primitive_type,
                                          initializer=dft_folder(elem, 'monomer', 0, **initializer),
                                          orbs=task,
                                          mode=run_mode,
                                          optimizer=optimizer)
          cascade.opt(immediplot=None,  # immediplot will cause threading bugs from matplotlib
                      diagnosis=True, 
                      options=options, 
                      nthreads=kwargs.get('nthreads_rcut', 1))
          cascade.plot('.')

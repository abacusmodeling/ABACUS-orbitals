'''
Concepts
--------
Interface of ABACUS module for ABACUS-ORBGEN
'''
from SIAB.io.convention import dft_folder, orb
from SIAB.abacus.io import INPUT, STRU, abacus_params
from SIAB.abacus.utils import is_duplicate
from SIAB.spillage.api import _coef_gen, _save_orb
from SIAB.spillage.radial import _nbes
from SIAB.spillage.datparse import read_input_script
from SIAB.io.read_input import natom_from_shape
import os
import unittest
import uuid

def _build_case(proto, 
                pertkind, 
                pertmag, 
                atomspecies,
                dftshared, 
                dftspecific):
    '''build the single dft calculation case

    Parameters
    ----------
    proto : str
        the prototype of the structure
    pertkind : str
        the kind of perturbation
    pertmag : float
        the magnitude of perturbation
    atomspecies : dict
        the atom species in the structure, whose keys are element symbols,
        values are 'pp' and 'orb', indicating the pseudopotential and orbital
        file paths
    dftshared : dict
        the shared dft calculation parameters, will be overwritten by the
        dftspecific
    dftspecific : dict
        the specific dft calculation parameters
    '''
    elem = list(atomspecies.keys())
    if len(elem) != 1:
        raise NotImplementedError(f'only one element is supported: {elem}')
    if pertkind != 'stretch':
        raise NotImplementedError('only stretch structral perturbation is supported')
    
    elem = elem[0]
    rcut = None if dftshared.get('basis_type', 'jy') == 'pw' \
        else dftshared.get('bessel_nao_rcut')
    
    # INPUT
    dftparam = dftshared|dftspecific # overwrite the shared parameters
    dftparam |= {
        'pseudo_dir': os.path.abspath(os.path.dirname(atomspecies[elem].get('pp'))),
        'orbital_dir': os.path.abspath(os.path.dirname(atomspecies[elem].get('orb', '.')))}
    param_ = INPUT(dftparam)

    # STRU
    pp = os.path.basename(atomspecies[elem].get('pp'))
    orb = None if atomspecies[elem].get('orb') is None \
        else os.path.basename(atomspecies[elem].get('orb'))
    pertgeom_, _ = STRU(proto, 
                        elem, 
                        1, 
                        pp, 
                        30,
                        pertmag,
                        dftparam.get('nspin', 1),
                        orb)
    
    folder = dft_folder(elem, proto, pertmag, rcut)
    if is_duplicate(folder, dftparam):
        return None
    
    os.makedirs(folder, exist_ok = True)
    with open(os.path.join(folder, 'INPUT'), 'w') as f:
        f.write(param_)
    with open(os.path.join(folder, 'STRU'), 'w') as f:
        f.write(pertgeom_)
    if dftparam.get('basis_type', 'jy') == 'pw':
        with open(os.path.join(folder, 'INPUTw'), 'w') as f:
            f.write('WANNIER_PARAMETERS\nout_spillage 2\n')
    return folder

def _build_atomspecies(elem,
                       pp,
                       ecut = None,
                       rcut = None,
                       lmaxmax = None,
                       primitive_type = 'reduced',
                       orbital_dir = 'primitive_jy'):
    '''build the atomspecies dictionary. Once ecut, rcut and lmaxmax
    all are provided, will generate the orbital file path
    
    Parameters
    ----------
    elem : str
        the element symbol
    pp : str
        the pseudopotential file path
    ecut : float
        the kinetic energy cutoff in Ry for planewave
    rcut : float
        the cutoff radius in a.u. for primtive jy
    lmaxmax : int
        the maximum angular momentum for generating the orbital file
    primitive_type : str
        the type of primitive basis set
    orbital_dir : str
        where the generated orbital file is stored
    
    Returns
    -------
    dict
        the atomspecies dictionary
    '''
    out = {'pp': os.path.abspath(pp)}
    if all([ecut, rcut, lmaxmax]):
        less_dof = 0 if primitive_type == 'normalized' else 1
        nzeta = [_nbes(l, rcut, ecut) - less_dof for l in range(lmaxmax + 1)]
        forb = os.path.join(orbital_dir, orb(elem, rcut, ecut, nzeta))
        out['orb'] = os.path.abspath(forb)
        if os.path.exists(forb):
            return {elem: out}
        coefs = _coef_gen(rcut, ecut, lmaxmax, primitive_type)[0]
        _ = _save_orb(coefs, elem, ecut, rcut, orbital_dir, primitive_type)
    return {elem: out}

def build_abacus_jobs(elem,
                      rcuts,
                      dftparams,
                      geoms,
                      spill_guess: str = 'atomic'):
    '''build series of abacus jobs
    
    Parameters
    ----------
    globalparams : dict
        the global parameters for orbital generation task
    dftparams : dict
        the dft calculation parameters (shared)
    geoms : list of dict
        the geometries of the structures, also contains specific dftparam
        to overwrite the shared dftparam
    spill_guess : str
        the guess for the spillage optimization
    '''
    jobs = []
    for geom in geoms:
        dftspecific = {k: v for k, v in geom.items() if k in abacus_params()}
        for pertmag in geom['pertmags']:
            if dftparams.get('basis_type', 'lcao') == 'pw':
                as_ = _build_atomspecies(elem, dftparams['pseudo_dir'])
                folder = _build_case(geom['proto'], 
                                     geom['pertkind'], 
                                     pertmag, 
                                     as_,
                                     dftparams|{'bessel_nao_rcut': rcuts},
                                     dftspecific)
                if folder is not None:
                    jobs.append(folder)
            else:
                for rcut in rcuts:
                    as_ = _build_atomspecies(elem,
                                             dftparams['pseudo_dir'],
                                             dftparams['ecutwfc'],
                                             rcut,
                                             geom.get('lmaxmax', 1))
                    folder = _build_case(geom['proto'],
                                         geom['pertkind'],
                                         pertmag,
                                         as_,
                                         dftparams|{'bessel_nao_rcut': rcut},
                                         dftspecific)
                    if folder is not None:
                        jobs.append(folder)
    # then the initial guess
    if spill_guess == 'atomic':
        nbndmax = int(max([
            geom['nbands']/natom_from_shape(geom['proto']) for geom in geoms]))
        if dftparams.get('basis_type', 'lcao') == 'pw':
            as_ = _build_atomspecies(elem, dftparams['pseudo_dir'])
            folder = _build_case('monomer', 
                                 'stretch', 
                                 0,
                                 as_,
                                 dftparams|{'bessel_nao_rcut': rcuts},
                                 {'nbands': nbndmax + 20})
            jobs.append(folder)
        else:
            lmaxmax = int(max([geom.get('lmaxmax', 1) for geom in geoms]))
            for rcut in rcuts:
                as_ = _build_atomspecies(elem,
                                         dftparams['pseudo_dir'],
                                         dftparams['ecutwfc'],
                                         rcut,
                                         lmaxmax)
                folder = _build_case('monomer',
                                     'stretch',
                                     0,
                                     as_,
                                     dftparams|{'bessel_nao_rcut': rcut},
                                     {'nbands': nbndmax + 20})
                jobs.append(folder)
    return [job for job in jobs if job is not None]

def job_done(folder):
    '''check if the abacus calculation is completed
    
    Parameters
    ----------
    folder : str
        the folder of the abacus calculation
    
    Returns
    -------
    bool
        True if the calculation is completed
    '''
    dftparam = read_input_script(os.path.join(folder, 'INPUT'))
    suffix = dftparam.get('suffix', 'ABACUS')
    runtype = dftparam.get('calculation', 'scf')
    outdir = os.path.join(folder, f'OUT.{suffix}')
    runninglog = os.path.join(outdir, f'running_{runtype}.log')
    if os.path.exists(runninglog):
        with open(runninglog, 'r') as f:
            lines = f.readlines()
        for i in range(len(lines) - 1, -1, -1):
            if 'Finish Time' in lines[i]:
                return True
    return False

class TestAbacusApi(unittest.TestCase):
    def test_build_atomspecies(self):
        '''test _build_atomspecies
        '''
        orbital_dir = str(uuid.uuid4())
        atomspecies = _build_atomspecies('H', 'H.pp')
        self.assertEqual(atomspecies, {'H': {'pp': 'H.pp'}})

        atomspecies = _build_atomspecies('H', 'H.upf', 100, 7, 1, 'reduced', orbital_dir)
        self.assertTrue('orb' in atomspecies['H'])
        self.assertTrue(os.path.exists(atomspecies['H']['orb']))
        os.remove(atomspecies['H']['orb'])
        os.system('rm -rf ' + orbital_dir)

    def test_build_case(self):
        '''test _build_case
        '''
        dftparam = {}
        atomspecies = {'H': {'pp': 'pseudo/H.pp'}}
        folder = _build_case('monomer', 'stretch', 0, atomspecies, dftparam, {})
        self.assertTrue(os.path.exists(folder))
        self.assertEqual(os.path.basename(folder), 'H-monomer')
        self.assertTrue(os.path.exists(os.path.join(folder, 'INPUT')))
        self.assertTrue(os.path.exists(os.path.join(folder, 'STRU')))
        os.system('rm -rf ' + folder)

        dftparam = {'basis_type': 'lcao', 'bessel_nao_rcut': 6}
        atomspecies = {'H': {'pp': 'pseudo/H.pp'}}
        folder = _build_case('monomer', 'stretch', 0, atomspecies, dftparam, {})
        self.assertTrue(os.path.exists(folder))
        self.assertTrue(os.path.exists(os.path.join(folder, 'INPUT')))
        self.assertTrue(os.path.exists(os.path.join(folder, 'STRU')))
        self.assertEqual(os.path.basename(folder), 'H-monomer-6au')
        os.system('rm -rf ' + folder)

        dftparam = {'basis_type': 'pw', 'bessel_nao_rcut': 6}
        atomspecies = {'H': {'pp': 'pseudo/H.pp', 'orb': 'orb/H.orb'}}
        folder = _build_case('dimer', 'stretch', 1.1, atomspecies, dftparam, {'lmaxmax': 2})
        self.assertTrue(os.path.exists(folder))
        self.assertTrue(os.path.exists(os.path.join(folder, 'INPUT')))
        self.assertTrue(os.path.exists(os.path.join(folder, 'STRU')))
        self.assertEqual(os.path.basename(folder), 'H-dimer-1.10')
        os.system('rm -rf ' + folder)

        dftparam = {'basis_type': 'pw', 'bessel_nao_rcut': [6, 7]}
        atomspecies = {'H': {'pp': 'pseudo/H.pp', 'orb': 'orb/H.orb'}}
        folder = _build_case('dimer', 'stretch', 1.1, atomspecies, dftparam, {'lmaxmax': 2})
        self.assertTrue(os.path.exists(folder))
        self.assertTrue(os.path.exists(os.path.join(folder, 'INPUT')))
        self.assertTrue(os.path.exists(os.path.join(folder, 'STRU')))
        self.assertEqual(os.path.basename(folder), 'H-dimer-1.10')
        os.system('rm -rf ' + folder)

        dftparam = {'basis_type': 'lcao', 'bessel_nao_rcut': 6}
        atomspecies = {'H': {'pp': 'pseudo/H.pp', 'orb': 'orb/H.orb'}}
        folder = _build_case('dimer', 'stretch', 1.1, atomspecies, dftparam, {'lmaxmax': 2})
        self.assertTrue(os.path.exists(folder))
        self.assertTrue(os.path.exists(os.path.join(folder, 'INPUT')))
        self.assertTrue(os.path.exists(os.path.join(folder, 'STRU')))
        self.assertEqual(os.path.basename(folder), 'H-dimer-1.10-6au')
        os.system('rm -rf ' + folder)

    def test_build_abacus_jobs(self):
        '''test build_abacus_jobs
        '''
        dftparams = {'basis_type': 'pw', 'pseudo_dir': 'pseudo', 'ecutwfc': 100}
        geoms = [{'proto': 'dimer', 'pertkind': 'stretch', 'pertmags': [0.25], 'nbands': 10},
                 {'proto': 'trimer', 'pertkind': 'stretch', 'pertmags': [0.5, 1.0], 'nbands': 20}]
        jobs = build_abacus_jobs('He', [6, 7], dftparams, geoms)
        self.assertEqual(len(jobs), 4) # including the initial guess
        for job in jobs:
            self.assertTrue(os.path.exists(job))
            os.system('rm -rf ' + job)
        self.assertSetEqual(set(jobs), {'He-dimer-0.25', 'He-trimer-0.50', 'He-trimer-1.00', 'He-monomer'})
        
        # test without the initial guess
        jobs = build_abacus_jobs('He', [6, 7], dftparams, geoms, 'random')
        self.assertEqual(len(jobs), 3)
        for job in jobs:
            self.assertTrue(os.path.exists(job))
            os.system('rm -rf ' + job)
        self.assertSetEqual(set(jobs), {'He-dimer-0.25', 'He-trimer-0.50', 'He-trimer-1.00'})

        dftparams = {'basis_type': 'lcao', 'pseudo_dir': 'pseudo', 'ecutwfc': 100, 'bessel_nao_rcut': 8}
        jobs = build_abacus_jobs('He', [6, 7], dftparams, geoms)
        self.assertEqual(len(jobs), 8)
        for job in jobs:
            self.assertTrue(os.path.exists(job))
            os.system('rm -rf ' + job)
        self.assertSetEqual(set(jobs), {'He-dimer-0.25-6au', 'He-dimer-0.25-7au', 
                                        'He-trimer-0.50-6au', 'He-trimer-0.50-7au', 
                                        'He-trimer-1.00-6au', 'He-trimer-1.00-7au', 
                                        'He-monomer-6au', 'He-monomer-7au'})
        # test without the initial guess
        jobs = build_abacus_jobs('He', [6, 7], dftparams, geoms, 'random')
        self.assertEqual(len(jobs), 6)
        for job in jobs:
            self.assertTrue(os.path.exists(job))
            os.system('rm -rf ' + job)
        self.assertSetEqual(set(jobs), {'He-dimer-0.25-6au', 'He-dimer-0.25-7au', 
                                        'He-trimer-0.50-6au', 'He-trimer-0.50-7au', 
                                        'He-trimer-1.00-6au', 'He-trimer-1.00-7au'})

if __name__ == '__main__':
    unittest.main()
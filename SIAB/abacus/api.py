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
from SIAB.io.read_input import natom_from_shape
import os
import unittest
import uuid

def build_case(proto, 
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
        raise NotImplementedError('only one element is supported')
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
    os.makedirs(folder, exist_ok = True)
    if is_duplicate(folder, dftparam):
        return None
    
    with open(os.path.join(folder, 'INPUT'), 'w') as f:
        f.write(param_)
    with open(os.path.join(folder, 'STRU'), 'w') as f:
        f.write(pertgeom_)
    return folder

def build_atomspecies(elem,
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
    out = {'pp': pp}
    if all([ecut, rcut, lmaxmax]):
        less_dof = 0 if primitive_type == 'normalized' else 1
        nzeta = [_nbes(l, rcut, ecut) - less_dof for l in range(lmaxmax + 1)]
        forb = os.path.join(orbital_dir, orb(elem, rcut, ecut, nzeta))
        out['orb'] = forb
        if os.path.exists(forb):
            return out
        coefs = _coef_gen(rcut, ecut, lmaxmax, primitive_type)[0]
        _ = _save_orb(coefs, elem, ecut, rcut, orbital_dir, primitive_type)
    return out

def build_abacus_jobs(globalparams,
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
        dftspecific = {k: v for k, v in geom.items() if k not in abacus_params()}
        for pertmag in geom['pertmags']:
            if dftparams.get('basis_type', 'lcao') == 'pw':
                dftparams.update({'bessel_nao_rcut': globalparams.get('bessel_nao_rcut')})
                as_ = build_atomspecies(globalparams['elem'], dftparams['pseudo_dir'])
                folder = build_case(geom['proto'], 
                                    geom['pertkind'], 
                                    pertmag, 
                                    as_,
                                    dftparams,
                                    dftspecific)
                if folder is not None:
                    jobs.append(folder)
            else:
                for rcut in globalparams.get('bessel_nao_rcut', [6]):
                    dftparams.update({'bessel_nao_rcut': rcut})
                    as_ = build_atomspecies(globalparams['elem'],
                                            dftparams['pseudo_dir'],
                                            dftparams['ecutwfc'],
                                            rcut,
                                            geom.get('lmaxmax', 1))
                    folder = build_case(geom['proto'],
                                        geom['pertkind'],
                                        pertmag,
                                        as_,
                                        dftparams,
                                        dftspecific)
                    if folder is not None:
                        jobs.append(folder)
    # then the initial guess
    if spill_guess == 'atomic':
        nbndmax = int(max([
            geom['nbands']/natom_from_shape(geom['proto']) for geom in geoms]))
        if dftparams.get('basis_type', 'lcao') == 'pw':
            as_ = build_atomspecies(globalparams['elem'], dftparams['pseudo_dir'])
            folder = build_case('monomer', 
                                'stretch', 
                                0,
                                as_,
                                dftparams,
                                {'nbands': nbndmax + 20})
            jobs.append(folder)
        else:
            lmaxmax = int(max([geom.get('lmaxmax', 1) for geom in geoms]))
            for rcut in globalparams.get('bessel_nao_rcut', [6]):
                dftparams.update({'bessel_nao_rcut': rcut})
                as_ = build_atomspecies(globalparams['elem'],
                                        dftparams['pseudo_dir'],
                                        dftparams['ecutwfc'],
                                        rcut,
                                        lmaxmax)
                folder = build_case('monomer',
                                    'stretch',
                                    0,
                                    as_,
                                    dftparams,
                                    {'nbands': nbndmax + 20})
                jobs.append(folder)
    return [job for job in jobs if job is not None]

class TestAbacusApi(unittest.TestCase):
    def test_build_atomspecies(self):
        '''test build_atomspecies
        '''
        orbital_dir = str(uuid.uuid4())
        atomspecies = build_atomspecies('H', 'H.pp')
        self.assertEqual(atomspecies, {'pp': 'H.pp'})

        atomspecies = build_atomspecies('H', 'H.upf', 100, 7, 1, 'reduced', orbital_dir)
        self.assertTrue('orb' in atomspecies)
        self.assertTrue(os.path.exists(atomspecies['orb']))
        os.remove(atomspecies['orb'])
        os.system('rm -rf ' + orbital_dir)

    def test_build_case(self):
        '''test build_case
        '''
        dftparam = {}
        atomspecies = {'H': {'pp': 'pseudo/H.pp'}}
        folder = build_case('monomer', 'stretch', 0, atomspecies, dftparam, {})
        self.assertTrue(os.path.exists(folder))
        os.system('rm -rf ' + folder)

        dftparam = {'basis_type': 'lcao'}
        atomspecies = {'H': {'pp': 'pseudo/H.pp', 'orb': 'orb/H.orb'}}
        folder = build_case('dimer', 'stretch', 1.1, atomspecies, dftparam, {'lmaxmax': 2})
        self.assertTrue(os.path.exists(folder))
        self.assertEqual(os.path.basename(folder), 'H-dimer-1.10')
        os.system('rm -rf ' + folder)

if __name__ == '__main__':
    unittest.main()
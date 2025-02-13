'''this module is to read the input of ABACUS ORBGEN-v3.0 input script'''
import json
import os
from SIAB.abacus.io import abacus_params
import logging
import unittest

def dryrun(params):
    '''check the correctness of the input parameters
    
    Parameters
    ----------
    params : dict
        the input parameters
    '''
    COMPULSORY_ = ['abacus_command', 'pseudo_dir', 'element', 'bessel_nao_rcut',
                   'geoms', 'orbitals']
    for key in COMPULSORY_:
        if key not in params:
            raise ValueError(f'key {key} is missing in the input')
    # check if there is really pseudopotential point to the right directory
    if not os.path.exists(params['pseudo_dir']):
        raise FileNotFoundError(f'pseudo_dir {params["pseudo_dir"]} does not exist')
    logging.info('pseudopotential existence check passed')

    # check if bessel_nao_rcut is list of int
    if not isinstance(params['bessel_nao_rcut'], list):
        raise TypeError('bessel_nao_rcut should be a list of int')
    if not all(isinstance(i, int) for i in params['bessel_nao_rcut']):
        raise TypeError('bessel_nao_rcut should be a list of int')
    logging.info('rcut check passed')

    # check if geom is a list of dict
    if not isinstance(params['geoms'], list):
        raise TypeError('geoms should be a list of dict')
    if not all(isinstance(i, dict) for i in params['geoms']):
        raise TypeError('geoms should be a list of dict')
    logging.info('geom type check passed')

    # check if every geom has at least keys: 'proto', 'pertkind', 'pertmags', 'lmaxmax'
    for i, geom in enumerate(params['geoms']):
        if not all(key in geom for key in ['proto', 'pertkind', 'pertmags', 'lmaxmax']):
            raise ValueError(f'geom {i} does not have all the compulsory keys')
        geomval_chk(geom)
    logging.info('geom key&val check passed')

    # check if orbitals is a list of dict
    if not isinstance(params['orbitals'], list):
        raise TypeError('orbitals should be a list of dict')
    logging.info('orbitals type check passed')

    # check if each orbital has keys: 'nzeta', 'geoms', 'nbands' and 'checkpoint'
    for i, orb in enumerate(params['orbitals']):
        if not all(key in orb for key in ['nzeta', 'geoms', 'nbands', 'checkpoint']):
            raise ValueError(f'orbital {i} does not have all the compulsory keys')
        orbval_chk(orb)
    logging.info('orbital key&val check passed')

    cmprhsive_chk(params)
    logging.info('comprehensive check passed')

    logging.info('dryrun passed')

def orbval_chk(orb):
    '''check the integrity of the orbital parameters
    
    Parameters
    ----------
    orb : dict
        the orbital parameters
    '''
    # check if nzeta is 'auto' or a list of int
    if orb['nzeta'] != 'auto':
        if not isinstance(orb['nzeta'], list):
            raise TypeError('nzeta should be a list of int')
        if not all(isinstance(i, int) for i in orb['nzeta']):
            raise TypeError('nzeta should be a list of int')
    # check if geoms is a list of int (index of geoms defined in geoms section)
    if not isinstance(orb['geoms'], list):
        raise TypeError('geoms should be a list of int')
    if not all(isinstance(i, int) for i in orb['geoms']):
        raise TypeError('geoms should be a list of int')
    # check if nbands is a list of int or str
    if not isinstance(orb['nbands'], (list, int, str)):
        raise TypeError('nbands should be a list, int or str')
    if isinstance(orb['nbands'], list) and not all(isinstance(i, (int, str)) for i in orb['nbands']):
        raise TypeError('if specify nbands as list, it should be list[int], list[str] or list[int, str]')
    # check if checkpoint is a int or None
    if not isinstance(orb['checkpoint'], (int, type(None))):
        raise TypeError('checkpoint should be a int or None')

def geomval_chk(geom):
    '''check the integrity of the geom parameters
    
    Parameters
    ----------
    geom : dict
        the geom parameters
    '''
    # check if pertmags is list of int or float
    if not isinstance(geom['pertmags'], list) and not (isinstance(geom['pertmags'], str)):
        raise TypeError('pertmags should be a list of int or float, or a string')
    if isinstance(geom['pertmags'], list):
        if not all(isinstance(i, (int, float)) for i in geom['pertmags']):
            raise TypeError('pertmags should be a list of int or float')
    # check if lmaxmax is a non-negative int
    if not isinstance(geom['lmaxmax'], (int, str)):
        raise TypeError('lmaxmax should be a int or a string (development use)')
    if isinstance(geom['lmaxmax'], int) and geom['lmaxmax'] < 0:
        raise ValueError('lmaxmax should be a non-negative int')

def cmprhsive_chk(params):
    '''check the correctness of the input parameters across sections
    
    Parameters
    ----------
    params : dict
        the input parameters
    '''
    import numpy as np
    # check if more bands are required in orbital section than in geom section
    for orb in params['orbitals']:
        for i in orb['geoms']:
            if i >= len(params['geoms']):
                raise ValueError(f'orbital {orb} requires geom {i} which is not defined')
            spill_nbnds = orb['nbands'] # can be str, int, list of int
            if isinstance(spill_nbnds, str):
                print(f'orbital {orb} requires `{spill_nbnds}` bands')
                continue
            geom_nbnds = params['geoms'][i].get('nbands')
            if geom_nbnds is None:
                continue
            spill_nbnds = [spill_nbnds] if not isinstance(spill_nbnds, list) else spill_nbnds
            if np.any([nb > geom_nbnds for nb in spill_nbnds]):
                raise ValueError(f'orbital {orb} requires more bands than geom {i} can provide')

def group(params):
    '''parameters defined in input script are for different fields,
    this function is to group them into different sets.
    
    Parameters
    ----------
    params : dict
        the input parameters

    Returns
    -------
    dict, dict, dict, dict
        the global parameters, the DFT parameters, the spillage parameters, 
        the compute parameters
    '''
    GLOBAL = ['element', 'bessel_nao_rcut']
    DFT = [k for k in abacus_params() if k != 'bessel_nao_rcut']
    COMPUTE = ['environment', 'mpi_command', 'abacus_command']
    SPILLAGE = ['fit_basis', 'primitive_type', 'optimizer', 
                'max_steps', 'spill_guess', 'nthreads_rcut', 'geoms', 'orbitals',
                'ecutjy']
    
    dftparams = {key: params.get(key) for key in DFT}
    
    optimizer = {k: v for k, v in params.items() if k.startswith('scipy.') or k.startswith('torch.')}
    spillparams = {key: params.get(key) for key in SPILLAGE if key in params}|optimizer
    spillparams['ecutjy'] = spillparams.get('ecutjy', dftparams.get('ecutwfc'))
    
    glbparams = {key: params.get(key) for key in GLOBAL}
    
    compute = {key: params.get(key) for key in COMPUTE}

    return glbparams, dftparams, spillparams, compute

def read(fn):
    '''read the input of ABACUS ORBGEN-v3.0 input script
    
    Parameters
    ----------
    fn : str
        the filename of the input script
    
    Returns
    -------
    dict, dict, dict, dict
        the global parameters, the DFT parameters, the spillage parameters,
        the compute parameters. For detailed explanation, see group function
    '''
    
    with open(fn) as f:
        params = json.load(f)
    
    try:
        dryrun(params)
    except Exception as e:
        logging.error(f'error in checking the input: {e}')
        raise e
    
    return group(params)

def orb_link_geom(indexes, geoms):
    '''link the indexes of geoms to proper geom parameters
    
    Parameters
    ----------
    indexes : list of int
        the indexes of geoms
    geoms : list of dict
        the geom parameters
    
    Returns
    -------
    list of dict
        the geom parameters
    '''
    return [{k: v for k, v in geoms[i].items() if k in ['proto', 'pertkind', 'pertmags']} 
            for i in indexes]

class TestReadv3p0(unittest.TestCase):

    def test_orbval_chk(self):
        orb = {'nzeta': 'auto', 'geoms': [0, 1], 'nbands': [1, 2], 'checkpoint': 1}
        orbval_chk(orb)
        orb = {'nzeta': [1, 2], 'geoms': [0, 1], 'nbands': [1, 2], 'checkpoint': 1}
        orbval_chk(orb)
        orb = {'nzeta': [1, 2], 'geoms': [0, 1], 'nbands': [1, 2], 'checkpoint': None}
        orbval_chk(orb)
        orb = {'nzeta': 'auto', 'geoms': [0, 1], 'nbands': [1, 2], 'checkpoint': '1'}
        with self.assertRaises(TypeError):
            orbval_chk(orb)
    
    def test_geomval_chk(self):
        geom = {'proto': 'test', 'pertkind': 'test', 'pertmags': [1, 2], 'lmaxmax': 1}
        geomval_chk(geom)
        geom = {'proto': 'test', 'pertkind': 'test', 'pertmags': [1, 2], 'lmaxmax': 0}
        geomval_chk(geom)
        geom = {'proto': 'test', 'pertkind': 'test', 'pertmags': [1, 2], 'lmaxmax': 1.0}
        with self.assertRaises(TypeError):
            geomval_chk(geom)
        geom = {'proto': 'test', 'pertkind': 'test', 'pertmags': [1, 2], 'lmaxmax': -1}
        with self.assertRaises(ValueError):
            geomval_chk(geom)
    
    def test_dryrun(self):
        here = os.path.dirname(__file__)
        parent = os.path.dirname(here)
        grandparent = os.path.dirname(parent)
        pporb = os.path.join(grandparent, 'tests', 'pporb')
        fpseudo = os.path.join(pporb, 'Si_ONCV_PBE-1.0.upf')

        params = {'abacus_command': 'test', 'pseudo_dir': fpseudo, 'element': 'test', 'bessel_nao_rcut': [1, 2],
                  'geoms': [{'proto': 'test', 'pertkind': 'test', 'pertmags': [1, 2], 'lmaxmax': 1}],
                  'orbitals': [{'nzeta': 'auto', 'geoms': [0, 1], 'nbands': [1, 2], 'checkpoint': 1}]}
        dryrun(params)
        params = {'abacus_command': 'test', 'pseudo_dir': 'test', 'element': 'test', 'bessel_nao_rcut': [1, 2],
                  'geoms': [{'proto': 'test', 'pertkind': 'test', 'pertmags': [1, 2], 'lmaxmax': 1}],
                  'orbitals': [{'nzeta': 'auto', 'geoms': [0, 1], 'nbands': [1, 2], 'checkpoint': 1}]}
        with self.assertRaises(FileNotFoundError):
            dryrun(params)
        params = {'abacus_command': 'test', 'pseudo_dir': fpseudo, 'element': 'test', 'bessel_nao_rcut': [1, 2],
                  'geoms': [{'proto': 'test', 'pertkind': 'test', 'pertmags': [1, 2], 'lmaxmax': 1}],
                  'orbitals': [{'nzeta': 'auto', 'geoms': [0, 1], 'nbands': [1, 2], 'checkpoint': 1.0}]}
        with self.assertRaises(TypeError):
            dryrun(params)
        params = {'abacus_command': 'test', 'pseudo_dir': fpseudo, 'element': 'test', 'bessel_nao_rcut': [1, 2],
                  'geoms': [{'proto': 'test', 'pertkind': 'test', 'pertmags': [1, 2], 'lmaxmax': 1}],
                  'orbitals': [{'nzeta': 'auto', 'geoms': [0, 1], 'nbands': [1, 2], 'checkpoint': '1'}]}
        with self.assertRaises(TypeError):
            dryrun(params)
        params = {'abacus_command': 'test', 'pseudo_dir': fpseudo, 'element': 'test', 'bessel_nao_rcut': [1, 2],
                  'geoms': [{'proto': 'test', 'pertkind': 'test', 'pertmags': [1, 2], 'lmaxmax': 1}],
                  'orbitals': [{'nzeta': 'auto', 'geoms': [0, 1], 'nbands': [1, 2], 'checkpoint': None}]}
        dryrun(params)
        params = {'abacus_command': 'test', 'pseudo_dir': fpseudo, 'element': 'test', 'bessel_nao_rcut': [1, 2],
                  'geoms': [{'proto': 'test', 'pertkind': 'test', 'pertmags': [1, 2], 'lmaxmax': 1}],
                  'orbitals': [{'nzeta': 'auto', 'geoms': [0, 1], 'nbands': [1, 2], 'checkpoint': '1'}]}
        with self.assertRaises(TypeError):
            dryrun(params)

if __name__ == '__main__':
    unittest.main()
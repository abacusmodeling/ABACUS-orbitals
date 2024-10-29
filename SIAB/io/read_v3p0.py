'''this module is to read the input of ABACUS ORBGEN-v3.0 input script'''
import json
import os
from SIAB.abacus.io import abacus_params
import logging
import unittest

def drycheck(params):
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

    # check if each orbital has keys: 'nzeta', 'spill_geoms', 'spill_nbnds' and 'checkpoint'
    for i, orb in enumerate(params['orbitals']):
        if not all(key in orb for key in ['nzeta', 'spill_geoms', 'spill_nbnds', 'checkpoint']):
            raise ValueError(f'orbital {i} does not have all the compulsory keys')
        orbval_chk(orb)
    logging.info('orbital key&val check passed')

    logging.info('drycheck passed')

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
    # check if spill_geoms is a list of int (index of geoms defined in geoms section)
    if not isinstance(orb['spill_geoms'], list):
        raise TypeError('spill_geoms should be a list of int')
    if not all(isinstance(i, int) for i in orb['spill_geoms']):
        raise TypeError('spill_geoms should be a list of int')
    # check if spill_nbnds is a list of int or str
    if not isinstance(orb['spill_nbnds'], list):
        raise TypeError('spill_nbnds should be a list of int or str')
    if not all(isinstance(i, (int, str)) for i in orb['spill_nbnds']):
        raise TypeError('spill_nbnds should be a list of int or str')
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
    if not isinstance(geom['pertmags'], list):
        raise TypeError('pertmags should be a list of int or float')
    if not all(isinstance(i, (int, float)) for i in geom['pertmags']):
        raise TypeError('pertmags should be a list of int or float')
    # check if lmaxmax is a non-negative int
    if not isinstance(geom['lmaxmax'], int):
        raise TypeError('lmaxmax should be a int')
    if geom['lmaxmax'] < 0:
        raise ValueError('lmaxmax should be non-negative')

def read_3p0(fn):
    '''read the input of ABACUS ORBGEN-v3.0 input script
    
    Parameters
    ----------
    fn : str
        the filename of the input script
    
    Returns
    -------
    dict
        the input parameters
    '''
    
    with open(fn) as f:
        params = json.load(f)
    
    try:
        drycheck(params)
    except Exception as e:
        logging.error(f'error in checking the input: {e}')
        raise e
    
    return params

class TestReadv3p0(unittest.TestCase):

    def test_orbval_chk(self):
        orb = {'nzeta': 'auto', 'spill_geoms': [0, 1], 'spill_nbnds': [1, 2], 'checkpoint': 1}
        orbval_chk(orb)
        orb = {'nzeta': [1, 2], 'spill_geoms': [0, 1], 'spill_nbnds': [1, 2], 'checkpoint': 1}
        orbval_chk(orb)
        orb = {'nzeta': [1, 2], 'spill_geoms': [0, 1], 'spill_nbnds': [1, 2], 'checkpoint': None}
        orbval_chk(orb)
        orb = {'nzeta': 'auto', 'spill_geoms': [0, 1], 'spill_nbnds': [1, 2], 'checkpoint': '1'}
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
    
    def test_drycheck(self):
        here = os.path.dirname(__file__)
        parent = os.path.dirname(here)
        grandparent = os.path.dirname(parent)
        pporb = os.path.join(grandparent, 'tests', 'pporb')
        fpseudo = os.path.join(pporb, 'Si_ONCV_PBE-1.0.upf')

        params = {'abacus_command': 'test', 'pseudo_dir': fpseudo, 'element': 'test', 'bessel_nao_rcut': [1, 2],
                  'geoms': [{'proto': 'test', 'pertkind': 'test', 'pertmags': [1, 2], 'lmaxmax': 1}],
                  'orbitals': [{'nzeta': 'auto', 'spill_geoms': [0, 1], 'spill_nbnds': [1, 2], 'checkpoint': 1}]}
        drycheck(params)
        params = {'abacus_command': 'test', 'pseudo_dir': 'test', 'element': 'test', 'bessel_nao_rcut': [1, 2],
                  'geoms': [{'proto': 'test', 'pertkind': 'test', 'pertmags': [1, 2], 'lmaxmax': 1}],
                  'orbitals': [{'nzeta': 'auto', 'spill_geoms': [0, 1], 'spill_nbnds': [1, 2], 'checkpoint': 1}]}
        with self.assertRaises(FileNotFoundError):
            drycheck(params)
        params = {'abacus_command': 'test', 'pseudo_dir': fpseudo, 'element': 'test', 'bessel_nao_rcut': [1, 2],
                  'geoms': [{'proto': 'test', 'pertkind': 'test', 'pertmags': [1, 2], 'lmaxmax': 1}],
                  'orbitals': [{'nzeta': 'auto', 'spill_geoms': [0, 1], 'spill_nbnds': [1, 2], 'checkpoint': 1.0}]}
        with self.assertRaises(TypeError):
            drycheck(params)
        params = {'abacus_command': 'test', 'pseudo_dir': fpseudo, 'element': 'test', 'bessel_nao_rcut': [1, 2],
                  'geoms': [{'proto': 'test', 'pertkind': 'test', 'pertmags': [1, 2], 'lmaxmax': 1}],
                  'orbitals': [{'nzeta': 'auto', 'spill_geoms': [0, 1], 'spill_nbnds': [1, 2], 'checkpoint': '1'}]}
        with self.assertRaises(TypeError):
            drycheck(params)
        params = {'abacus_command': 'test', 'pseudo_dir': fpseudo, 'element': 'test', 'bessel_nao_rcut': [1, 2],
                  'geoms': [{'proto': 'test', 'pertkind': 'test', 'pertmags': [1, 2], 'lmaxmax': 1}],
                  'orbitals': [{'nzeta': 'auto', 'spill_geoms': [0, 1], 'spill_nbnds': [1, 2], 'checkpoint': None}]}
        drycheck(params)
        params = {'abacus_command': 'test', 'pseudo_dir': fpseudo, 'element': 'test', 'bessel_nao_rcut': [1, 2],
                  'geoms': [{'proto': 'test', 'pertkind': 'test', 'pertmags': [1, 2], 'lmaxmax': 1}],
                  'orbitals': [{'nzeta': 'auto', 'spill_geoms': [0, 1], 'spill_nbnds': [1, 2], 'checkpoint': '1'}]}
        with self.assertRaises(TypeError):
            drycheck(params)
        

if __name__ == '__main__':
    unittest.main()
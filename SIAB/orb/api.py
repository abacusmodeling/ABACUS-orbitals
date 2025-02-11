from SIAB.orb.orb_jy import OrbitalJY
from SIAB.orb.orb_pw import OrbitalPW
from SIAB.orb.orb import OrbgenCascade
import os
import re

def _sanitychk(elem,
            rcut,
            ecut,
            nzeta,
            mode,
            primitive_type,
            folders,
            nbnds,
            iorb_frozen,
            optimizer):
    '''check the input of GetOrbCascadeInstance'''
    if not isinstance(elem, str):
        raise TypeError('elem should be a str')
    
    if not isinstance(rcut, (int, float)):
        raise TypeError('rcut should be a float or int')
    
    if not isinstance(ecut, (int, float)):
        raise TypeError('ecut should be a float or int')
    
    if not isinstance(nzeta, list):
        raise TypeError('nzeta should be a list.')
    
    if mode == 'pw':
        if not all([isinstance(nz_it, list) for nz_it in nzeta]) and\
           not all([isinstance(nz, int) for nz_it in nzeta for nz in nz_it]):
            raise TypeError('nzeta should be a list of list of int')
    else:
        if not all([isinstance(nz_it, (list, str)) for nz_it in nzeta]) and\
           not all([isinstance(nz, int) for nz_it in nzeta for nz in nz_it if isinstance(nz_it, list)]) and\
           not all([nz == 'auto' for nz_it in nzeta for nz in nz_it if isinstance(nz_it, str)]):
            raise TypeError('nzeta should be a list of list[int] or `auto`')
        
    if not isinstance(primitive_type, str):
        raise TypeError('primitive_type should be a str')
    if primitive_type not in ['reduced', 'normalized']:
        raise ValueError('primitive_type should be either reduced or normalized')
    
    if not (isinstance(folders, list) and\
            all([isinstance(fd_it, list) for fd_it in folders]) and\
            all([isinstance(fd, str) for fd_it in folders for fd in fd_it])):
        raise TypeError('folders should be a list of list of str')
    if not all([os.path.exists(fd) for fd_it in folders for fd in fd_it]):
        raise FileNotFoundError('some folders do not exist')
    
    if not (isinstance(nbnds, list) and\
            all([isinstance(nbnd_it, list) for nbnd_it in nbnds]) and\
            all([isinstance(nbnd, (range, str, int)) for nbnd_it in nbnds for nbnd in nbnd_it])):
        raise TypeError('nbnds should be a list of list of int, range or str')
    
    if not (isinstance(iorb_frozen, list) and\
            all([isinstance(orb_i, int) or orb_i is None for orb_i in iorb_frozen])):
        raise TypeError('iorb_frozen should be a list of int or None')
    
    if not isinstance(mode, str):
        raise TypeError('mode should be a str')
    if mode not in ['jy', 'pw']:
        raise ValueError('mode should be either jy or pw')
    
    if not re.match(r'^torch\..*|scipy\..*', optimizer):
        raise ValueError(f'currently only optimizer implemented under torch and scipy are'\
                         f' supported, got {optimizer}')

def GetOrbCascadeInstance(elem,
                          rcut, 
                          ecut, 
                          primitive_type,
                          initializer, 
                          orbs,
                          mode,
                          optimizer='pytorch.swats'):
    '''build an instance/task for optimizing orbitals in a cascade
    
    Parameters
    ----------
    elem : str
        the element of this cascade of orbitals
    rcut : float
        the cutoff radius of the orbitals
    ecut : float
        the kinetic energy cutoff of the underlying jy
    primitive_type : str
        the type of jy, can be `reduced` or `normalized`
    orbgraph : list[dict]
        the graph of the orbitals, each element is a dict containing
        the information of the orbital, including nzeta, folders, nbnds
        and iorb_frozen, are number of zeta functions for each angular
        momentum, the folders where the orbital optimization will extract
        information, the number of bands to be included in the optimization
        and the index of its inner shell orbital to be frozen.
    mode : str
        the mode of the optimization, can be `jy` or `pw`

    Returns
    -------
    OrbgenCascade
        the instance of the OrbgenCascade
    '''
    if orbs is None:
        raise ValueError('orbgraph should not be None')
    nzeta = [orb.get('nzeta') for orb in orbs]
    folders = [orb.get('folders') for orb in orbs]
    autoset = lambda x, n: x if isinstance(x, list) else [x]*n
    nbnds = [autoset(orb.get('nbnds'), len(folders[i])) for i, orb in enumerate(orbs)]
    iorb_frozen = [orb.get('iorb_frozen') for orb in orbs]

    # do the check
    _sanitychk(elem, rcut, ecut, nzeta, mode, primitive_type, folders, nbnds, iorb_frozen, optimizer)
    
    # describe the orbital
    GetOrbitalInstance = OrbitalJY if mode == 'jy' else OrbitalPW
    orbs = [GetOrbitalInstance(rcut=rcut,                     # cutoff radius of orbital to gen
                               ecut=ecut,                     # kinetic energy cutoff
                               elem=elem,                     # a element symbol
                               nzeta=nz,                      # number of zeta functions of each l
                               primitive_type=primitive_type, # `reduce` or `normalized`
                               folders=fd,                    # folders in which reference data is stored
                               nbnds=nbnd)                    # number of bands to refer
            for nz, fd, nbnd in zip(nzeta, folders, nbnds)]
    
    # plug orbitals in optimization cascade
    return OrbgenCascade(initializer=initializer, 
                         orbitals=orbs, 
                         ifrozen=iorb_frozen, 
                         mode=mode, 
                         optimizer=optimizer)

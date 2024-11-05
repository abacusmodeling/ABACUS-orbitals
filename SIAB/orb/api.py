from SIAB.orb.orb_jy import OrbitalJY
from SIAB.orb.orb_pw import OrbitalPW
from SIAB.orb.orb import OrbgenCascade
import os

def orb_cascade(elem,
                rcut, 
                ecut, 
                primitive_type,
                initializer, 
                orbs,
                mode,
                optimizer='pytorch.swats'):
    '''build an OrbgenCascade based on one orbgen graph/scheme
    
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
    '''
    if orbs is None:
        raise ValueError('orbgraph should not be None')
    nzeta = [orb.get('nzeta') for orb in orbs]
    folders = [orb.get('folders') for orb in orbs]
    nbnds = [orb.get('nbnds') for orb in orbs]
    iorb_frozen = [orb.get('iorb_frozen') for orb in orbs]
    
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
    impl, optimizer = optimizer.split('.')
    if impl not in ['pytorch', 'scipy']:
        raise ValueError('optimizer should be either pytorch or scipy')
    if mode == 'jy':
        orbs = [OrbitalJY(rcut, ecut, elem, nz, primitive_type, fd, nbnd) 
                for nz, fd, nbnd in zip(nzeta, folders, nbnds)]
    else:
        orbs = [OrbitalPW(rcut, ecut, elem, nz, primitive_type, fd, nbnd) 
                for nz, fd, nbnd in zip(nzeta, folders, nbnds)]
        
    return OrbgenCascade(initializer, orbs, iorb_frozen, mode, impl)

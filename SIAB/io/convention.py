'''
this module controls the naming convention of files and folders
of ABACUS-ORBGEN. There is also another use of this module, that
is, the mapping from parameters to specific folders.
'''
def dft_folder(elem, geom, pert, rcut = None):
    '''return the folder name for a dft calculation
    
    Parameters
    ----------
    elem: str
        element symbol
    geom: str
        geometry of the structure
    pert: str
        perturbation of the structure
    rcut: float
        cutoff radius

    Returns
    -------
    str
        the folder name
    '''
    rcut = str(rcut) if rcut is not None else None
    words = [elem, geom, f'{pert:.2f}', rcut]
    words = [str(w) for w in words if w is not None]
    return '-'.join(words)

def orb_folder(elem, nzeta):
    '''return the folder name for an orbital calculation
    
    Parameters
    ----------
    elem: str
        element symbol
    nzeta: list of int
        number of zeta for each angular momentum
    
    Returns
    -------
    str
        the folder name
    '''
    SPECTRUM = 'spdfghijklmnopqrstuvwxyz'
    conf = ''
    for nz, sym in zip(nzeta, SPECTRUM):
        if nz > 0:
            conf += sym + str(nz)
    return '_'.join([str(w) for w in [elem, conf]])

def orb(elem, rcut, ecut, nzeta, xc = 'gga'):
    '''return the orbital file name
    
    Parameters
    ----------
    elem: str
        element symbol
    rcut: float
        cutoff radius
    ecut: float
        energy cutoff
    nzeta: list of int
        number of zeta for each angular momentum
    xc: str
        exchange-correlation functional, default is gga
    '''
    SPECTRUM = 'spdfghijklmnopqrstuvwxyz'
    conf = ''
    for nz, sym in zip(nzeta, SPECTRUM):
        if nz > 0:
            conf += sym + str(nz)
    return '_'.join([elem, xc, str(rcut)+'au', str(ecut)+'Ry', conf]) + '.orb'
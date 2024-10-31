'''
Concepts
--------
After calling DFT to calculate several quantities, the viewpoint
now fully transforms to the question "how to deal with an orbital".

On the other hand, 
the parameter list of Spillage.opt function defines a semi-mathematical
problem that:
def opt(self, coef_init, coef_frozen, iconfs, ibands,
        options, nthreads=1)
the first and the second parameters are purely initial guess and a fixed
component, which defines purely a mathematical problem of Spillage function
optimization. The lattering two are Spillage function specific, they
will be those terms building the Spillage function. The last two are
optimization options.

Classes defined here are for connecting between the (semi-)purely
mathematical problem and the physical (maybe?) problem, say the instance
of orbital.
'''
from SIAB.spillage.util import _spil_bnd_autoset
from SIAB.spillage.spillage import Spillage_pw, Spillage_jy
from SIAB.spillage.listmanip import merge, nestpat
from SIAB.spillage.api import _save_orb, _coef_subset
from SIAB.spillage.datparse import read_input_script
from SIAB.spillage.radial import _nbes
from SIAB.io.convention import dft_folder
import os
import numpy as np
import unittest

class Orbital:
    '''the orbital here corresponds to a set of coefficients of jy'''
    coef_ = None

    def __init__(self, 
                 rcut, 
                 ecut, 
                 elem, 
                 nzeta,
                 primitive_type,
                 folders, # should it be included when defining an orb?
                 nbnds):  # then how about this?
        '''instantiate an orbital object, in this function the nzeta is
        calculated
        
        Parameters
        ----------
        rcut : float
            the cutoff radius of the orbital
        ecut : float
            the kinetic energy cutoff of the underlying jy
        elem : str
            the element of the orbital
        nzeta : list[int]|list[str]|None
            the number of zeta for each angular momentum
        primitive_type : str
            the type of jy, can be `reduced` or `normalized`. The latter
            is highly not recommended.
        folders : list[str]
            the folders where orbital optimization will extract information
        nbnds : list[range]|list[str]
            the number of bands to be included in the optimization. Besides
            the normal range, it can also be `occ` or `all`
        '''
        if isinstance(rcut, (int, float)):
            self.rcut_ = rcut
        else:
            raise TypeError('rcut should be a float or int')
        if isinstance(ecut, (int, float)):
            self.ecut_ = ecut
        else:
            raise TypeError('ecut should be a float or int')
        if isinstance(elem, str):
            self.elem_ = elem
        else:
            raise TypeError('elem should be a str')
        
        self.nzeta_ = nzeta # needed when initialize the coef

        if isinstance(primitive_type, str):
            self.primitive_type_ = primitive_type
        else:
            raise TypeError('primitive_type should be a str')

        if isinstance(folders, list) and all([isinstance(f, str) for f in folders]):
            for f in folders:
                if not os.path.exists(f):
                    raise FileNotFoundError(f'{f} does not exist')
        else:
            raise TypeError('folders should be a list of str')
        self.folders_ = folders

        self.nbnds_ = nbnds
        # infer the nbands_ref if it is a string
        if len(nbnds) != len(folders):
            raise ValueError('Mismatch of nbnds and folders when instantiating Orbital')
        self.nbnds_ = [range(_spil_bnd_autoset(nb, f)) for nb, f in zip(nbnds, self.folders_)]

    def init(self, srcdir, nzshift, diagnosis = True):
        '''calculate the initial value of the contraction coefficient of jy,
        crucial in further optimization tasks
        
        Parameters
        ----------
        srcdir : str
            the directory where the single atomic calculation data
            is stored
        nzshift : list[int]|None
            the shift of number of zeta for each angular momentum
        diagnosis : bool
            diagnose the purity of the initial guess
        '''
        if srcdir is None:
            print('WARNING: initializing orbital with random coefficients')
            less_dof = 0 if self.primitive_type_ == 'normalized' else 1
            coefs_rnd = [np.random.random((nz, _nbes(l, self.rcut_, self.ecut_) - less_dof)).tolist()
                         for l, nz in enumerate(self.nzeta_)]
            return _coef_subset(self.nzeta_, nzshift, coefs_rnd)[0] # [l][iz][q]
        if not os.path.exists(srcdir):
            raise FileNotFoundError(f'{srcdir} does not exist')
        if nzshift is not None and not isinstance(nzshift, list):
            raise TypeError('nzshift should be a list of int or None')
        if not isinstance(diagnosis, bool):
            raise TypeError('diagnosis should be a bool')
        return None

    def coef(self):
        '''return the contraction coefficient of jy'''
        if not self.coef_:
            raise ValueError('coef not initialized')
        return self.coef_
    
    def __eq__(self, value):
        '''compare two orbitals'''
        if not isinstance(value, Orbital):
            return False
        return self.rcut_ == value.rcut_ and\
            self.ecut_ == value.ecut_ and\
            self.elem_ == value.elem_ and\
            self.nzeta_ == value.nzeta_ and\
            self.primitive_type_ == value.primitive_type_ and\
            self.folders_ == value.folders_ and\
            self.nbnds_ == value.nbnds_ and\
            self.coef_ == value.coef_

    def __ne__(self, value):
        '''compare two orbitals'''
        return not self.__eq__(value)

class OrbgenCascade:
    '''
    Concepts
    --------
    The OrbgenCascade is a runner for the optimization of orbitals 
    in manner of onion, let orbitals form a cascade. The optimization 
    of the outer shell will be based on the inner shell
    '''

    minimizer_ = None
    initializer_ = None
    def __init__(self,
                 initializer: str|None,
                 orbitals: list[Orbital],
                 ifrozen: list,
                 mode: str = 'jy'):
        '''instantiation of the an orbital cascade
        
        Parameters
        ----------
        initializer : str
            the folder where the initial guess of the coefficients are stored
        orbitals : list[Orbital]
            the list of orbitals forming the cascade
        ifrozen : list[int|None]
            the index of its inner shell orbital to be frozen. If None, it
            means a fully optimization
        mode : str
            the mode of the optimization, can be `jy` or `pw`
        '''
        if initializer is None:
            pass # if initializer to be none, then use random for all orbitals
        else:
            if not isinstance(initializer, str):
                raise TypeError('initializer should be a str')
            if not os.path.exists(initializer):
                raise FileNotFoundError(f'{initializer} does not exist')
        self.initializer_ = initializer

        rcuts = list(set([orb.rcut_ for orb in orbitals]))
        if len(rcuts) != 1:
            raise ValueError('rcut should be the same for all orbitals')
        self.orbitals_ = orbitals
        
        if not all([i is None or isinstance(i, int) for i in ifrozen]):
            raise TypeError('ifrozen should be a list of int or None')
        self.ifrozen_ = ifrozen
        
        # unique folders...avoid one folder being imported for multiple times
        uniqfds = list(set([f for orb in orbitals for f in orb.folders_]))
        # then index of the unique folders
        self.iuniqfds_ = [[uniqfds.index(f) for f in orb.folders_] for orb in self.orbitals_]

        if mode not in ['jy', 'pw']:
            raise ValueError('mode should be either jy or pw')
        if mode == 'jy':
            self.minimizer_ = Spillage_jy()
            for f in uniqfds:
                suffix = read_input_script(os.path.join(f, 'INPUT')).get('suffix', 'ABACUS')
                self.minimizer_.config_add(os.path.join(f, f'OUT.{suffix}'))
                print(f'OrbgenCascade: a new term added: {f} -> Generalized Spillage S = sum <ref|(1-P)|ref>')
        else:
            self.minimizer_ = Spillage_pw()
            OLD_MATRIX_PW = {'orb_matrix_0': 'orb_matrix.0.dat',
                             'orb_matrix_1': 'orb_matrix.1.dat'}
            NEW_MATRIX_PW = {'orb_matrix_0': f'orb_matrix_rcut{rcuts[0]}deriv0.dat',
                             'orb_matrix_1': f'orb_matrix_rcut{rcuts[0]}deriv1.dat'}
            for f in uniqfds:
                use_old = all([os.path.exists(os.path.join(f, v)) for v in OLD_MATRIX_PW.values()])
                use_new = all([os.path.exists(os.path.join(f, v)) for v in NEW_MATRIX_PW.values()])
                if use_old and use_new:
                    raise ValueError(f'{f} contains both the old and new matrix file')
                if not use_old and not use_new:
                    raise FileNotFoundError(f'{f} does not contain the matrix file')
                fmat = OLD_MATRIX_PW if use_old else NEW_MATRIX_PW
                fmat = {k: os.path.join(f, v) for k, v in fmat.items()}
                self.minimizer_.config_add(**fmat)
                print(f'OrbgenCascade: a new term added: {f} -> Generalized Spillage S = sum <ref|(1-P)|ref>')
                
    def opt(self,
            immediplot = None, 
            diagnosis = True,
            options = None, 
            nthreads = 1):
        '''optimize the cascade of orbitals
        
        Parameters
        ----------
        immediplot : str|None
            where to store the plot immediately on the fly, if None, no plot will be stored
        diagnosis : bool
            whether to diagnose the purity of the initial guess
        options : dict|None
            the options for the minimizer
        nthreads : int
            the number of threads to use
        
        Returns
        -------
        list[str]: the list of the file names of the plot
        '''
        if self.minimizer_ is None:
            raise ValueError('Spillage instance not initialized')
        
        out = []
        nzmax = [np.array(orb.nzeta_) for orb in self.orbitals_]
        lmaxmax = np.max([len(nz) for nz in nzmax])
        nzmax = [np.pad(nz, (0, lmaxmax - len(nz)), 'constant') for nz in nzmax]
        nzmax = np.max(nzmax, axis = 0).tolist()

        print(f'OrbgenCascade: start optimizing orbitals in cascade')
        for orb, ifroz, iconfs in zip(self.orbitals_, self.ifrozen_, self.iuniqfds_):
            orb_frozen = self.orbitals_[ifroz] if ifroz is not None else None
            coefs_frozen = [orb_frozen.coef_] if orb_frozen else None
            
            # only initialize when necessary
            nzshift = None if orb_frozen is None else orb_frozen.nzeta_
            orb.init(self.initializer_, nzshift, diagnosis) if isinstance(self.minimizer_, Spillage_jy)\
                else orb.init(self.initializer_, nzmax, nzshift, diagnosis)
            # then optimize
            coefs_shell, spillage = self.minimizer_.opt([orb.coef_], 
                                                         coefs_frozen, 
                                                         iconfs, 
                                                         orb.nbnds_, 
                                                         options, 
                                                         nthreads)
            print(f'OrbgenCascade: orbital optimization ends with spillage = {spillage}', flush=True)
            orb.coef_ = merge(coefs_frozen, coefs_shell, 2)[0] if coefs_frozen else coefs_shell[0]
            if immediplot:
                f = _save_orb(orb.coef_, orb.elem_, orb.ecut_, orb.rcut_, immediplot)
                out.append(f)
        return out
    
    def plot(self, outdir):
        '''plot the orbitals
        
        Parameters
        ----------
        outdir : str
            the folder to store the plot
        
        Returns
        -------
        list[str]: the list of the file names of the plot
        '''
        out = []
        for orb in self.orbitals_:
            f = _save_orb(orb.coef_, orb.elem_, orb.ecut_, orb.rcut_, outdir)
            out.append(f)
        return out

'''
Concept
-------
orbgraph
    the graph expressing the relationship between orbitals. The initial one
    will always be an initializer (but can also leave as None, then will 
    use purely random number to initialize orbitals in each opt run), then
    with the connection (dependency) between orbitals, the optimizer can
    optimize the orbitals in a cascade manner.
'''
def build_orbgraph(elem,
                   rcut,
                   ecut,
                   primitive_type,
                   mode,
                   scheme, 
                   folders):
    '''build an orbgraph based on the scheme set by user.
    
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
    mode : str
        the mode of the optimization, can be `jy` or `pw`
    scheme : list[dict]
        the scheme of the orbitals, each element is a dict containing
        the information of the orbital, including nzeta, folders, nbnds
        and iorb_frozen, are number of zeta functions for each angular
        momentum, the folders where the orbital optimization will extract
        information, the number of bands to be included in the optimization
        and the index of its inner shell orbital to be frozen.
    folders : list[str]
        the folders where orbital optimization will extract information
    
    Returns
    -------
    dict: the orbgraph
    '''
    out = {'elem': elem, 'rcut': rcut, 'ecut': ecut, 'primitive_type': primitive_type, 
           'mode': mode, 'initializer': None, 'orbs': []}
    # this function is responsible for arranging orbitals so that there wont
    # be the case that one orbital refers to another that is not optimized
    # yet.
    if scheme.get('spill_guess') == 'atomic':
        _r = None if mode == 'pw' else rcut
        out['initializer'] = dft_folder(elem, 'monomer', None, _r)
    for orb in scheme['orbs']: # for each orbital...
        orb_ref = orb.get('checkpoint')
        orb_ref = orb_ref if orb_ref != 'none' or orb_ref is not None else None
        out['orbs'].append({
            'nzeta': orb['nzeta'],
            'nbnds': orb['nbands'],
            'iorb_frozen': orb_ref,
            'folders': [folders[i] for i in orb['geoms']]
        })
    return out
    
class TestOrbital(unittest.TestCase):
    def test_instantiate(self):
        here = os.path.dirname(__file__)
        # testfiles in another folder
        parent = os.path.dirname(here)
        outdir = os.path.join(parent, 'spillage/testfiles/Si/jy-7au/monomer-gamma/')

        orb = Orbital(7, 
                      100, 
                      'Si', 
                      [1, 1, 0], 
                      'reduced', 
                      [outdir], 
                      [4])
        self.assertEqual(orb.nzeta_, [1, 1, 0])
        self.assertEqual(orb.rcut_, 7)
        self.assertEqual(orb.ecut_, 100)
        self.assertEqual(orb.elem_, 'Si')

if __name__ == "__main__":
    unittest.main()

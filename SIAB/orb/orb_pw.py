'''
Brief
-----
derived class of Orbital for handling specifically the pw as fit_basis case
'''
import os
from SIAB.orb.orb import Orbital, OrbgenCascade
from SIAB.spillage.spillage import initgen_pw
from SIAB.spillage.api import _coef_subset
from SIAB.spillage.datparse import read_input_script
import unittest

class OrbitalPW(Orbital):
    '''the derived class of Orbital for handling specifically the pw
    as fit_basis case'''
    coef_ = None # [l][iz][q]

    def __init__(self, 
                 rcut, 
                 ecut, 
                 elem, 
                 nzeta,
                 primitive_type,
                 folders,
                 nbnds):
        '''specifically for fit_basis jy, it is possible to do things
        additionally compared with the pw:
        
        nzeta-infer:
            inferring the nzeta from the nbnds
        '''
        super().__init__(rcut, ecut, elem, nzeta, primitive_type, folders, nbnds)
        # for OrbitalPW, the nzeta-infer is not supported

    def init(self, srcdir, nzmax, nzshift, diagnosis = True):
        '''initialize the contraction coefficients of jy. This function works differently
        from the other derived class of Orbital, the OrbitalJY. Each time will extract the
        full set of initial guess of coefs, then extract different subset.
        
        Parameters
        ----------
        srcdir: str
            the directory containing the matrix file. Unlike OrbitalJY, the orb_matrix
            file will be in the jobdir, instead of outdir
        nzmax: list[int]
            the maximum number of zeta for each angular momentum
        nzshift: list[int]
            the starting index from which initial guess of coefs will be extracted
        diagnosis: bool
            whether to print the diagnostic information
        '''
        # use it to do param check, plus if srcdir is None, init with random coefs
        coefs_default = super().init(srcdir, nzshift, diagnosis) 
        if coefs_default is not None:
            self.coef_ = coefs_default
            return
        
        OLD_MATRIX = 'orb_matrix.0.dat'
        NEW_MATRIX = f'orb_matrix_rcut{self.rcut_}deriv0.dat'

        if not any([os.path.exists(os.path.join(srcdir, f))\
                    for f in [OLD_MATRIX, NEW_MATRIX]]):
            raise FileNotFoundError(f'{srcdir} does not contain the matrix file')
        if all([os.path.exists(os.path.join(srcdir, f))\
                for f in [OLD_MATRIX, NEW_MATRIX]]):
            raise ValueError(f'{srcdir} contains both the old and new matrix file')
        
        fmat = OLD_MATRIX if os.path.exists(os.path.join(srcdir, OLD_MATRIX)) else NEW_MATRIX
        self.coef_ = _coef_subset(self.nzeta_, 
                                  nzshift, 
                                  initgen_pw(os.path.join(srcdir, fmat), nzmax))[0]

class TestOrbitalPW(unittest.TestCase):

    def test_instantiate(self):
        here = os.path.dirname(__file__)
        # testfiles in another folder
        parent = os.path.dirname(here)
        outdir = os.path.join(parent, 'spillage/testfiles/Si/pw/monomer-gamma/')

        orb = OrbitalPW(7, 
                        100, 
                        'Si', 
                        [1, 1, 0], 
                        'reduced', 
                        [outdir], 
                        [4])
        self.assertEqual(orb.nzeta_, [1, 1, 0])

    def test_opt_initrnd(self):
        here = os.path.dirname(__file__)
        # testfiles in another folder
        parent = os.path.dirname(here)
        outdir = os.path.join(parent, 'spillage/testfiles/Si/pw/monomer-gamma/')

        orb = OrbitalPW(7, 
                        40, 
                        'Si', 
                        [1, 1, 0], 
                        'reduced', 
                        [outdir], 
                        [4])
        options = {"maxiter": 10, "disp": False, "ftol": 0, "gtol": 1e-6, 'maxcor': 20}

        cascade = OrbgenCascade(None,
                                [orb],
                                [None],
                                'pw')
        forbs = cascade.opt(diagnosis=True, options=options)
        self.assertEqual(len(forbs), 0) # no orbital will be saved

    def test_opt_initatomic(self):
        here = os.path.dirname(__file__)
        # testfiles in another folder
        parent = os.path.dirname(here)
        outdir = os.path.join(parent, 'spillage/testfiles/Si/pw/monomer-gamma/')

        orb = OrbitalPW(7, 
                        40, 
                        'Si', 
                        [1, 1, 0], 
                        'reduced', 
                        [outdir], 
                        [4])
        options = {"maxiter": 10, "disp": False, "ftol": 0, "gtol": 1e-6, 'maxcor': 20}

        initializer = outdir
        cascade = OrbgenCascade(initializer,
                                [orb],
                                [None],
                                'pw')
        forbs = cascade.opt(diagnosis=True, options=options)
        self.assertEqual(len(forbs), 0) # no orbital will be saved

if __name__ == "__main__":
    unittest.main()

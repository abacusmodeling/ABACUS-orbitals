'''
Brief
-----
the derived class of Orbital for handling specifically the jy as fit_basis case
'''

from SIAB.orb.orb import Orbital, OrbgenCascade
from SIAB.spillage.legacy.api import _nzeta_mean_conf
from SIAB.orb.jy_expmt import _coef_init
from SIAB.spillage.datparse import read_input_script
import numpy as np
import unittest
import os

class OrbitalJY(Orbital):
    
    coef_ = None

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
        self.nzeta_ = [self.nzeta_] if isinstance(self.nzeta_, list) else 1.0
        self.nzeta_ = _nzeta_mean_conf([nbnd.stop for nbnd in self.nbnds_], 
                                       self.folders_, 
                                       'max',
                                       'svd-fold', 
                                       self.nzeta_)
        self.nzeta_ = [int(np.ceil(nz)) for nz in self.nzeta_] # make sure it is integer

    def init(self, srcdir, nzshift, diagnosis = True):
        coefs_default = super().init(srcdir, nzshift, diagnosis)
        if coefs_default is not None:
            self.coef_ = coefs_default
        else:
            base = os.path.basename(srcdir)
            if 'OUT.' not in base:
                dftparam = read_input_script(os.path.join(srcdir, 'INPUT'))
                suffix = dftparam.get('suffix', 'ABACUS')
                srcdir = os.path.join(srcdir, f'OUT.{suffix}')
            self.coef_ = _coef_init(srcdir, self.nzeta_, nzshift, diagnosis = diagnosis)

class TestOrbitalJY(unittest.TestCase):
    def test_instantiate(self):
        here = os.path.dirname(__file__)
        # testfiles in another folder
        parent = os.path.dirname(here)
        outdir = os.path.join(parent, 'spillage/testfiles/Si/jy-7au/monomer-gamma/')

        orb = OrbitalJY(rcut=7, 
                        ecut=100, 
                        elem='Si', 
                        nzeta=[1, 1, 0], 
                        primitive_type='reduced', 
                        folders=[outdir], 
                        nbnds=[4])
        self.assertEqual(orb.nzeta_, [1, 1, 0])
    
    def test_init(self):
        here = os.path.dirname(__file__)
        # testfiles in another folder
        parent = os.path.dirname(here)
        outdir = os.path.join(parent, 'spillage/testfiles/Si/jy-7au/monomer-gamma/')

        orb = OrbitalJY(7, 
                        100, 
                        'Si', 
                        [1, 1, 0], 
                        'reduced', 
                        [outdir], 
                        [4])
        suffix = read_input_script(os.path.join(outdir, 'INPUT')).get('suffix', 'ABACUS')
        orb.init(os.path.join(outdir, f'OUT.{suffix}'), None, diagnosis = True)
        self.assertEqual(len(orb.coef_), 3) # up to l = 2
        
    def test_opt_initrnd(self):
        here = os.path.dirname(__file__)
        # testfiles in another folder
        parent = os.path.dirname(here)
        outdir = os.path.join(parent, 'spillage/testfiles/Si/jy-7au/monomer-gamma/')

        orb = OrbitalJY(7, 
                        100, 
                        'Si', 
                        [1, 1, 0], 
                        'reduced', 
                        [outdir], 
                        [4])
        options = {"maxiter": 10, "disp": False, "ftol": 0, "gtol": 1e-6, 'maxcor': 20}

        cascade = OrbgenCascade(None,
                                [orb],
                                [None],
                                'jy')
        forbs = cascade.opt(diagnosis=True, options=options)
        self.assertEqual(len(forbs), 0) # no orbital will be saved

    def test_opt_initatomic(self):
        here = os.path.dirname(__file__)
        # testfiles in another folder
        parent = os.path.dirname(here)
        outdir = os.path.join(parent, 'spillage/testfiles/Si/jy-7au/monomer-gamma/')

        orb = OrbitalJY(7, 
                        100, 
                        'Si', 
                        [1, 1, 0], 
                        'reduced', 
                        [outdir], 
                        [4])
        options = {"maxiter": 10, "disp": False, "ftol": 0, "gtol": 1e-6, 'maxcor': 20}

        suffix = read_input_script(os.path.join(outdir, 'INPUT')).get('suffix', 'ABACUS')
        initializer = os.path.join(outdir, f'OUT.{suffix}')
        cascade = OrbgenCascade(initializer,
                                [orb],
                                [None],
                                'jy')
        forbs = cascade.opt(diagnosis=True, options=options)
        self.assertEqual(len(forbs), 0) # no orbital will be saved


if __name__ == '__main__':
    unittest.main()

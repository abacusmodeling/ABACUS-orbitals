from SIAB.spillage.index import _lin2comp
import os
import numpy as np
import scipy.linalg as la

def _wll(C, S, natom, nzeta):
    '''
    Band-wise angular momentum analysis of wave function coefficients.

    Parameters
    ----------
        C : 2D array of shape (nao, nbands)
            Wave function coefficients in LCAO basis. The datatype is
            complex for multi-k calculations and float for gamma-only.
        S : 2D array of shape (nao, nao)
            Overlap matrix.
        natom : list of int
            Number of atoms for each type.
        nzeta : list of list of int
            nzeta[i][l] specifies the number of zeta orbitals of the
            angular momentum l of type i. len(nzeta) must equal len(natom).

    Returns
    -------
        wll : 3D array of shape (nbands, lmax+1, lmax+1)
            np.sum(wll[ib]) should be 1.0 for all ib.
            To get a "weight" of l for each band, one may sum over wll[ib]
            along either the row or the column axis, i.e., np.sum(wll[ib], 0)

    '''
    nao, nbands = C.shape
    lmax = max(len(nz) for nz in nzeta) - 1
    lin2comp = _lin2comp(natom, nzeta)

    # idx[l] gives a list of linearized indices whose basis functions have
    # angular momentum l.
    idx = [[] for _ in range(lmax+1)]
    for i, (_, _, l, _, _) in enumerate(lin2comp):
        idx[l].append(i)

    wll = np.zeros((nbands, lmax+1, lmax+1), dtype=C.dtype)
    for ib in range(nbands):
        for lr, idx_lr in enumerate(idx):
            for lc, idx_lc in enumerate(idx):
                wll[ib, lr, lc] = C[idx_lr, ib].conj().T \
                        @ S[idx_lr][:, idx_lc] @ C[idx_lc, ib]

    return wll

def _wfc_reinterp(C, nbands, natom, nzeta, pop_view = 'reduce'):
    '''reinterpret wavefunction coefficients in different view, rearrange
    and concatenate the wavefunction coefficients of all bands into a 
    matrix
    
    Parameters
    ----------
        C : 2D array of shape (nao, nbands)
            Wave function coefficients in LCAO basis. The datatype is
            complex for multi-k calculations and float for gamma-only.
        nbands : int|str
            Number of bands selected for the analysis, 'all' for all bands.
        natom : list of int
            Number of atoms for each type.
        nzeta : list of list of int
            nzeta[i][l] specifies the number of zeta orbitals of the
            angular momentum l of type i. len(nzeta) must equal len(natom).
        pop_view : str
            Method to concatenate the wave function coefficients. Options are
            'decompose' and 'reduce'. decompose: concatentate the C matrix of 
            atoms in direction of basis. reduce: concatenate the C matrix of 
            atoms in direction of bands.
    '''

    ntyp = len(nzeta)

    assert pop_view in ['decompose', 'reduce']

    nao, nbands_max = C.shape
    nbands = nbands_max if nbands == 'all' else nbands
    assert nbands <= nbands_max, 'nbands selected is larger than the total nbands'

    lin2comp = _lin2comp(natom, nzeta)
    comp2lin = sorted([((it, l, m, iz, ia), i)
                        for i, (it, ia, l, iz, m) in enumerate(lin2comp)])
    mu = 0
    Ct = [[[] for _ in nzeta[it]] for it in range(ntyp)]
    while mu < nao:
        it, l, _, _, _ = comp2lin[mu][0]
        stride = natom[it] * nzeta[it][l]
        idx = [comp2lin[mu+i][1] for i in range(stride)]
        tlm_shape = (-1, nbands) if pop_view == 'decompose' else (nzeta[it][l], -1)
        Ct[it][l].append(C[idx, :nbands].reshape(tlm_shape))
        mu += stride

    return Ct

def _svdlz(C, 
           S,
           nbands,
           natom, 
           nzeta, 
           l_isotrop = 'rotational-invariant',
           reinterp_view = 'reduce'):
    '''perform svd on the wave function coefficients, return the
    singular value of zeta function of each atomtype each l, which 
    represents the weight.
    
    Parameters
    ----------
        C : 2D array of shape (nao, nbands)
            Wave function coefficients in LCAO basis. The datatype is
            complex for multi-k calculations and float for gamma-only.
        S : 2D array of shape (nao, nao)
            Overlap matrix.
        nbands : int
            Number of bands selected for the analysis.
        natom : list of int
            Number of atoms for each type.
        nzeta : list of list of int
            nzeta[i][l] specifies the number of zeta orbitals of the
            angular momentum l of type i. len(nzeta) must equal len(natom).
        l_isotrop : str
            Method to average over m for each l (remove anisotropicity). 
            Options are 'rotational-invariant' and 'max'.
        reinterp_view : str
            Method to reinterpret the wave function coefficients. Options are
            'decompose' and 'reduce'.
    
    Returns
    -------
        sigma : list of list of float
            Singular values for each atomtype and each l, each zeta function.
    '''
    
    nao, nbands_max = C.shape
    nbands = nbands_max if nbands == 'all' else nbands
    assert nbands <= nbands_max, 'nbands selected is larger than the total nbands'
    assert reinterp_view in ['decompose', 'reduce']

    # orthogonalize the wave function coefficients
    L = la.cholesky(S, lower=True)
    C = la.solve_triangular(L, C, lower=True)

    lmax = [len(nz) - 1 for nz in nzeta]
    ntyp = len(natom)

    #############################################################################
    # concatenate the wavefunction coefficients together of the same atomtype,  #
    # angular momentum and magnetic quantum number into a matrix, then perform  #
    # reshape operation to either (nat*nz, nbnd) or (nz, nat*nbnd)              #
    #############################################################################
    lin2comp = _lin2comp(natom, nzeta)
    comp2lin = sorted([((it, l, m, iz, ia), i)
                        for i, (it, ia, l, iz, m) in enumerate(lin2comp)])
    mu = 0
    Ct = [[[] for _ in nzeta[it]] for it in range(ntyp)]
    while mu < nao:
        it, l, _, _, _ = comp2lin[mu][0]
        stride = natom[it] * nzeta[it][l]
        idx = [comp2lin[mu+i][1] for i in range(stride)]
        tlm_shape = (-1, nbands) if reinterp_view == 'decompose' else (nzeta[it][l], -1)
        Ct[it][l].append(C[idx, :nbands].reshape(tlm_shape))
        mu += stride
    # return Ct

    # perform SVD on the wave function coefficients, evaluate significance of
    # zeta functions by singular values
    sigma_tlm = [[[] for _ in range(len(nzeta[it]))] for it in range(ntyp)]
    for it in range(ntyp):
        for l in range(lmax[it]+1):
            pref = 1 if l_isotrop == 'max' else 1 / np.sqrt(2*l+1)
            for m in range(2*l+1):
                sigma_tlm[it][l].append(la.svd(Ct[it][l][m], compute_uv=False) * pref)
    
    # average over m
    ord = np.inf if l_isotrop == 'max' else 2
    norm_op = {'axis': 0, 'ord': ord}
    return [[np.linalg.norm(sigma_tlm[it][l], **norm_op)
            for l in range(len(nzeta[it]))] for it in range(ntyp)]

############################################################
#                           Test
############################################################
import unittest

from SIAB.spillage.datparse import read_wfc_lcao_txt, read_triu, \
        read_running_scf_log

class TestLCAOWfcAnalysis(unittest.TestCase):

    def test_wll_gamma(self):
        
        here = os.path.dirname(os.path.abspath(__file__))
        outdir = os.path.join(here, 'testfiles/Si/jy-7au/monomer-gamma/OUT.ABACUS/')
        # outdir = './testfiles/Si/jy-7au/monomer-gamma/OUT.ABACUS/'
        
        wfc = read_wfc_lcao_txt(outdir + 'WFC_NAO_GAMMA1.txt')[0]
        S = read_triu(outdir + 'data-0-S')
        dat = read_running_scf_log(outdir + 'running_scf.log')

        wll = _wll(wfc, S, dat['natom'], dat['nzeta'])

        for ib, wb in enumerate(wll):
            self.assertAlmostEqual(np.sum(wb.real), 1.0, places=6)

        return # suppress output

        for ib, wb in enumerate(wll):
            wl_row_sum = np.sum(wb.real, 0)
            print(f"Band {ib+1}    ", end='')
            for x in wl_row_sum:
                print(f"{x:6.3f} ", end='')
            print(f"    sum = {np.sum(wl_row_sum):6.3f}")
            print('')

    def test_wfc_reinterp(self):
            # nbands
        C = [[ 1,  2,  3], # 1s 
             [ 4,  5,  6], # 2s
             [ 7,  8,  9], # 1px
             [10, 11, 12], # 1py
             [13, 14, 15], # 1pz
             [16, 17, 18], # 2px
             [19, 20, 21], # 2py
             [22, 23, 24], # 2pz
             [25, 26, 27], # 1s
             [28, 29, 30], # 2s
             [31, 32, 33], # 1px
             [34, 35, 36], # 1py
             [37, 38, 39], # 1pz
             [40, 41, 42], # 2px
             [43, 44, 45], # 2py
             [46, 47, 48]] # 2pz
        Cref = [
            [ # it = 0
                [ # l = 0
                    [[ 1,  2,  3], [25, 26, 27], [ 4,  5,  6], [28, 29, 30]]
                ],
                [ # l = 1
                    [[13, 14, 15], [37, 38, 39], [22, 23, 24], [46, 47, 48]],
                    [[ 7,  8,  9], [31, 32, 33], [16, 17, 18], [40, 41, 42]],
                    [[10, 11, 12], [34, 35, 36], [19, 20, 21], [43, 44, 45]],
                ]
            ]
        ]
        C = np.array(C)
        # the C_tlm will be reformed to C[t][l][m], where t is the atom type
        # the matrix indexed will be those coefficients correspondint to orb
        # with the same t, l and m.
        # for decompose, the dimension will be (natom[it]*nzeta[it][l], nbands)
        nao, nbands = C.shape # 16, 3
        natom = [2]
        nzeta = [[2, 2]]
        Ct = _wfc_reinterp(C, 'all', natom, nzeta, 'decompose')
        self.assertEqual(len(Ct), 1) # 1 type
        self.assertEqual(len(Ct[0]), 2) # 2 angular momenta
        self.assertEqual(len(Ct[0][0]), 1) # 1 m for s
        self.assertEqual(len(Ct[0][1]), 3) # 3 m for p
        self.assertEqual(Ct[0][0][0].shape, (4, 3)) # 2atom*2zeta, 3 bands
        for i in range(3):
            self.assertTrue(Ct[0][1][i].shape, (4, 3)) # 2atom*2zeta, 3 bands
        for it in range(1):
            for l in range(2):
                for m in range(2*l+1):
                    #print(Ct[it][l][m].tolist(), Cref[it][l][m])
                    self.assertTrue(np.allclose(Ct[it][l][m].tolist(), Cref[it][l][m]))

        # for reduce, the dimension will be (nzeta[it][l], natom[it]*nbands)
        Ct = _wfc_reinterp(C, 'all', natom, nzeta, 'reduce')
        Cref = [
            [ # it = 0
                [ # l = 0
                    [[ 1,  2,  3, 25, 26, 27], 
                     [ 4,  5,  6, 28, 29, 30]]
                ],
                [
                    [[13, 14, 15, 37, 38, 39],
                     [22, 23, 24, 46, 47, 48]],
                    [[ 7,  8,  9, 31, 32, 33],
                     [16, 17, 18, 40, 41, 42]],
                    [[10, 11, 12, 34, 35, 36],
                     [19, 20, 21, 43, 44, 45]]
                ]
                
            ]
        ]
        self.assertEqual(len(Ct), 1) # 1 type
        self.assertEqual(len(Ct[0]), 2) # 2 angular momenta
        self.assertEqual(len(Ct[0][0]), 1) # 1 m for s
        self.assertEqual(len(Ct[0][1]), 3) # 3 m for p
        self.assertEqual(Ct[0][0][0].shape, (2, 6)) # 2zeta, 2atom*3bands
        for i in range(3):
            self.assertEqual(Ct[0][1][i].shape, (2, 6)) # 2zeta, 2atom*3bands
        for it in range(1):
            for l in range(2):
                for m in range(2*l+1):
                    #print(Ct[it][l][m].tolist(), Cref[it][l][m])
                    self.assertTrue(np.allclose(Ct[it][l][m].tolist(), Cref[it][l][m]))
        
    def test_svdlz_rotinv_decomp(self):
        
        here = os.path.dirname(os.path.abspath(__file__))
        outdir = os.path.join(here, 'testfiles/Si/jy-7au/monomer-gamma/OUT.ABACUS/')

        wfc = read_wfc_lcao_txt(outdir + 'WFC_NAO_GAMMA1.txt')[0]
        S = read_triu(outdir + 'data-0-S')
        dat = read_running_scf_log(outdir + 'running_scf.log')

        sigma = _svdlz(wfc, S, 'all', 
                       dat['natom'], 
                       dat['nzeta'], 
                       'max',
                       'decompose')
        self.assertEqual(len(sigma), len(dat['natom'])) # number of atom types
        for i, (nt, nz) in enumerate(zip(dat['natom'], dat['nzeta'])):
            self.assertEqual(len(sigma[i]), len(nz)) # number of l orbitals
        
        return # suppress output
        for i, (nt, nz) in enumerate(zip(dat['natom'], dat['nzeta'])):
            print(f"Atom type {i+1}")
            for l, s in enumerate(sigma[i]):
                print(f"l = {l}")
                for ix, x in enumerate(s):
                    print(f"{x:6.3f} ", end='')
                    if ix % 5 == 4 and ix != len(s) - 1:
                        print('')
                print('')
                print(f'sum = {np.sum(s):6.3f}\n')
            print('')

    def test_svdlz_max_reduce(self):
        
        here = os.path.dirname(os.path.abspath(__file__))
        outdir = os.path.join(here, 'testfiles/Si/jy-7au/monomer-gamma/OUT.ABACUS/')

        wfc = read_wfc_lcao_txt(outdir + 'WFC_NAO_GAMMA1.txt')[0]
        S = read_triu(outdir + 'data-0-S')
        dat = read_running_scf_log(outdir + 'running_scf.log')

        sigma = _svdlz(wfc, S, 'all', 
                       dat['natom'], 
                       dat['nzeta'], 
                       'max',
                       'reduce')
        self.assertEqual(len(sigma), len(dat['natom'])) # number of atom types
        for i, (nt, nz) in enumerate(zip(dat['natom'], dat['nzeta'])):
            self.assertEqual(len(sigma[i]), len(nz)) # number of l orbitals
        
        return # suppress output
        for i, (nt, nz) in enumerate(zip(dat['natom'], dat['nzeta'])):
            print(f"Atom type {i+1}")
            for l, s in enumerate(sigma[i]):
                print(f"l = {l}")
                for ix, x in enumerate(s):
                    print(f"{x:6.3f} ", end='')
                    if ix % 5 == 4 and ix != len(s) - 1:
                        print('')
                print('')
                print(f'sum = {np.sum(s):6.3f}\n')
            print('')

    def test_wll_multi_k(self):
        
        here = os.path.dirname(os.path.abspath(__file__))
        outdir = os.path.join(here, 'testfiles/Si/jy-7au/dimer-2.8-k/OUT.ABACUS/')
        #outdir = './testfiles/Si/jy-7au/dimer-2.8-k/OUT.ABACUS/'

        wfc = read_wfc_lcao_txt(outdir + 'WFC_NAO_K6.txt')[0]
        S = read_triu(outdir + 'data-5-S')
        dat = read_running_scf_log(outdir + 'running_scf.log')

        wll = _wll(wfc, S, dat['natom'], dat['nzeta'])

        for ib, wb in enumerate(wll):
            self.assertAlmostEqual(np.sum(wb.real), 1.0, places=6)

        return # suppress output

        for ib, wb in enumerate(wll):
            wl_row_sum = np.sum(wb.real, 1)
            print(f"Band {ib+1}    ", end='')
            for x in wl_row_sum:
                print(f"{x:6.3f} ", end='')
            print(f"    sum = {np.sum(wl_row_sum):6.3f}")
            print('')

if __name__ == '__main__':
    unittest.main()

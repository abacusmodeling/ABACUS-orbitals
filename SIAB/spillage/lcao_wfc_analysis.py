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

def _svd_aniso_max(C, S, nband, natom, nzeta):
    '''perform svd-based analysis on wfc with anisotropicity of m for each l,
    based on taking the maximal singular value of m for each l, returning the
    nzeta for each type for each l.'''

    nao, nbands_max = C.shape
    nbands = nbands_max if nband == 'all' else nband
    assert nbands <= nbands_max, 'nbands selected is larger than the total nbands'

    C = la.sqrtm(S) @ C

    lmax = [len(nz) - 1 for nz in nzeta]
    ntyp = len(natom)

    lin2comp = _lin2comp(natom, nzeta)
    comp2lin = sorted([((it, l, m, iz, ia), i)
                        for i, (it, ia, l, iz, m) in enumerate(lin2comp)])
    mu = 0
    Ct = [[[] for _ in nzeta[it]] for it in range(ntyp)]
    while mu < nao:
        it, l, _, _, _ = comp2lin[mu][0]
        stride = natom[it] * nzeta[it][l]
        idx = [comp2lin[mu+i][1] for i in range(stride)]
        Ct[it][l].append(C[idx, :nbands].reshape(nzeta[it][l], -1))
        mu += stride
    
    sigma_tlm = [[[] for _ in range(len(nzeta[it]))] for it in range(ntyp)]
    for it in range(ntyp):
        for l in range(lmax[it]+1):
            for m in range(2*l+1):
                sigma_tlm[it][l].append(la.svd(Ct[it][l][m], compute_uv=False))

    return [[np.linalg.norm(sigma_tlm[it][l], axis=0, ord=np.inf)
             for l in range(len(nzeta[it]))] for it in range(ntyp)]

def _svd_aniso_svd(C, S, nband, natom, nzeta):
    '''perform svd-based analysis on wfc with anisotropicity of m for each l,
    based on svd on m for each l, returning the nzeta for each type for each l.'''

    nao, nbands_max = C.shape
    nbands = nbands_max if nband == 'all' else nband
    assert nbands <= nbands_max, 'nbands selected is larger than the total nbands'

    C = la.sqrtm(S) @ C

    lmax = [len(nz) - 1 for nz in nzeta]
    ntyp = len(natom)

    lin2comp = _lin2comp(natom, nzeta)
    comp2lin = sorted([((it, l, iz, m, ia), i)
                        for i, (it, ia, l, iz, m) in enumerate(lin2comp)])
    mu = 0
    Ct = [[[] for _ in nzeta[it]] for it in range(ntyp)]
    while mu < nao:
        it, l, _, _, _ = comp2lin[mu][0]
        stride = natom[it] * nzeta[it][l] * (2*l+1) # consider all m of present l
        idx = [comp2lin[mu+i][1] for i in range(stride)]
        Ct[it][l] = C[idx, :nbands].reshape(nzeta[it][l], -1) # (nz, nat*nbnd*2l+1)
        mu += stride
    
    sigma_tlm = [[[] for _ in range(len(nzeta[it]))] for it in range(ntyp)]
    for it in range(ntyp):
        for l in range(lmax[it]+1):
            sigma_tlm[it][l] = la.svd(Ct[it][l], compute_uv=False)
    return sigma_tlm

def _svd_iso(C, S, nband, natom, nzeta):
    '''perform svd-based analysis on wfc with isotropicity of m for each l,
    based on averaging over all m for each l, returning the nzeta for each
    type for each l.'''

    nao, nbands_max = C.shape
    nbands = nbands_max if nband == 'all' else nband
    assert nbands <= nbands_max, 'nbands selected is larger than the total nbands'

    C = la.sqrtm(S) @ C

    lmax = [len(nz) - 1 for nz in nzeta]
    ntyp = len(natom)

    lin2comp = _lin2comp(natom, nzeta)
    comp2lin = sorted([((it, l, m, iz, ia), i)
                        for i, (it, ia, l, iz, m) in enumerate(lin2comp)])
    mu = 0
    Ct = [[[] for _ in nzeta[it]] for it in range(ntyp)]
    while mu < nao:
        it, l, _, _, _ = comp2lin[mu][0]
        stride = natom[it] * nzeta[it][l]
        idx = [comp2lin[mu+i][1] for i in range(stride)]
        Ct[it][l].append(C[idx, :nbands].reshape(nzeta[it][l], -1))
        mu += stride
    
    sigma_tlm = [[[] for _ in range(len(nzeta[it]))] for it in range(ntyp)]
    for it in range(ntyp):
        for l in range(lmax[it]+1):
            for m in range(2*l+1):
                sigma_tlm[it][l].append(la.svd(Ct[it][l][m], compute_uv=False) / np.sqrt(2*l+1))

    return [[np.linalg.norm(sigma_tlm[it][l], axis=0, ord=2)
             for l in range(len(nzeta[it]))] for it in range(ntyp)]

def _svd_atomic(C, S, nband, natom, nzeta):
    '''perform svd-based analysis to get the significance of each zeta func
    for each atom. Then '''

    nao, nbands_max = C.shape
    nbands = nbands_max if nband == 'all' else nband
    assert nbands <= nbands_max, 'nbands selected is larger than the total nbands'

    C = la.sqrtm(S) @ C

    lmax = [len(nz) - 1 for nz in nzeta]
    ntyp = len(natom)

    lin2comp = _lin2comp(natom, nzeta)
    comp2lin = sorted([((it, l, ia, m, iz), i)
                        for i, (it, ia, l, iz, m) in enumerate(lin2comp)])
    mu = 0
    Ct = [[[] for _ in nzeta[it]] for it in range(ntyp)]
    while mu < nao:
        it, l, m, ia, iz = comp2lin[mu][0]
        stride = natom[it] * nzeta[it][l] * (2*l+1)
        idx = [comp2lin[mu+i][1] for i in range(stride)]
        Ct[it][l] = C[idx, :nbands].reshape(natom[it], 2*l+1, nzeta[it][l], -1)
        mu += stride
    # then we only svd on the last two dimensions (nzeta[it][l], nbands)
    # for m, we use rotational-invariant method to average over m
    out = [[] for it in range(ntyp)]
    for it in range(ntyp):
        for l in range(lmax[it]+1):
            s_tl = [0] * nzeta[it][l]
            for ia in range(natom[it]):
                s_tla = np.linalg.norm(
                    [la.svd(Ct[it][l][ia, m], compute_uv=False) for m in range(2*l+1)], 
                    axis=0, ord=2) / np.sqrt(2*l+1)
                # only when this atom has sig >= 0.9, we consider it significant
                s_tla = np.where(s_tla >= 0.9, s_tla, 0)
                s_tla = np.concatenate([s_tla, [0] * (nzeta[it][l] - len(s_tla))])
                s_tl = [max(s_tl[i], s_tla[i]) for i in range(nzeta[it][l])]
            out[it].append(s_tl)
    return out

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
            Options are 'rotational-invariant', 'max' and 'aniso'. 
            'rotational-invariant' averages over m by taking abs of each m, 
            then summing over all their squares, then multplying by 1/sqrt(2l+1).
            'max' only extracts the maximum value of m. 
            'aniso' does not average over m, but only get the largest 
            contribution.
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

    # coderabbit.ai recommends the Cholesky decomposition with triangular_solve
    # but the result is not correct at all.
    C = la.sqrtm(S) @ C

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
        svdblk_shape = (-1, nbands) if reinterp_view == 'decompose' else (nzeta[it][l], -1)

        stride = natom[it] * nzeta[it][l]
        idx = [comp2lin[mu+i][1] for i in range(stride)]
        Ct[it][l].append(C[idx, :nbands].reshape(svdblk_shape))
        mu += stride
    # return Ct

    # perform SVD on the wave function coefficients, evaluate significance of
    # zeta functions by singular values
    sigma_tlm = [[[] for _ in range(len(nzeta[it]))] for it in range(ntyp)]
    for it in range(ntyp):
        for l in range(lmax[it]+1):
            if l_isotrop == 'aniso':
                sigma_tlm[it][l] = [
                    la.svd(np.array(Ct[it][l]).reshape(nzeta[it][l], -1), compute_uv=False)]
                continue
            # otherwise, average over m
            pref = 1 if l_isotrop in ['max', 'aniso'] else 1 / np.sqrt(2*l+1)
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

    def est_wll_gamma(self):
        
        #here = os.path.dirname(os.path.abspath(__file__))
        #outdir = os.path.join(here, 'testfiles/Si/jy-7au/monomer-gamma/OUT.ABACUS/')
        outdir = '/root/documents/simulation/orbgen/Test1Aluminum-20241011/Al-dimer-2.00-10au/OUT.Al-dimer-2.00-10au/'
        
        wfc = read_wfc_lcao_txt(outdir + 'WFC_NAO_GAMMA1.txt')[0]
        S = read_triu(outdir + 'data-0-S')
        dat = read_running_scf_log(outdir + 'running_scf.log')

        wll = _wll(wfc, S, dat['natom'], dat['nzeta'])

        for ib, wb in enumerate(wll):
            self.assertAlmostEqual(np.sum(wb.real), 1.0, places=6)

        # return # suppress output

        for ib, wb in enumerate(wll):
            wl_row_sum = np.sum(wb.real, 0)
            print(f"Band {ib+1}    ", end='')
            for x in wl_row_sum:
                print(f"{x:6.3f} ", end='')
            print(f"    sum = {np.sum(wl_row_sum):6.3f}")
            print('')

    def est_wfc_reinterp(self):
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
        
    def est_svdlz_rotinv_decomp(self):
        
        #here = os.path.dirname(os.path.abspath(__file__))
        #outdir = os.path.join(here, 'testfiles/Si/jy-7au/monomer-gamma/OUT.ABACUS/')
        outdir = '/root/documents/simulation/orbgen/Test1Aluminum-20241011/Al-dimer-2.00-10au/OUT.Al-dimer-2.00-10au/'

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

    def est_svdlz_max_reduce(self):
        
        # here = os.path.dirname(os.path.abspath(__file__))
        # outdir = os.path.join(here, 'testfiles/Si/jy-7au/monomer-k/OUT.ABACUS/')
        outdir = '/root/documents/simulation/orbgen/Test1Aluminum-20241011/Al-dimer-2.00-10au/OUT.Al-dimer-2.00-10au/'
        # outdir = 'testfiles/Si/jy-7au/monomer-k/OUT.ABACUS/'

        wfc = read_wfc_lcao_txt(outdir + 'WFC_NAO_GAMMA1.txt')[0]
        S = read_triu(outdir + 'data-0-S')
        dat = read_running_scf_log(outdir + 'running_scf.log')

        sigma = _svdlz(wfc, S, 22, 
                       dat['natom'], 
                       dat['nzeta'], 
                       'max',
                       'reduce')
        self.assertEqual(len(sigma), len(dat['natom'])) # number of atom types
        for i, (nt, nz) in enumerate(zip(dat['natom'], dat['nzeta'])):
            self.assertEqual(len(sigma[i]), len(nz)) # number of l orbitals
        
        # return # suppress output
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

    def est_wll_multi_k(self):
        
        here = os.path.dirname(os.path.abspath(__file__))
        outdir = os.path.join(here, 'testfiles/Si/jy-7au/monomer-k/OUT.ABACUS/')
        #outdir = './testfiles/Si/jy-7au/dimer-2.8-k/OUT.ABACUS/'

        wfc = read_wfc_lcao_txt(outdir + 'WFC_NAO_K2.txt')[0]
        S = read_triu(outdir + 'data-1-S')
        dat = read_running_scf_log(outdir + 'running_scf.log')

        wll = _wll(wfc, S, dat['natom'], dat['nzeta'])

        for ib, wb in enumerate(wll):
            self.assertAlmostEqual(np.sum(wb.real), 1.0, places=6)

        # return # suppress output

        for ib, wb in enumerate(wll):
            wl_row_sum = np.sum(wb.real, 1)
            print(f"Band {ib+1}    ", end='')
            for x in wl_row_sum:
                print(f"{x:6.3f} ", end='')
            print(f"    sum = {np.sum(wl_row_sum):6.3f}")
            print('')

    def est_nzeta_nband_plt(self):
        
        outdir = '/root/documents/simulation/orbgen/Test1Aluminum-20241011/Al-dimer-2.00-10au/OUT.Al-dimer-2.00-10au/'

        wfc = read_wfc_lcao_txt(outdir + 'WFC_NAO_GAMMA1.txt')[0]
        S = read_triu(outdir + 'data-0-S')
        dat = read_running_scf_log(outdir + 'running_scf.log')

        nao, nbands = wfc.shape

        import matplotlib.pyplot as plt
        sigmas = []
        fig, ax = plt.subplots(1, 3, figsize=(18, 6))
        for nbnd in range(4, nbands, 5):
            sigma = _svdlz(wfc, S, nbnd, dat['natom'], dat['nzeta'], 'max', 'reduce')
            for i, (nt, nz) in enumerate(zip(dat['natom'], dat['nzeta'])):
                for l in range(len(nz)):
                    #print(f"atom type {i+1}, l={l}, nbnd={nbnd}")
                    #print(sigma[i][l])
                    ax[l].plot(np.log10(sigma[i][l]), '-o', label=f'nbnd={nbnd}')
                    # ax[l].set_yscale('log')
                    thrs = [1.0, 0.5, 0.1, 0.01]
                    for thr in thrs:
                        ax[l].axhline(np.log10(thr), color='k', linestyle='--')
                    ax[l].legend()
                    ax[l].set_xlim(0, 12)
                    ax[l].set_ylim(-3, 0.5)
                    ax[l].set_title(f'atom type {i+1}, l={l}')
                    ax[l].set_xlabel('zeta functions')
                    ax[l].set_ylabel('log10(sigma)')

        plt.show()
        #plt.savefig('test_single_case.png')
        #plt.close()
    
    def est_svd_atomic(self):
        here = os.path.dirname(os.path.abspath(__file__))
        outdir = os.path.join(here, 'testfiles/Si/jy-7au/monomer-k/OUT.ABACUS/')

        wfc = read_wfc_lcao_txt(outdir + 'WFC_NAO_K2.txt')[0]
        S = read_triu(outdir + 'data-1-S')
        dat = read_running_scf_log(outdir + 'running_scf.log')

        sigma = _svd_atomic(wfc, S, 'all', dat['natom'], dat['nzeta'])
        self.assertEqual(len(sigma), len(dat['natom'])) # number of atom types
        for i, (nt, nz) in enumerate(zip(dat['natom'], dat['nzeta'])):
            self.assertEqual(len(sigma[i]), len(nz)) # number of l orbitals
    
    def est_svd_aniso_max(self):
        here = os.path.dirname(os.path.abspath(__file__))
        outdir = os.path.join(here, 'testfiles/Si/jy-7au/monomer-gamma/OUT.ABACUS/')

        wfc = read_wfc_lcao_txt(outdir + 'WFC_NAO_GAMMA1.txt')[0]
        S = read_triu(outdir + 'data-0-S')
        dat = read_running_scf_log(outdir + 'running_scf.log')

        sigma = _svd_aniso_max(wfc, S, 'all', dat['natom'], dat['nzeta'])
        print(sigma)
        self.assertEqual(len(sigma), len(dat['natom']))
        for i, (nt, nz) in enumerate(zip(dat['natom'], dat['nzeta'])):
            self.assertEqual(len(sigma[i]), len(nz))
    
    def test_svd_aniso_svd(self):
        # here = os.path.dirname(os.path.abspath(__file__))
        # outdir = os.path.join(here, 'testfiles/Si/jy-7au/monomer-gamma/OUT.ABACUS/')
        outdir = outdir = '/root/documents/simulation/orbgen/Test1Aluminum-20241011/Al-dimer-2.00-10au/OUT.Al-dimer-2.00-10au/'
        wfc = read_wfc_lcao_txt(outdir + 'WFC_NAO_GAMMA1.txt')[0]
        S = read_triu(outdir + 'data-0-S')
        dat = read_running_scf_log(outdir + 'running_scf.log')

        sigma = _svd_aniso_svd(wfc, S, 12, dat['natom'], dat['nzeta'])
        for sigma_t in sigma:
            for l, sigma_l in enumerate(sigma_t):
                print(f'l = {l}')
                print(sigma_l**2)

if __name__ == '__main__':
    unittest.main()

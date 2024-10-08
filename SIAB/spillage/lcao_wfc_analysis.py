from SIAB.spillage.index import _lin2comp

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

def _mean_rotinv(a):
    '''average over m for each l with rotational invariance:
    out = sqrt{sum_m |C_m|^2 * 4pi/(2l+1)}
    '''
    l = (a.shape[0] - 1) // 2
    out = np.sum(np.abs(a)**2, axis=0) / (2*l + 1)
    return np.sqrt(out)

def _mean_max(a):
    '''average over m for each l with maximum:
    out = max_m |C_m|
    '''
    return np.max(np.abs(a), axis=0)

def _wfc_interp(C, nbands, natom, nzeta, view = 'reduce'):
    '''interpret wavefunction coefficients in different view, rearrange
    and concatenate the wavefunction coefficients of all bands into a 
    matrix
    
    Parameters
    ----------
        C : 2D array of shape (nao, nbands)
            Wave function coefficients in LCAO basis. The datatype is
            complex for multi-k calculations and float for gamma-only.
        natom : list of int
            Number of atoms for each type.
        nzeta : list of list of int
            nzeta[i][l] specifies the number of zeta orbitals of the
            angular momentum l of type i. len(nzeta) must equal len(natom).
        method : str
            Method to concatenate the wave function coefficients. Options are
            'decompose' and 'reduce'. decompose: concatentate the C matrix of 
            atoms in direction of basis. reduce: concatenate the C matrix of 
            atoms in direction of bands.
    '''
    assert view in ['decompose', 'reduce']

    nao, nbands_max = C.shape
    assert nbands <= nbands_max, 'nbands selected is larger than the total nbands'

    ntyp = len(nzeta)
    if view == 'decompose':
        Ct = [[np.zeros(shape=(2*l+1, natom[it]*nz, nbands)) # nz = nzeta[it][l]
               for l, nz in enumerate(nzeta[it])] for it in range(ntyp)]
        for i, (it, ia, l, iz, m) in enumerate(_lin2comp(natom, nzeta)):
            iaiz = ia*nzeta[it][l] + iz
            for ib in range(nbands):
                Ct[it][l][m, iaiz, ib] = C[i, ib]
    else:
        Ct = [[np.zeros(shape=(2*l+1, nz, natom[it]*nbands)) # nz = nzeta[it][l]
                for l, nz in enumerate(nzeta[it])] for it in range(ntyp)]
        for i, (it, ia, l, iz, m) in enumerate(_lin2comp(natom, nzeta)):
            for ib in range(nbands):
                iaib = ia*nbands + ib
                Ct[it][l][m, iz, iaib] = C[i, ib]
    return Ct

def _svd_on_wfc(C, 
                S,
                nbands,
                natom, 
                nzeta, 
                fold_m = 'rotational-invariant',
                svd_view = 'reduce'):
    '''perform svd on the wave function coefficients, return the
    singular value for each atomtype and each l, each zeta function
    
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
        fold_m : str
            Method to average over m for each l. Options are
            'rotational-invariant' and 'max'.
        svd_view : str
            Method to concatenate the wave function coefficients. Options are
            'decompose' and 'reduce'.
    
    Returns
    -------
        sigma : list of list of float
            Singular values for each atomtype and each l, each zeta function.
    '''
    mean_map = {'rotational-invariant': _mean_rotinv,
                'max': _mean_max}

    C = la.sqrtm(S) @ C

    lmax = [len(nz) - 1 for nz in nzeta]

    ntyp = len(natom)
    Ct = _wfc_interp(C, nbands, natom, nzeta, svd_view)

    mat_tlm = [[np.zeros(shape=(2*l+1, nz)) # nz = nzeta[it][l] 
                for l, nz in enumerate(nzeta[it])] for it in range(ntyp)]
    for it in range(ntyp):
        for l in range(lmax[it]+1):
            for m in range(2*l+1):
                mat_tlm[it][l][m] = la.svd(Ct[it][l][m], compute_uv=False)
    
    mean = mean_map[fold_m]
    out = [[mean(mat_tlm[it][l]) for l in range(len(nzeta[it]))] for it in range(ntyp)]
    
    return out

############################################################
#                           Test
############################################################
import unittest

from SIAB.spillage.datparse import read_wfc_lcao_txt, read_triu, \
        read_running_scf_log

class TestLCAOWfcAnalysis(unittest.TestCase):

    def test_wll_gamma(self):
        import os
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

    def test_svd_on_wfc(self):
        import os
        here = os.path.dirname(os.path.abspath(__file__))
        outdir = os.path.join(here, 'testfiles/Si/jy-7au/monomer-gamma/OUT.ABACUS/')

        wfc = read_wfc_lcao_txt(outdir + 'WFC_NAO_GAMMA1.txt')[0]
        S = read_triu(outdir + 'data-0-S')
        dat = read_running_scf_log(outdir + 'running_scf.log')
        nbands = 25

        sigma = _svd_on_wfc(wfc, S, nbands, 
                            dat['natom'], 
                            dat['nzeta'], 
                            'rotational-invariant',
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


    def test_wll_multi_k(self):
        import os
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


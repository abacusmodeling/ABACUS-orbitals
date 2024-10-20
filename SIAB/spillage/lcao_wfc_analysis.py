'''
Performing analysis on lcao wavefunctions, get number of zeta functions
for each l.
'''

from SIAB.spillage.index import _lin2comp
import os
import numpy as np
import scipy.linalg as la

def api(C, S, natom, nzeta, method, **kwargs):
    '''an API to perform analysis on wave functions.
    
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
        method : str
            Method used to perform the analysis. The available methods are:
            'svd-aniso-max', 'svd-aniso-svd', 'svd-atomic', 'svd-iso', 'wll'.
        **kwargs : optional
            Additional parameters for the analysis. The available parameters are:
            'nband': int|str, number of bands selected for the analysis, 'all' for all bands.
            'loss_thr': float, threshold for the significance of zeta functions.
    
    Returns
    -------
        out : list of list of list of float
            Singular values for each atomtype and each l, each zeta function.
    '''
    assert method in ['svd-aniso-max', 'svd-aniso-svd', 'svd-atomic',
        'svd-iso', 'wll'], 'method not recognized'
    if method == 'wll':
        raise ValueError('method wll is deprecated due to its unstable behavior')
    nband = kwargs.get('nband', 'all')
    loss_thr = kwargs.get('loss_thr', 0.9)
    svd = {'svd-aniso-max': _svd_aniso_max, 'svd-aniso-svd': _svd_aniso_svd,
           'svd-atomic': _svd_atomic, 'svd-iso': _svd_iso}
    return svd[method](C, S, nband, natom, nzeta, loss_thr)

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

def _svd_aniso_max(C, S, nband, natom, nzeta, loss_thr = 0.9):
    '''perform svd-based analysis on wfc for each (nz, nband*nat) block,
    then take the maximum singular value of each m for each l, returning
    the maximal singluar value for each zeta function of each l for each
    atomtype. If there are equivalent atoms, then the ideally largest
    singluar value will be sqrt(Nat), where Nat is the number of equivalent
    atoms.
    
    Parameters
    ----------
        C : 2D array of shape (nao, nbands)
            Wave function coefficients in LCAO basis. The datatype is
            complex for multi-k calculations and float for gamma-only.
        S : 2D array of shape (nao, nao)
            Overlap matrix.
        nband : int|str
            Number of bands selected for the analysis, 'all' for all bands.
        natom : list of int
            Number of atoms for each type.
        nzeta : list of list of int
            nzeta[i][l] specifies the number of zeta orbitals of the
            angular momentum l of type i. len(nzeta) must equal len(natom).
        loss_thr : float
            Threshold for the significance of zeta functions. If the singular
            value of a zeta function is smaller than thr, it is considered
            insignificant. The loss will be evaluated by Frobenius norm.
    
    Returns
    -------
        sigma_tlm : list of list of list of float
            Singular values for each atomtype and each l, each zeta function.
    '''

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
            for m in range(2*l+1): # record each m, then take max
                sigma_tlm[it][l].append(la.svd(Ct[it][l][m], compute_uv=False))

    out = [[np.linalg.norm(sigma_tlm[it][l], axis=0, ord=np.inf)
            for l in range(len(nzeta[it]))] for it in range(ntyp)]
    
    ##############################################
    # SVD-based space truncation loss estimation #
    ##############################################
    loss = 0
    for it in range(ntyp):
        for l in range(lmax[it]+1):
            smax = [s if s/np.sqrt(natom[it]) >= loss_thr else 0 for s in out[it][l]]
            for m in range(2*l+1):
                ut, _, vt = la.svd(Ct[it][l][m], full_matrices=False)
                loss += la.norm(Ct[it][l][m] - ut @ np.diag(smax) @ vt, 'fro')**2
    print(f'SVD: with threshold {loss_thr}, the Frobenius norm loss is estimated to {loss:.8e}')
    return out

def _svd_aniso_svd(C, S, nband, natom, nzeta, loss_thr = 0.9):
    '''perform svd-based analysis on wfc for each block with size (nz, nband*nat*m)
    for each l. The ideal singluar values are sqrt(2l+1)*sqrt(Nat) at most for each
    zeta function, in which the l is the angular momentum, Nat is the number of atoms
    in the identical chemical environment. This value means there are 2l+1 magnetic
    channels of l are equally significant, and Nat atoms of the type it are equally
    significant.
    
    Parameters
    ----------
        C : 2D array of shape (nao, nbands)
            Wave function coefficients in LCAO basis. The datatype is
            complex for multi-k calculations and float for gamma-only.
        S : 2D array of shape (nao, nao)
            Overlap matrix.
        nband : int|str
            Number of bands selected for the analysis, 'all' for all bands.
        natom : list of int
            Number of atoms for each type.
        nzeta : list of list of int
            nzeta[i][l] specifies the number of zeta orbitals of the
            angular momentum l of type i. len(nzeta) must equal len(natom).
        loss_thr : float
            Threshold for the significance of zeta functions. If the singular
            value of a zeta function is smaller than thr, it is considered
            insignificant.
    
    Returns
    -------
        sigma_tlm : list of list of list of float
            Singular values for each atomtype and each l, each zeta function.
    '''

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
    
    ##############################################
    # SVD-based space truncation loss estimation #
    ##############################################
    loss = 0
    sigma_tlm = [[[] for _ in range(len(nzeta[it]))] for it in range(ntyp)]
    for it in range(ntyp):
        for l in range(lmax[it]+1):
            u, sigma, vh = la.svd(Ct[it][l], full_matrices=False)
            sigma_tlm[it][l] = sigma
            # sigma = np.array([s if s/np.sqrt(2*l+1)/np.sqrt(natom[it]) >= loss_thr 
            #          else 0 for s in sigma])
            sigma = np.array([s if s/np.sqrt(natom[it]) >= loss_thr else 0 for s in sigma])
            loss += la.norm(Ct[it][l] - u @ np.diag(sigma) @ vh, 'fro') ** 2
    print(f'SVD: with threshold {loss_thr}, the Frobenius norm loss is estimated to {loss:.8e}')
    return sigma_tlm

def _svd_iso(C, S, nband, natom, nzeta, loss_thr = 0.9):
    '''perform svd-based analysis on wfc with isotropicity of m for each l,
    based on averaging over all m for each l, returning the nzeta for each
    type for each l. This is a deprecated method.'''

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

    out = [[np.linalg.norm(sigma_tlm[it][l], axis=0, ord=2)
            for l in range(len(nzeta[it]))] for it in range(ntyp)]
    
    ##############################################
    # SVD-based space truncation loss estimation #
    ##############################################
    loss = 0
    for it in range(ntyp):
        for l in range(lmax[it]+1):
            smax = [s if s/np.sqrt(natom[it]) >= loss_thr else 0 for s in out[it][l]]
            for m in range(2*l+1):
                ut, _, vt = la.svd(Ct[it][l][m], full_matrices=False)
                loss += la.norm(Ct[it][l][m] - ut @ np.diag(smax) @ vt, 'fro')**2
    print(f'SVD: with threshold {loss_thr}, the Frobenius norm loss is estimated to {loss:.8e}')
    return out

def _svd_atomic(C, S, nband, natom, nzeta, loss_thr = 0.9):
    '''perform svd-based analysis to get the significance of each zeta func
    for each atom (each type, l, also m-distinct).
    
    Parameters
    ----------
        C : 2D array of shape (nao, nbands)
            Wave function coefficients in LCAO basis. The datatype is
            complex for multi-k calculations and float for gamma-only.
        S : 2D array of shape (nao, nao)
            Overlap matrix.
        nband : int|str
            Number of bands selected for the analysis, 'all' for all bands.
        natom : list of int
            Number of atoms for each type.
        nzeta : list of list of int
            nzeta[i][l] specifies the number of zeta orbitals of the
            angular momentum l of type i. len(nzeta) must equal len(natom).
        loss_thr : float
            Threshold for the significance of zeta functions. If the singular
            value of a zeta function is smaller than thr, it is considered
            insignificant.
    
    Returns
    -------
        out : list of list of list of float
            Singular values for each atomtype and each l, each zeta function.
    '''

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

    out = [[] for it in range(ntyp)]
    for it in range(ntyp): # this is, unavoidable because wfc itself has all type info
        for l in range(lmax[it]+1):
            s_tl = [la.svd(Ct[it][l][ia, m], compute_uv=False) for ia in range(natom[it]) for m in range(2*l+1)]
            # take max over ia and m
            s_tl = np.linalg.norm(s_tl, axis=0, ord=np.inf)
            out[it].append(s_tl)

    ##############################################
    # SVD-based space truncation loss estimation #
    ##############################################
    loss = 0
    for it in range(ntyp):
        for l in range(lmax[it]+1):
            smax = out[it][l] # for all atom within the type and all m
            for ia in range(natom[it]):
                for m in range(2*l+1):
                    u, _, vh = la.svd(Ct[it][l][ia, m], full_matrices=False)
                    loss += la.norm(Ct[it][l][ia, m] - u @ np.diag(smax) @ vh, 'fro')**2
    print(f'SVD: with threshold {loss_thr}, the Frobenius norm loss is estimated to {loss:.8e}')
    return out

############################################################
#                           Test
############################################################
import unittest

from SIAB.spillage.datparse import read_wfc_lcao_txt, read_triu, \
        read_running_scf_log

class TestLCAOWfcAnalysis(unittest.TestCase):

    @unittest.skip('deprecated, failed for case Si-trimer-2.00-10au')
    def test_wll_gamma(self):
        
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

    @unittest.skip('deprecated, failed for case Si-trimer-2.00-10au')
    def test_wll_multi_k(self):
        
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

    @unittest.skip('this case is for plotting the relation between nbands and nzeta')
    def test_nzeta_nband_plt(self):
        
        outdir = '/root/documents/simulation/orbgen/Test1Aluminum-20241011/Al-dimer-2.00-10au/OUT.Al-dimer-2.00-10au/'
        method = {'aniso-max': _svd_aniso_max, 'aniso-svd': _svd_aniso_svd, 
                  'atomic': _svd_atomic}
        wfc = read_wfc_lcao_txt(outdir + 'WFC_NAO_GAMMA1.txt')[0]
        S = read_triu(outdir + 'data-0-S')
        dat = read_running_scf_log(outdir + 'running_scf.log')

        nao, nbands = wfc.shape

        import matplotlib.pyplot as plt
        sigmas = []
        fig, ax = plt.subplots(1, 3, figsize=(18, 6))
        for nbnd in range(4, nbands, 5):
            sigma = method(wfc, S, nbnd, dat['natom'], dat['nzeta'])
            for i, (_, nz) in enumerate(zip(dat['natom'], dat['nzeta'])):
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

    def test_svd_iso(self):
        here = os.path.dirname(os.path.abspath(__file__))
        outdir = os.path.join(here, 'testfiles/Si/jy-7au/monomer-gamma/OUT.ABACUS/')

        wfc = read_wfc_lcao_txt(outdir + 'WFC_NAO_GAMMA1.txt')[0]
        S = read_triu(outdir + 'data-0-S')
        dat = read_running_scf_log(outdir + 'running_scf.log')

        sigma = _svd_iso(wfc, S, 'all', dat['natom'], dat['nzeta'])
        self.assertEqual(len(sigma), len(dat['natom']))

        print(f'Method: svd-iso')
        for i, (_, nz) in enumerate(zip(dat['natom'], dat['nzeta'])):
            print(f'atom type {i+1}')
            for l in range(len(nz)):
                print(f'l = {l}')
                print(np.array(sigma[i][l]) ** 2)
        
    def test_svd_atomic(self):
        here = os.path.dirname(os.path.abspath(__file__))
        outdir = os.path.join(here, 'testfiles/Si/jy-7au/monomer-k/OUT.ABACUS/')

        wfc = read_wfc_lcao_txt(outdir + 'WFC_NAO_K2.txt')[0]
        S = read_triu(outdir + 'data-1-S')
        dat = read_running_scf_log(outdir + 'running_scf.log')

        sigma = _svd_atomic(wfc, S, 'all', dat['natom'], dat['nzeta'])
        self.assertEqual(len(sigma), len(dat['natom'])) # number of atom types
        for i, (_, nz) in enumerate(zip(dat['natom'], dat['nzeta'])):
            self.assertEqual(len(sigma[i]), len(nz)) # number of l orbitals

        print(f'Method: svd-atomic')
        for i, (_, nz) in enumerate(zip(dat['natom'], dat['nzeta'])):
            print(f'atom type {i+1}')
            for l in range(len(nz)):
                print(f'l = {l}')
                print(np.array(sigma[i][l]) ** 2)

    def test_svd_aniso_max(self):
        here = os.path.dirname(os.path.abspath(__file__))
        outdir = os.path.join(here, 'testfiles/Si/jy-7au/monomer-gamma/OUT.ABACUS/')

        wfc = read_wfc_lcao_txt(outdir + 'WFC_NAO_GAMMA1.txt')[0]
        S = read_triu(outdir + 'data-0-S')
        dat = read_running_scf_log(outdir + 'running_scf.log')

        sigma = _svd_aniso_max(wfc, S, 'all', dat['natom'], dat['nzeta'])
        self.assertEqual(len(sigma), len(dat['natom']))
        for i, (_, nz) in enumerate(zip(dat['natom'], dat['nzeta'])):
            self.assertEqual(len(sigma[i]), len(nz))

        print(f'Method: svd-aniso-max')
        for i, (_, nz) in enumerate(zip(dat['natom'], dat['nzeta'])):
            print(f'atom type {i+1}')
            for l in range(len(nz)):
                print(f'l = {l}')
                print(np.array(sigma[i][l]) ** 2)

    def test_svd_aniso_svd(self):
        here = os.path.dirname(os.path.abspath(__file__))
        outdir = os.path.join(here, 'testfiles/Si/jy-7au/monomer-gamma/OUT.ABACUS/')

        wfc = read_wfc_lcao_txt(outdir + 'WFC_NAO_GAMMA1.txt')[0]
        S = read_triu(outdir + 'data-0-S')
        dat = read_running_scf_log(outdir + 'running_scf.log')

        sigma = _svd_aniso_svd(wfc, S, 'all', dat['natom'], dat['nzeta'])

        self.assertEqual(len(sigma), len(dat['natom']))
        for i, (_, nz) in enumerate(zip(dat['natom'], dat['nzeta'])):
            self.assertEqual(len(sigma[i]), len(nz))
        
        print(f'Method: svd-aniso-svd')
        for i, (_, nz) in enumerate(zip(dat['natom'], dat['nzeta'])):
            print(f'atom type {i+1}')
            for l in range(len(nz)):
                print(f'l = {l}')
                print(np.array(sigma[i][l]) ** 2)

if __name__ == '__main__':
    unittest.main()

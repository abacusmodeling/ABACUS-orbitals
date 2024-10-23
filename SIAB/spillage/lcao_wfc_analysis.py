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
            'svd-fold' or 'svd-max'
        **kwargs : optional
            Additional parameters for the analysis. The available parameters are:
            'nband': int|str, number of bands selected for the analysis, 'all' for all bands.
            'filter': float, threshold for the significance of zeta functions.
    
    Returns
    -------
        out : list of list of list of float
            Singular values for each atomtype and each l, each zeta function.
    '''
    if method not in ['svd-fold', 'svd-max']:
        raise ValueError('method not recognized')
    nband = kwargs.get('nband', 'all')
    filter = kwargs.get('filter', None)
    svd = {'svd-fold': _svdfold,'svd-max': _svdmax}
    return svd[method](C, S, nband, natom, nzeta, filter)

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

def _svdmax(C, 
            S, 
            nband, 
            natom, 
            nzeta, 
            filter = None):
    '''perform svd analysis on submatrix with size (nz, nbnd) extracted from
    the whole wavefunction. The submatrix is certainly indexed by [it][l][iat][m]
    , then taking maximum over both [iat] and [m] dimensions, yielding
    the maximal singular value of each zeta function of each l for each atomtype.
    
    Parameters
    ----------
        C : 2D array of shape (nao, nbands)
            Wave function coefficients in LCAO basis. The datatype is
            complex for multi-k calculations and float for gamma-only.
        S : 2D array of shape (nao, nao)
            Overlap matrix.
        nband : int|str
            Number of bands selected for the analysis, 'all' for all bands.
        natom : list[int]
            Number of atoms for each type.
        nzeta : list[list[int]]
            nzeta[i][l] specifies the number of zeta orbitals of the
            angular momentum l of type i. len(nzeta) must equal len(natom).
        filter : float|list[float]|list[list[int]]|None
            if specified as list[float], will be the threshold for filtering 
            singular values for each atom type. All values larger than it 
            will be treated as significant, omit otherwise.
            if specified as float, will be the threshold for all atom types.
            if specified as list[list[int]], will be the number of singular 
            values for each l to be kept.
            if specified as None (default, thus not specified), will not do any
            filtering.
            **For this method, the maximal value of sigma will be 1.**
    
    Returns
    -------
    sigma: list[list[list[float]]]
        all the singluar values before filtering, can be indexed by [it][l][iz]
    nzeta_sub: list[list[int]]|None
        the number of zeta functions after filtering, can be indexed by [it][l].
        if filter is not specified, will be None.
    loss: list[float]|None
        loss of space truncation, can be indexed by [it]. If filter is not
        specified, will be None.
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

    out = [[] for _ in range(ntyp)]
    for it in range(ntyp): # this is, unavoidable because wfc itself has all type info
        for l in range(lmax[it]+1):
            sigma = [la.svd(Ct[it][l][ia, m], compute_uv=False)\
                     for ia in range(natom[it]) for m in range(2*l+1)]
            # take max over ia and m
            sigma_max = np.linalg.norm(sigma, axis=0, ord=np.inf)
            out[it].append(sigma_max)

    if filter is None:
        return out, None, None
    
    ##############################################
    # SVD-based space truncation loss estimation #
    ##############################################
    filter = [filter] * ntyp if isinstance(filter, float) else filter
    jobtype = 'cal_nz' if isinstance(filter, list) and len(filter) == ntyp\
        and all(isinstance(f, float) for f in filter) else 'cal_loss'

    nz = [[len(np.where(np.array(sigma) >= filter[it])[0]) 
           for sigma in out[it]] for it in range(ntyp)] if jobtype == 'cal_nz'\
           else filter
    
    loss = 0
    for it in range(ntyp):
        for l in range(lmax[it]+1):
            for ia in range(natom[it]):
                for m in range(2*l+1):
                    u, _, vh = la.svd(Ct[it][l][ia, m], full_matrices=False)
                    # tf_ += la.norm(Ct[it][l][ia, m] - u @ np.diag(out[it][l]) @ vh, 'fro')**2
                    loss += la.norm(u[:, nz[it][l]:] @ np.diag(out[it][l][nz[it][l]:]) @ vh[nz[it][l]:, :], 'fro')**2
    return out, nz, loss

def _svdfold(C, 
             S, 
             nband, 
             natom, 
             nzeta, 
             filter = None):
    '''perform svd-based analysis on wfc for each block with size (nz, nband*nat*m)
    for each l. The ideal singluar values are sqrt(2l+1)*sqrt(nat) at most for each
    zeta function, in which the l is the angular momentum, nat is the number of atoms
    in the identical chemical environment. This value means there are 2l+1 magnetic
    channels of l are equally significant, and nat atoms of the type it are equally
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
        natom : list[int]
            Number of atoms for each type.
        nzeta : list[list[int]]
            nzeta[i][l] specifies the number of zeta orbitals of the
            angular momentum l of type i. len(nzeta) must equal len(natom).
        filter : float|list[float]|list[list[int]]|None
            if specified as list[float], will be the threshold for filtering 
            singular values for each atom type. All values larger than it 
            will be treated as significant, omit otherwise.
            if specified as float, will be the threshold for all atom types.
            if specified as list[list[int]], will be the number of singular 
            values for each l to be kept.
            if specified as None (default, thus not specified), will not do any
            filtering.
            **For this method, the maximal value of sigma will be sqrt(nat)\*
            sqrt(2l+1).**
    
    Returns
    -------
    sigma: list[list[list[float]]]
        all the singluar values before filtering, can be indexed by [it][l][iz]
    nzeta_sub: list[list[int]]|None
        the number of zeta functions after filtering, can be indexed by [it][l].
        if filter is not specified, will be None.
    loss: list[float]|None
        loss of space truncation, can be indexed by [it]. If filter is not
        specified, will be None.
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
    
    sigma = [[[] for _ in range(len(nzeta[it]))] for it in range(ntyp)]
    for it in range(ntyp):
        for l in range(lmax[it]+1):
            s = la.svd(Ct[it][l], compute_uv=False)
            sigma[it][l] = s

    if filter is None:
        return sigma, None, None
    
    ##############################################
    # SVD-based space truncation loss estimation #
    ##############################################
    filter = [filter] * ntyp if isinstance(filter, float) else filter
    jobtype = 'cal_nz' if isinstance(filter, list) and len(filter) == ntyp\
        and all(isinstance(f, float) for f in filter) else 'cal_loss'
    
    nz = [[len(np.where(np.array(sigma_l) >= filter[it])[0])
           for sigma_l in sigma[it]] for it in range(ntyp)]\
           if jobtype == 'cal_nz' else filter
    
    loss = 0
    for it in range(ntyp):
        for l in range(lmax[it]+1):
            u, s, vh = la.svd(Ct[it][l], full_matrices=False)
            loss += la.norm(u[:, nz[it][l]:] @ np.diag(s[nz[it][l]:]) @ vh[nz[it][l]:, :], 'fro')**2
    return sigma, nz, loss

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
        method = {'svd-fold': _svdfold, 'svd-max': _svdmax}
        wfc = read_wfc_lcao_txt(outdir + 'WFC_NAO_GAMMA1.txt')[0]
        S = read_triu(outdir + 'data-0-S')
        dat = read_running_scf_log(outdir + 'running_scf.log')

        nao, nbands = wfc.shape

        import matplotlib.pyplot as plt
        sigmas = []
        fig, ax = plt.subplots(1, 3, figsize=(18, 6))
        for nbnd in range(4, nbands, 5):
            sigma, _, _ = method(wfc, S, nbnd, dat['natom'], dat['nzeta'])
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
    
    def test_svdmax(self):
        here = os.path.dirname(os.path.abspath(__file__))
        outdir = os.path.join(here, 'testfiles/Si/jy-7au/monomer-k/OUT.ABACUS/')

        wfc = read_wfc_lcao_txt(outdir + 'WFC_NAO_K2.txt')[0]
        S = read_triu(outdir + 'data-1-S')
        dat = read_running_scf_log(outdir + 'running_scf.log')

        sigma, nz, loss = _svdmax(wfc, S, 'all', dat['natom'], dat['nzeta'])
        print(f'Method: svd-max, loss = {loss}')
        self.assertEqual(len(sigma), len(dat['natom'])) # number of atom types
        for i, (_, nz) in enumerate(zip(dat['natom'], dat['nzeta'])):
            self.assertEqual(len(sigma[i]), len(nz)) # number of l orbitals

        for i, (_, nz) in enumerate(zip(dat['natom'], dat['nzeta'])):
            print(f'atom type {i+1}')
            for l in range(len(nz)):
                print(f'l = {l}')
                print(np.array(sigma[i][l]) ** 2)

    def test_svdfold(self):
        here = os.path.dirname(os.path.abspath(__file__))
        outdir = os.path.join(here, 'testfiles/Si/jy-7au/monomer-gamma/OUT.ABACUS/')

        wfc = read_wfc_lcao_txt(outdir + 'WFC_NAO_GAMMA1.txt')[0]
        S = read_triu(outdir + 'data-0-S')
        dat = read_running_scf_log(outdir + 'running_scf.log')

        sigma, nz, loss = _svdfold(wfc, S, 'all', dat['natom'], dat['nzeta'])
        print(f'Method: svd-fold, loss = {loss}')

        self.assertEqual(len(sigma), len(dat['natom']))
        for i, (_, nz) in enumerate(zip(dat['natom'], dat['nzeta'])):
            self.assertEqual(len(sigma[i]), len(nz))
        
        for i, (_, nz) in enumerate(zip(dat['natom'], dat['nzeta'])):
            print(f'atom type {i+1}')
            for l in range(len(nz)):
                print(f'l = {l}')
                print(np.array(sigma[i][l]) ** 2)

if __name__ == '__main__':
    unittest.main()

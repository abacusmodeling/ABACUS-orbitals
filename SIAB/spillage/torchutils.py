import numpy as np
from SIAB.spillage.index import _lin2comp
from scipy.linalg import block_diag
import torch as th

def _t_jy2ao(coef: list, natom: int, nbes: list) -> th.Tensor:
    '''torch specific implementation of spillage/basistrans:jy2ao.
    For more information, please see the original docstring.'''
    assert len(natom) == len(nbes) == len(coef)

    assert all(len(coef_t) <= len(nbes_t)
               for nbes_t, coef_t in zip(nbes, coef))

    assert all(all(all(len(coef_tlz) <= nbes_tl for coef_tlz in coef_tl)
                   for nbes_tl, coef_tl in zip(nbes_t, coef_t))
               for nbes_t, coef_t in zip(nbes, coef))
    
    def _gen_q2zeta(coef, natom, nbes):
        lmax = [len(nbes_t) - 1 for nbes_t in nbes]
        for itype, _, l, _ in _lin2comp(natom, lmax=lmax):
            if l >= len(coef[itype]) or len(coef[itype][l]) == 0:
                yield np.zeros((nbes[itype][l], 0))
            else:
                C = np.zeros((nbes[itype][l], len(coef[itype][l])))
                C[:len(coef[itype][l][0])] = np.array(coef[itype][l]).T
                yield C
    
    return th.Tensor(block_diag(*_gen_q2zeta(coef, natom, nbes)))

def _t_mrdiv(X: th.Tensor, Y: th.Tensor) -> th.Tensor:
    assert (len(X.shape) == 1 and len(Y.shape) == 2) \
            or (len(X.shape) > 1 and len(Y.shape) > 1)
    return th.linalg.solve(Y.swapaxes(-2, -1), X.swapaxes(-2, -1)) \
            .swapaxes(-2, -1) \
            if len(X.shape) > 1 else th.linalg.solve(Y.T, X)

def _t_rfrob(X: th.Tensor, Y: th.Tensor, rowwise=False):
    '''torch specific implementation of spillage/linalg_helper:rfrob.
    For more information, please see the original docstring.'''
    return (X * Y.conj()).real.sum(-1 if rowwise else (-2,-1))

import os
from SIAB.spillage.datparse import read_running_scf_log, \
read_triu, read_wfc_lcao_txt
def _t_jy_data_extract(outdir):
    info = read_running_scf_log(os.path.join(outdir, 'running_scf.log'))
    nspin, wk, natom, nzeta = [info[key] for key in
                               ['nspin', 'wk', 'natom', 'nzeta']]
    
    nk = len(wk)
    S = [read_triu(os.path.join(outdir, f'data-{ik}-S'))
         for ik in range(nk)]
    T = [read_triu(os.path.join(outdir, f'data-{ik}-T'))
         for ik in range(nk)]

    wfc_suffix = 'GAMMA' if nk == 1 else 'K'
    C = [read_wfc_lcao_txt(os.path.join(outdir, f'WFC_NAO_{wfc_suffix}{ik+1}.txt'))[0].tolist()
         for ik in range(nspin * nk)]
    
    if nspin == 2:
        S = [*S, *S]
        T = [*T, *T]
        wk = [*wk, *wk]

    return {'natom': natom, 'nzeta': nzeta, 'wk': wk, 
            'S': th.Tensor(S), 'T': th.Tensor(T), 'C': th.Tensor(C)}


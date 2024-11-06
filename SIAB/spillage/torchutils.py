import numpy as np
from SIAB.spillage.index import _lin2comp, _coef_flatten
import torch as th
import unittest
from torch_optimizer import SWATS, Yogi, DiffGrad, RAdam
from SIAB.spillage.listmanip import flatten, nest, nestpat
from time import time

def minimize(f, c, method, maxiter, disp, ndisp, **kwargs):
    '''Minimize the spillage function f with respect to the coefficients c
    using various optimization methods implemented in PyTorch
    
    Parameters
    ----------
    f : function
        The spillage function to be minimized
    c : list[list[list[list[float]]]]
        The initial coefficients to be optimized
    method : str
        The optimization method to be used, default is 'swats'
    maxiter : int
        The maximum number of iterations
    disp : bool
        Whether to display the optimization process
    ndisp : int
        The number of times to display the optimization process
    **kwargs : dict
        Additional keyword arguments for the optimization method, only `lr`, 
        `beta`, `eps`, `weight_decay` are supported
    '''

    c0 = th.Tensor(flatten(c))
    c0.requires_grad = True # necessary for autodiff

    optimizer = {'swats': SWATS, 'yogi': Yogi, 'diffgrad': DiffGrad, 
                 'radam': RAdam}[method.lower()]([c0], **kwargs)
    if disp:
        print(f'\nPyTorch.{method} Spillage optimization with Nmax = {maxiter} steps')
        print(f'{"method":>10s} {"step":>8s} {"loss (spillage)":>20s} {"time/s":>10s}')
    
    method = method.upper()
    t0 = time()
    for i in range(maxiter):
        optimizer.zero_grad()
        loss = f(c0)
        if disp and i % (maxiter//ndisp) == 0:
            dt = time() - t0
            t0 = time()
            print(f'{method:>10s} {i:>8d} {loss.item():>20.10e} {dt:>10.2f}')
        # autodiff
        loss.backward()
        optimizer.step()
    if disp:
        print(f'{method:>10s} {maxiter:>8d} {loss.item():>20.10e} {time()-t0:>10.2f}\n')
    
    return nest(c0.tolist(), nestpat(c)), loss.item()

def _t_transpose(a: th.Tensor, axes: tuple) -> th.Tensor:
    '''the alternative to the numpy transpose function, which supports
    the multi-dimensional transpose by accepting the axes tuple.'''
    return a.permute(axes)

def _t_jy2ao(coef: th.Tensor, natom: int, nzeta: list, nbes: list) -> th.Tensor:
    '''Generate the transformation matrix from primitive jy basis to pseudo-
    atomic orbital basis. This is a PyTorch adapted implementation of 
    spillage/basistrans:jy2ao. For more information, please see the original 
    docstring.
    
    Note
    ----
    the coef here is flattened, unlike what in the original impl.
    coef is indexed by [itype, l, zeta, q] -> float

    natom still in type of list[int] and nbes still list[list[int]],
    indexed by [itype, l] -> int
    '''
    assert len(natom) == len(nbes) # == len(coef) # ntype equal, but coef is flattened

    def _gen_q2zeta(coef, natom, nzeta, nbes):
        lmax = [len(nbes_t) - 1 for nbes_t in nbes]
        index_, range_ = _coef_flatten(nzeta, nbes)
        for itype, _, l, _ in _lin2comp(natom, lmax=lmax):
            # the len(coef[ityp]) is the lmax of ityp, can be len(nzeta[ityp])
            # the len(coef[ityp][l]) is nzeta[ityp][l]
            if l >= len(nzeta[itype]) or nzeta[itype][l] == 0:
                yield th.zeros((nbes[itype][l], 0))
            else:
                i = index_.index((itype, l))
                start, stop = range_[i].start, range_[i].stop
                yield th.Tensor(coef[start:stop]).reshape(-1, nbes[itype][l]).T
    
    return th.block_diag(*_gen_q2zeta(coef, natom, nzeta, nbes))

def _t_mrdiv(X: th.Tensor, Y: th.Tensor) -> th.Tensor:
    '''
    Right matrix division X @ inv(Y). This is PyTorch adapted 
    implementation of spillage/linalg_helper:mrdiv.
    '''
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
    '''PyTorch adapted jy_data_extract function, returns wk, S, T and C as torch.Tensor.
    For detailed information, please see the original docstring in spillage/datparse.py.'''

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

    return {'natom': natom, 'nzeta': nzeta, 
            'wk': th.Tensor(np.array(wk)), 
            'S': th.Tensor(np.array(S)), 
            'T': th.Tensor(np.array(T)), 
            'C': th.Tensor(np.array(C))}

class TestTorchUtils(unittest.TestCase):

    def test_t_transpose(self):
        '''test the _t_transpose function'''
        a = th.Tensor(np.random.randn(2, 3, 4))
        axes = (1, 0, 2)
        b = _t_transpose(a, axes)
        self.assertTrue(b.shape == (3, 2, 4))
        # check values
        for i in range(3):
            for j in range(2):
                for k in range(4):
                    self.assertAlmostEqual(a[j, i, k], b[i, j, k])

if __name__ == '__main__':
    unittest.main()
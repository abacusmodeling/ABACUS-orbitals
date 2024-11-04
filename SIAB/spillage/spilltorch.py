'''
This module defines the minimizer of Spillage
for PyTorch implementation
'''
import torch as th
from SIAB.spillage.spillage import Spillage
from SIAB.spillage.torchutils import _t_rfrob, _t_jy2ao, _t_mrdiv,\
    _t_jy_data_extract
import numpy as np
from SIAB.spillage.listmanip import nestpat, nest, flatten
from SIAB.spillage.index import perm_zeta_m, _lin2comp
from torch_optimizer import SWATS

class SpillTorch(Spillage):


    def __init__(self):
        super().__init__()

    def reset(self):
        super().reset()

    def _tab_frozen(self, coef_frozen):
        self.spill_frozen = [None] * len(self.config)
        self.ref_Pfrozen_jy = [None] * len(self.config)

        if coef_frozen is None:
            return

        for iconf, dat in enumerate(self.config):
            jy2frozen = _t_jy2ao(coef_frozen, dat['natom'], dat['nbes'])

            jy_jy = th.Tensor(dat['jy_jy'])
            frozen_frozen = jy2frozen.T @ jy_jy @ jy2frozen
            ref_frozen = th.Tensor(dat['ref_jy']) @ jy2frozen

            # no need to compute <ref|op|frozen_dual>
            ref_frozen_dual = _t_mrdiv(ref_frozen[0], frozen_frozen[0])

            self.ref_Pfrozen_jy[iconf] = \
                    ref_frozen_dual @ jy2frozen.T @ jy_jy

            # spill_frozen before weighted sum over k
            tmp = _t_rfrob(ref_frozen_dual @ frozen_frozen[1],
                        ref_frozen_dual, True) \
                    - 2.0 * _t_rfrob(ref_frozen_dual, ref_frozen[1], True)

            self.spill_frozen[iconf] = th.Tensor(dat['wk']) @ tmp

    # def _tab_deriv # not implemented, because PyTorch can do automatic differentiation

    def _generalized_spillage(self, iconf, coef, ibands):
        '''PyTorch adapted version of Spillage._generalized_spillage'''
        dat = self.config[iconf]

        if ibands == 'all':
            ibands = range(dat['ref_ref'][1].shape[1])

        wk = th.Tensor(dat['wk'])
        spill = (wk @ dat['ref_ref'][1][:, ibands]).real.sum()
        _jy2ao = _t_jy2ao(coef, dat['natom'], dat['nbes'])

        V = th.Tensor(dat['ref_jy'])[:,:,ibands,:] @ _jy2ao
        if self.spill_frozen is not None:
            V -= self.ref_Pfrozen_jy[iconf][:,:,ibands,:] @ _jy2ao
            spill += self.spill_frozen[iconf][:,ibands].sum()

        W = _jy2ao.T @ th.Tensor(dat['jy_jy']) @ _jy2ao
        
        V_dual = _t_mrdiv(V[0], W[0])
        _V_dual = np.array(V_dual.tolist())
        VdaggerV = th.Tensor(_V_dual.transpose((0,2,1)).conj()) @ V_dual

        spill += wk @ (_t_rfrob(W[1], VdaggerV)
                                         - 2.0 * _t_rfrob(V_dual, V[1]))
        return spill / len(ibands)
    
    def opt(self, coef_init, coef_frozen, iconfs, ibands,
            options, nthreads=1):
        
        if coef_frozen is not None:
            self._tab_frozen(coef_frozen)

        if iconfs == 'all':
            iconfs = range(len(self.config))
        nconfs = len(iconfs)

        if not isinstance(ibands, list):
            ibands = [ibands]

        assert len(ibands) == nconfs

        pat = nestpat(coef_init)
        def _t_f(c):
            '''PyTorch adapted version of Spillage.f'''
            s = lambda i: self._generalized_spillage(iconfs[i],
                                                     nest(c.tolist(), pat),
                                                     ibands[i])
            spills = th.Tensor([s(i) for i in range(nconfs)])
            return spills.sum() / nconfs
        
        c0 = th.Tensor(flatten(coef_init))

        # only swats
        optimizer = SWATS([c0], lr=options.get('learning_rate', 0.01))

        maxiter, ndisp = options.get('maxiter', 1000), options.get('ndisp', 10)
        for i in range(maxiter):
            optimizer.zero_grad()
            loss = _t_f(c0)
            loss.backward()
            optimizer.step()
            if options.get('disp', True) and i % (maxiter//ndisp) == 0:
                print(f'step {i:8d}, loss {loss.item():.8e}')
            
        return c0.tolist(), loss.item()

class SpillTorch_jy(SpillTorch):
    
    def config_add(self, outdir, weight=(0.0, 1.0)):
        raw = _t_jy_data_extract(outdir)
        C = raw['C']
        S = raw['S']
        T = raw['T']

        wov, wop = weight
        ref_ov_ref = th.sum(C.conj() * (S @ C), -2)
        ref_op_ref = th.sum(C.conj() * (T @ C), -2)

        ref_ov_jy = C.swapaxes(-2, -1).conj() @ S
        ref_op_jy = C.swapaxes(-2, -1).conj() @ T

        ref_ref = np.array([ref_ov_ref, wov*ref_ov_ref + wop*ref_op_ref])
        ref_jy  = np.array([ref_ov_jy, wov*ref_ov_jy + wop*ref_op_jy])
        jy_jy   = np.array([S, wov*S + wop*T])

        p = perm_zeta_m(_lin2comp(raw['natom'], nzeta=raw['nzeta']))

        ref_jy = ref_jy[:,:,:,p].copy()
        jy_jy = jy_jy[:,:,:,p][:,:,p,:].copy()

        self.config.append({
            'natom': raw['natom'],
            'nbes': raw['nzeta'],
            'wk': raw['wk'],
            'ref_ref': th.Tensor(ref_ref),
            'ref_jy': th.Tensor(ref_jy),
            'jy_jy': th.Tensor(jy_jy)
        })

class SpillTorch_pw(SpillTorch):

    def __init__(self):
        raise NotImplementedError('SpillTorch_pw is not implemented yet')


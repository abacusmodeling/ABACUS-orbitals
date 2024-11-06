'''
This module defines the minimizer of Spillage
for PyTorch implementation
'''
import torch as th
from SIAB.spillage.spillage import Spillage
from SIAB.spillage.torchutils import _t_rfrob, _t_mrdiv, \
    _t_jy_data_extract, _t_jy2ao, minimize
import numpy as np
from SIAB.spillage.listmanip import flatten
from SIAB.spillage.index import perm_zeta_m, _lin2comp

class SpillTorch(Spillage):
    '''PyTorch adapted version of Spillage'''
    def __init__(self):
        super().__init__()

    def reset(self):
        super().reset()

    def _tab_frozen(self, coef_frozen):
        '''PyTorch adapted version of Spillage._tab_frozen. For detailed
        information, please refer to the original version in file
        SIAB/spillage/spillage.py
        
        Parameters
        ----------
        coef_frozen : list[list[list[list[float]]]]
            The frozen coefficients, indexed by [it][l][nz][q] -> float
        '''
        self.spill_frozen = [None] * len(self.config)
        self.ref_Pfrozen_jy = [None] * len(self.config)

        if coef_frozen is None:
            return

        for iconf, dat in enumerate(self.config):
            nzeta = [[len(coef_tl) for coef_tl in coef_t] for coef_t in coef_frozen]
            # <jy|pao_frozen>
            jy2frozen = _t_jy2ao(flatten(coef_frozen), dat['natom'], nzeta, dat['nbes'])
            # <jy|S|jy> and <jy|w1*S+w2*T|jy>
            jy_jy = dat['jy_jy'] # has been converted to torch.Tensor
            # in the following, use O represent S and w1*S+w2*T vstack together

            frozen_frozen = jy2frozen.T @ jy_jy @ jy2frozen   # <pao_frozen|O|pao_frozen>
            ref_frozen = dat['ref_jy'] @ jy2frozen # <ref|O|pao_frozen>

            # |pao_frozen_dual> = <pao_frozen|O|pao_frozen>^-1 |pao_frozen>
            # thus <pao_frozen_dual|O|pao_frozen> = I
            # <ref|O|pao_frozen_dual> = <ref|O|pao_frozen> <pao_frozen|O|pao_frozen>^-1
            ref_frozen_dual = _t_mrdiv(X=ref_frozen[0], Y=frozen_frozen[0])

            # <ref|O|pao_frozen_dual> <pao_frozen|jy> <jy|O|jy>
            # in which |pao_fronzen_dual><pao_frozen| = |pao_frozen>S^-1<pao_frozen| = Pfrozen
            # thus <ref|O*Pfrozen*Pjy*O|jy>
            self.ref_Pfrozen_jy[iconf] = \
                    ref_frozen_dual @ jy2frozen.T @ jy_jy

            # spill_frozen before weighted sum over k
            tmp = _t_rfrob(ref_frozen_dual @ frozen_frozen[1],
                        ref_frozen_dual, True) \
                    - 2.0 * _t_rfrob(ref_frozen_dual, ref_frozen[1], True)
            
            self.spill_frozen[iconf] = dat['wk'] @ tmp

    # def _tab_deriv # not implemented, because PyTorch can do automatic differentiation

    def _generalized_spillage(self, 
                              iconf, 
                              coef1d, 
                              nzeta,
                              ibands):
        '''
        PyTorch adapted version of Spillage._generalized_spillage. 
        Distinct from the original version, this function accecpt the 
        coef_1d as a 1D tensor.
        
        Parameters
        ----------
        iconf : int
            The index of the configuration in self.config
        coef1d : torch.Tensor
            The flattened coefficients that stored in PyTorch tensor
        nzeta : list[list[int]]
            The number of zeta for each l in each it, indexed by [it][l] -> int
        ibands : list[int]
            The indices of the bands to be calculated

        Returns
        -------
        float
            The spillage of the given configuration based on present given
            coefficients
        '''
        dat = self.config[iconf] 

        if ibands == 'all':
            ibands = range(dat['ref_ref'][1].shape[1])

        wk = dat['wk']

        spill = (wk @ dat['ref_ref'][1][:, ibands]).real.sum()
        _jy2ao = _t_jy2ao(coef1d, dat['natom'], nzeta, dat['nbes'])

        # <ref|S|ao> and <ref|w1*S+w2*T|ao>
        V = dat['ref_jy'][:,:,ibands,:] @ _jy2ao
        if self.spill_frozen is not None:
            V -= self.ref_Pfrozen_jy[iconf][:,:,ibands,:] @ _jy2ao
            spill += self.spill_frozen[iconf][ibands].sum()

        W = _jy2ao.T @ dat['jy_jy'] @ _jy2ao
        
        V_dual = _t_mrdiv(V[0], W[0])
        VdaggerV = V_dual.swapaxes(-2, -1).conj() @ V_dual
        
        spill += wk @ (_t_rfrob(W[1], VdaggerV) - 2.0 * _t_rfrob(V_dual, V[1]))

        return spill / len(ibands)
    
    def opt(self, coef_init, coef_frozen, iconfs, ibands,
            options, nthreads=1):
        '''
        This is the PyTorch overload of Spillage.opt.
        The optimization employing torch_optimizer is based on multiprocessing
        parallelism instead of multithreading, therefore the `nthreads` is not
        used in this function.
        '''
        if coef_frozen is not None:
            self._tab_frozen(coef_frozen)

        if iconfs == 'all':
            iconfs = range(len(self.config))
        nconfs = len(iconfs)

        if not isinstance(ibands, list):
            ibands = [ibands]

        nzeta = [[len(coef_tl) for coef_tl in coef_t] for coef_t in coef_init]
        def _t_f(c1d: th.Tensor):
            '''PyTorch adapted version of Spillage.f'''
            spill = 0
            for i in range(nconfs):
                out = self._generalized_spillage(iconfs[i],
                                                 c1d,
                                                 nzeta,
                                                 ibands[i])
                spill += out
            return spill/nconfs
        
        compulsory = {'method': options.get('method', 'swats'),
                      'maxiter': options.get('maxiter', 5000),
                      'disp': options.get('disp', False),
                      'ndisp': options.get('ndisp', 10)}
        optional = {k: v for k, v in options.items() if k not in compulsory}
        return minimize(_t_f, 
                        coef_init, 
                        **compulsory,
                        **optional)

class SpillTorch_jy(SpillTorch):
    
    def config_add(self, outdir, weight=(0.0, 1.0)):
        raw = _t_jy_data_extract(outdir)
        C = raw['C'] # torch.Tensor
        S = raw['S'] # torch.Tensor
        T = raw['T'] # torch.Tensor

        wov, wop = weight
        ref_ov_ref = th.sum(C.conj() * (S @ C), -2) # <ref|S|ref>
        ref_op_ref = th.sum(C.conj() * (T @ C), -2) # <ref|T|ref>

        ref_ov_jy = C.swapaxes(-2, -1).conj() @ S # @ I = <ref|S|jy>
        ref_op_jy = C.swapaxes(-2, -1).conj() @ T # @ I = <ref|T|jy>

        ref_ref = th.Tensor(np.array([ref_ov_ref, wov*ref_ov_ref + wop*ref_op_ref]))
        ref_jy  = np.array([ref_ov_jy, wov*ref_ov_jy + wop*ref_op_jy])
        jy_jy   = np.array([S, wov*S + wop*T])

        # permute the zeta and m in ref_jy and jy_jy
        p = perm_zeta_m(_lin2comp(raw['natom'], nzeta=raw['nzeta']))
        ref_jy = th.Tensor(ref_jy[:,:,:,p].copy())
        jy_jy = th.Tensor(jy_jy[:,:,:,p][:,:,p,:].copy())

        self.config.append({
            'natom': raw['natom'],
            'nbes': raw['nzeta'],
            'wk': raw['wk'],
            'ref_ref': ref_ref,
            'ref_jy': ref_jy,
            'jy_jy': jy_jy
        })

class SpillTorch_pw(SpillTorch):

    def __init__(self):
        raise NotImplementedError('SpillTorch_pw is not implemented yet')


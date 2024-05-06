from datparse import read_orb_mat, _assert_consistency
from radial import jl_reduce, jl_raw_norm
from listmanip import flatten, nest, nestpat
from jlzeros import JLZEROS

import numpy as np
from scipy.optimize import minimize

from copy import deepcopy
import time

def _mrdiv(X, Y):
    '''
    Right matrix division.

    Given two 3-d arrays X and Y, returns a 3-d array Z such that

        Z[k] = X[k] @ inv(Y[k])

    '''
    # TODO explore the possibility of using scipy.linalg.solve with assume_a='sym'
    assert len(X.shape) == 3 and len(Y.shape) == 3
    return np.array([np.linalg.solve(Yk.T, Xk.T).T for Xk, Yk in zip(X, Y)])


def _rfrob(X, Y, rowwise=False):
    '''
    Real part of the Frobenius inner product.

    The Frobenius inner product between two matrices or vectors is defined as

        <X, Y> \equiv Tr(X @ Y.T.conj()) = (X * Y.conj()).sum()

    X and Y must have shapes compatible with element-wise multiplication. If
    their dimensions are 3 or more, the inner product is computed slice-wise,
    i.e., sum() is taken over the last two axes. If rowwise is True, sum() is
    taken over the last axis only.

    Notes
    -----
    The inner product is assumed to have the Hermitian conjugate on the
    second argument, not the first.

    '''
    return (X * Y.conj()).real.sum(-1 if rowwise else (-2,-1))


def _jy2ao(coef, lin2comp, nbes, rcut):
    '''
    Basis transformation matrix from a Bessel basis to a pseudo-atomic
    orbital basis.

    This function constructs the transformation matrix from some Bessel
    basis ([some Bessel radial] x [spherical harmonics]) arranged in the
    lexicographic order of (itype, iatom, l, m, q) (q being the index
    of Bessel radial functions) to some pseudo-atomic orbital basis
    arranged in the lexicographic order of (itype, iatom, l, m, zeta).
    The entire transformation matrix is block-diagonal, with each block
    corresponding to a specific q->zeta.

    Parameters
    ----------
        coef : nested list
            The coefficients of pseudo-atomic orbital basis orbitals
            in terms of the Bessel basis. coef[itype][l][zeta] gives
            a list of coefficients that specifies an orbital.
            Note that the length of this coefficient list is allowed to
            be smaller than nbes; the list will be padded with zeros.
        lin2comp : dict
            linear-to-composite index map (not including q):

                    mu -> (itype, iatom, l, 0, m).

            NOTE: zeta is supposed to be always 0 in this function.
        nbes : int
            Number of Bessel basis functions.
        rcut : float
            Cutoff radius.

    '''
    from scipy.linalg import block_diag

    def _gen_q2zeta(coef, lin2comp, nbes, rcut):
        for mu in lin2comp:
            itype, _, l, _, _ = lin2comp[mu]
            if l >= len(coef[itype]) or len(coef[itype][l]) == 0:
                # The generator should yield a zero matrix with the
                # appropriate size when no coefficient is provided.
                yield np.zeros((nbes, 0))
            else:
                C = np.zeros((nbes, len(coef[itype][l])))
                C[:len(coef[itype][l][0])] = np.array(coef[itype][l]).T
                yield C

    return block_diag(*_gen_q2zeta(coef, lin2comp, nbes, rcut))


def _overlap_spillage(ovlp, coef, ibands, coef_frozen=None):
    '''
    Standard spillage function (overlap spillage).

    Note
    ----
    This function is not supposed to be used in the optimization.
    As a special case of the generalized spillage (op = I), it serves
    as a cross-check for the implementation of the generalized spillage.

    '''
    spill = (ovlp['wk'] @ ovlp['mo_mo'][:,ibands]).real.sum()

    mo_jy = ovlp['mo_jy'][:,ibands,:]
    jy2ao = _jy2ao(coef, ovlp['lin2comp'], ovlp['nbes'], ovlp['rcut'])

    V = mo_jy @ jy2ao
    W = jy2ao.T @ ovlp['jy_jy'] @ jy2ao

    if coef_frozen is not None:
        jy2frozen = _jy2ao(coef_frozen, ovlp['lin2comp'], ovlp['nbes'], ovlp['rcut'])

        X = mo_jy @ jy2frozen
        S = jy2frozen.T @ ovlp['jy_jy'] @ jy2frozen
        X_dual = _mrdiv(X, S)

        spill -= ovlp['wk'] @ _rfrob(X_dual, X)

        V -= X_dual @ jy2frozen.T @ ovlp['jy_jy'] @ jy2ao

    spill -= ovlp['wk'] @ _rfrob(_mrdiv(V, W), V)

    return spill / len(ibands)


def initgen(nzeta, ov, reduced=True):
    '''
    Generate an initial guess from the single-atom overlap data
    for the spillage optimization.

    '''
    assert ov['ntype'] == 1 and ov['natom'][0] == 1 and ov['ecutwfc'] <= ov['ecutjlq']

    lmax = len(nzeta) - 1
    assert lmax <= ov['lmax'][0]

    rcut = ov['rcut']
    ecut = ov['ecutwfc']

    # Finds the number of truncated spherical Bessel functions whose energy
    # is smaller than ecut. This is based on the following:
    #
    # 1. The kinetic energy of a normalized truncated spherical Bessel function
    #    j_l(k*r) * Y_{lm}(r) is k^2
    #
    # 2. The wavenumbers of truncated spherical Bessel functions are chosen such
    #    that the function is zero at rcut, i.e., JLZEROS/rcut

    # make sure the tabulated JLZEROS is sufficient
    assert(all((JLZEROS[l][-1]/rcut)**2 > ecut for l in range(lmax+1)))

    # the number of truncated spherical Bessel functions whose energy is smaller
    # than ecut for each l
    nbes_ecut = [sum((JLZEROS[l]/rcut)**2 < ecut) for l in range(lmax+1)]
    if reduced:
        nbes_ecut = [n - 1 for n in nbes_ecut]

    assert all(n > 0 for n in nbes_ecut)

    # <mo|jy> reshaped to [nk, nbands, nao, nbes]
    Y = ov['mo_jy'].reshape(ov['nk'], ov['nbands'], -1, ov['nbes'])
    comp2lin = ov['comp2lin']

    coef = []
    for l in range(lmax+1):
        idx_start = comp2lin[(0, 0, l, 0, 0)]
        Yl = Y[:, :, idx_start:idx_start+2*l+1, :].reshape(ov['nk'], -1, ov['nbes'])
        Yl = Yl[:,:,nbes_ecut[l]]
        YdaggerY = (ov['wk'].reshape(-1,1,1) * (Yl.transpose((0, 2, 1)).conj() @ Yl)) \
                   .sum(0).real
        vec = np.linalg.eigh(YdaggerY)[1]

        # eigenvectors corresponding to the largest nzeta eigenvalues
        coef.append(vec[:,-nzeta[l]:][:,::-1].T.tolist())

    return [np.linalg.qr(np.array(coef_l).T)[0].T.tolist() for coef_l in coef]


class Spillage:
    '''
    Generalized spillage function and its optimization.

    Attributes
    ----------
        reduced: bool
            If true, the optimization is performed in the end-smoothed mixed
            spherical Bessel basis; otherwise in the normalized truncated
            spherical Bessel basis.
        config : list
            A list of dict. Each dict contains the data for a geometric
            configuration, including both the overlap and operator matrix
            elements.
            The overlap and operator data are read from orb_matrix.0.dat
            and orb_matrix.1.dat respectively. Before appending to config,
            the two datasets are subject to a consistency check, after which
            a new one consisting of the common part of overlap and operator
            data plus the stacked matrix data are appended to config.
            NOTE: this behavior may be subject to change in the future.
        rcut : float
            Cutoff radius. So far only one rcut is allowed throughout the
            entire dataset.
        spill_frozen : list
            The band-wise spillage contribution from frozen orbitals.
        mo_Pfrozen_jy : list
            <mo|P_frozen|jy> and <mo|P_frozen op|jy> for each configuration,
            where P_frozen is the projection operator onto the frozen subspace.
        mo_Qfrozen_dao : list
            The derivatives of <mo|Q_frozen|ao> and <mo|Q_frozen op|ao> w.r.t.
            the coefficients for each configuration, where Q_frozen is the
            projection operator onto the complement of the frozen subspace.
        dao_jy : list
            The derivatives of <ao|jy> and <ao|op|jy> w.r.t. the coefficients
            for each configuration.

    '''

    def __init__(self, reduced=True):
        self.reset()
        self.reduced = reduced

    def reset(self):
        self.config = []
        self.rcut = None

        self._reset_frozen()
        self._reset_deriv()


    def _reset_frozen(self):
        self.spill_frozen = None
        self.mo_Pfrozen_jy = None


    def _reset_deriv(self):
        self.mo_Qfrozen_dao = []
        self.dao_jy = []


    def add_config(self, ovlp_dat, op_dat):
        '''
        '''
        # The overlap and operator data must be consistent except
        # for their matrix data (mo_mo, mo_jy and jy_jy).
        _assert_consistency(ovlp_dat, op_dat)

        # The dict to append to config is a new one consisting of
        # the common part of ovlp & op data plus their stacked
        # matrix data.
        dat = deepcopy(ovlp_dat)

        ntype = ovlp_dat['ntype']
        lmax = ovlp_dat['lmax']
        nbes = ovlp_dat['nbes']
        rcut = ovlp_dat['rcut']

        if self.reduced:
            # truncated spherical Bessel to end-smoothed mixed spherical Bessel
            coef = [[jl_reduce(l, nbes, rcut).T.tolist()
                     for l in range(lmax[itype]+1)]
                    for itype in range(ntype)]
            dat['nbes'] -= 1
        else:
            # truncated spherical Bessel to normalized truncated spherical Bessel
            uvec = lambda v, k, n: [v if i == k else 0 for i in range(n)]
            coef = [[[uvec(1. / jl_raw_norm(l, q, rcut), q, nbes)
                      for q in range(nbes)]
                     for l in range(lmax[itype]+1)]
                    for itype in range(ntype)]

        C = _jy2ao(coef, ovlp_dat['lin2comp'], nbes, rcut)

        dat['mo_mo'] = np.array([ovlp_dat['mo_mo'], op_dat['mo_mo']])
        dat['mo_jy'] = np.array([ovlp_dat['mo_jy'] @ C, op_dat['mo_jy'] @ C])
        dat['jy_jy'] = np.array([C.T @ ovlp_dat['jy_jy'] @ C, C.T @ op_dat['jy_jy'] @ C])

        self.config.append(dat)

        # NOTE currently a dataset merely contains one atom type
        # and one rcut. This may change in the future.
        if self.rcut is None:
            self.rcut = ovlp_dat['rcut']
        else:
            assert self.rcut == ovlp_dat['rcut']


    def _tab_frozen(self, coef_frozen):
        '''
        Tabulates for each configuration the band-wise spillage contribution
        from frozen orbitals and

                            <mo|P_frozen   |jy>
                            <mo|P_frozen op|jy>

        where P_frozen is the projection operator onto the frozen subspace:

                        P_frozen = |frozen_dual><frozen|

        '''
        # jy -> frozen orbital transformation matrices
        jy2frozen = [_jy2ao(coef_frozen, dat['lin2comp'], dat['nbes'], dat['rcut'])
                     for dat in self.config]

        frozen_frozen = [jy2froz.T @ dat['jy_jy'] @ jy2froz
                         for dat, jy2froz in zip(self.config, jy2frozen)]

        mo_frozen = [dat['mo_jy'] @ jy2froz
                     for dat, jy2froz in zip(self.config, jy2frozen)]

        # <mo|frozen_dual> only; no need to compute <mo|op|frozen_dual>
        mo_frozen_dual = [_mrdiv(mo_froz[0], froz_froz[0])
                          for mo_froz, froz_froz in zip(mo_frozen, frozen_frozen)]

        # for each config, indexed as [0/1][k][mo][jy]
        self.mo_Pfrozen_jy = [mo_froz_dual @ jy2froz.T @ dat['jy_jy']
                              for mo_froz_dual, dat, jy2froz in
                              zip(mo_frozen_dual, self.config, jy2frozen)]

        self.spill_frozen = [_rfrob(mo_froz_dual @ froz_froz[1], mo_froz_dual, rowwise=True)
                             - 2.0 * _rfrob(mo_froz_dual, mo_froz[1], rowwise=True)
                             for mo_froz_dual, mo_froz, froz_froz in
                             zip(mo_frozen_dual, mo_frozen, frozen_frozen)]

        # weighted sum over k
        self.spill_frozen = [dat['wk'] @ spill_froz
                             for dat, spill_froz in zip(self.config, self.spill_frozen)]


    def _tab_deriv(self, coef):
        '''
        Tabulates for each configuration the derivatives of

                                <ao|jy>
                                <ao|op|jy>

                            <mo|Q_frozen   |ao>
                            <mo|Q_frozen op|ao>

        with respect to the coefficients that specifies |ao>, where Q_frozen
        is the projection operator onto the complement of the frozen subspace:

                        Q_frozen = 1 - |frozen_dual><frozen|

        (Q_frozen = 1 if there is no frozen orbitals)


        Note
        ----
        The only useful information of coef is its nesting pattern, which
        determines what derivatives to compute.

        '''
        # jy -> (d/dcoef)ao transformation matrices
        jy2dao_all = [[_jy2ao(nest(ci.tolist(), nestpat(coef)),
                              dat['lin2comp'], dat['nbes'], dat['rcut'])
                       for ci in np.eye(len(flatten(coef)))]
                      for dat in self.config]

        # derivatives of <ao|jy>, indexed as [0/1][deriv][k][ao][jy] for each config
        self.dao_jy = [np.array([jy2dao_i.T @ dat['jy_jy'] for jy2dao_i in jy2dao])
                       .transpose(1,0,2,3,4)
                       for dat, jy2dao in zip(self.config, jy2dao_all)]

        # derivatives of <mo|ao> and <mo|op|ao>
        self.mo_Qfrozen_dao = [np.array([dat['mo_jy'] @ jy2dao_i for jy2dao_i in jy2dao])
                               for dat, jy2dao in zip(self.config, jy2dao_all)]
        # at this stage, the index for each config follows [deriv][0/1][k][mo][ao]
        # where 0->overlap; 1->operator

        if self.spill_frozen is not None:
            # if frozen orbitals are present, subtract from the previous results
            # <mo|P_frozen|ao> and <mo|P_frozen op|ao>
            self.mo_Qfrozen_dao = [mo_Qfroz_dao -
                                   np.array([mo_Pfroz_jy @ jy2dao_i for jy2dao_i in jy2dao])
                                   for mo_Qfroz_dao, mo_Pfroz_jy, jy2dao in
                                   zip(self.mo_Qfrozen_dao, self.mo_Pfrozen_jy, jy2dao_all)]

        # transpose to [0/1][deriv][k][mo][ao]
        self.mo_Qfrozen_dao = [dV.transpose(1,0,2,3,4) for dV in self.mo_Qfrozen_dao]



    def _generalize_spillage(self, iconf, coef, ibands, with_grad=False):
        '''
        Generalized spillage function and its gradient.

        '''
        dat = self.config[iconf]

        spill = (dat['wk'] @ dat['mo_mo'][1][:,ibands]).real.sum()

        # jy->ao basis transformation matrix
        jy2ao = _jy2ao(coef, dat['lin2comp'], dat['nbes'], dat['rcut'])

        # <mo|Q_frozen|ao> and <mo|Q_frozen op|ao>
        V = dat['mo_jy'][:,:,ibands,:] @ jy2ao
        if self.spill_frozen is not None:
            V -= self.mo_Pfrozen_jy[iconf][:,:,ibands,:] @ jy2ao
            spill += self.spill_frozen[iconf][ibands].sum()

        # <ao|ao> and <ao|op|ao>
        W = jy2ao.T @ dat['jy_jy'] @ jy2ao

        V_dual = _mrdiv(V[0], W[0]) # overlap only; no need for op
        VdaggerV = V_dual.transpose((0,2,1)).conj() @ V_dual

        spill += dat['wk'] @ (_rfrob(W[1], VdaggerV) - 2.0 * _rfrob(V_dual, V[1]))
        spill /= len(ibands)

        if with_grad:
            # (d/dcoef)<ao|ao> and (d/dcoef)<ao|op|ao>
            dW = self.dao_jy[iconf] @ jy2ao
            dW += dW.transpose((0,1,2,4,3)).conj()

            # (d/dcoef)<mo|Q_frozen|ao> and (d/dcoef)<mo|Q_frozen op|ao>
            dV = self.mo_Qfrozen_dao[iconf][:,:,:,ibands,:]

            grad = (_rfrob(dW[1], VdaggerV)
                    - 2.0 * _rfrob(V_dual, dV[1])
                    + 2.0 * _rfrob(dV[0] - V_dual @ dW[0],
                                   _mrdiv(V_dual @ W[1] - V[1], W[0]))
                    ) @ dat['wk']

            grad /= len(ibands)
            grad = nest(grad.tolist(), nestpat(coef))

        return (spill, grad) if with_grad else spill


    def opt(self, coef_init, coef_frozen, iconfs, ibands, options, nthreads=1):
        '''
        Spillage minimization w.r.t. end-smoothed mixed spherical Bessel coefficients.

        Parameters
        ----------
            coef_init : nested list
                Initial guess for the coefficients.
            coef_frozen : nested list
                Coefficients for the frozen orbitals.
            iconfs : list of int or 'all'
                List of configuration indices to be included in the optimization.
                If 'all', all configurations are included.
            ibands : range/tuple or list of range/tuple
                Band indices to be included in the spillage calculation. If a range
                or tuple is given, the same indices are used for all configurations.
                If a list of range/tuple is given, each range/tuple will be applied
                to the configuration specified by iconfs respectively.
            options : dict
                Options for the optimization.
            nthreads : int
                Number of threads for config-level parallellization.

        '''
        from multiprocessing.pool import ThreadPool
        pool = ThreadPool(nthreads)

        if coef_frozen is not None:
            self._tab_frozen(coef_frozen)

        self._tab_deriv(coef_init)

        iconfs = range(len(self.config)) if iconfs == 'all' else iconfs
        nconf = len(iconfs)

        ibands = [ibands] * nconf if not isinstance(ibands, list) else ibands
        assert len(ibands) == nconf

        pat = nestpat(coef_init)
        def f(c): # function to be minimized
            s = lambda i: self._generalize_spillage(iconfs[i], nest(c.tolist(), pat),
                                                    ibands[i], with_grad=True)
            spills, grads = zip(*pool.map(s, range(nconf)))
            return (sum(spills) / nconf, sum(np.array(flatten(g)) for g in grads) / nconf)

        c0 = np.array(flatten(coef_init))

        # Restricts the coefficients to [-1, 1] for better numerical stability
        # FIXME Is this necessary?
        bounds = [(-1.0, 1.0) for _ in c0]
        #bounds = None

        res = minimize(f, c0, jac=True, method='L-BFGS-B',
                       bounds=bounds, options=options)

        pool.close()

        coef_opt = nest(res.x.tolist(), pat)
        return [[np.linalg.qr(np.array(coef_tl).T)[0].T.tolist()
                 for coef_tl in coef_t] for coef_t in coef_opt]


############################################################
#                           Test
############################################################
import unittest

class _TestSpillage(unittest.TestCase):

    def setUp(self):
        self.orbgen_reduced = Spillage(True)
        self.orbgen_raw = Spillage(False)

        self.datadir = './testfiles/orb_matrix/'
        self.config = ['Si-dimer-1.8', 'Si-dimer-2.8', 'Si-dimer-3.8',
                       'Si-trimer-1.7', 'Si-trimer-2.7',
                       ]


    def est_initgen(self):
        ov = read_orb_mat('../../tmp/Si-single-atom/orb_matrix.0.dat')

        nzeta = [2,2,1]
        coef = initgen(nzeta, ov, reduce=False)
        assert(len(coef) == len(nzeta))
        assert([len(coef[l]) for l in range(len(nzeta))] == nzeta)

        from radial import build_reduced
        import matplotlib.pyplot as plt

        rcut = ov['rcut']
        dr = 0.01
        r = np.linspace(0, rcut, int(rcut/dr)+1)
        chi = build_reduced(coef, rcut, r, True, True)


        lmax = len(chi)-1
        nzeta = [len(chi_l) for chi_l in chi]
        nzetamax = max(nzeta)
        chimax = np.max([np.max(np.abs(chi_l)) for chi_l in chi])

        fig, ax = plt.subplots(nzetamax, lmax+1, figsize=((lmax+1)*6, nzetamax*5),
                               layout='tight', squeeze=False)

        for l in range(lmax+1):
            for zeta in range(nzeta[l]):
                # adjust the sign so that the largest value is positive
                if chi[l][zeta][np.argmax(np.abs(chi[l][zeta]))] < 0:
                    chi[l][zeta] *= -1

                ax[zeta, l].plot(r, chi[l][zeta])
                #ax[zeta, l].plot(r, (r*nao['chi'][l][zeta])**2)

                ax[zeta, l].axhline(0, color='black', linestyle=':')
                #ax[zeta, l].axvline(2.0, color='black', linestyle=':')

                # title
                ax[zeta, l].set_title('l=%d, zeta=%d' % (l, zeta), fontsize=20)

                ax[zeta, l].set_xlim([0, rcut])
                ax[zeta, l].set_ylim([-0.4*chimax, chimax*1.1])

        plt.show()


    def test_mrdiv(self):
        n_slice = 3
        m = 5
        n = 6

        # make each slice of S unitary to make it easier to verify
        Y = np.random.randn(n_slice, n, n) + 1j * np.random.randn(n_slice, n, n)
        Y = np.linalg.qr(Y)[0]

        X = np.random.randn(n_slice, m, n) + 1j * np.random.randn(n_slice, m, n)
        Z = _mrdiv(X, Y)

        self.assertEqual(Z.shape, X.shape)
        for i in range(n_slice):
            self.assertTrue( np.allclose(Z[i], X[i] @ Y[i].T.conj()) )


    def test_rfrob(self):
        n_slice = 5
        m = 3
        n = 4
        w = np.random.randn(n_slice)
        X = np.random.randn(n_slice, m, n) + 1j * np.random.randn(n_slice, m, n)
        Y = np.random.randn(n_slice, m, n) + 1j * np.random.randn(n_slice, m, n)

        wsum = 0.0
        for wk, Xk, Yk in zip(w, X, Y):
            wsum += wk * np.trace(Xk @ Yk.T.conj()).sum()

        self.assertAlmostEqual(w @ _rfrob(X, Y), wsum.real)

        wsum = np.zeros(m, dtype=complex)
        for i in range(m):
            for k in range(n_slice):
                wsum[i] += w[k] * (X[k,i] @ Y[k,i].T.conj())

        self.assertTrue( np.allclose(w @ _rfrob(X, Y, rowwise=True), wsum.real) )


    def test_jy2ao(self):
        from indexmap import _index_map

        ntype = 3
        natom = [1, 2, 3]
        lmax = [2, 1, 0]
        nzeta = [[1, 1, 1], [2, 2], [3]]
        _, lin2comp = _index_map(ntype, natom, lmax, nzeta)

        nbes = 5
        rcut = 6.0

        coef = [ [np.random.randn(nzeta[itype][l], nbes).tolist()
                  for l in range(lmax[itype]+1)]
                for itype in range(ntype) ]

        M = _jy2ao(coef, lin2comp, nbes, rcut)

        icol = 0
        for mu, (itype, iatom, l, _, m) in lin2comp.items():
            nzeta = len(coef[itype][l])
            self.assertTrue(np.allclose(\
                    M[mu*nbes:(mu+1)*nbes, icol:icol+nzeta], \
                    np.array(coef[itype][l]).T))
            icol += nzeta


    def test_add_config(self):
        for iconf, config in enumerate(self.config):
            for orbgen in [self.orbgen_raw, self.orbgen_reduced]:
                ov = read_orb_mat(self.datadir + config + '/orb_matrix.0.dat')
                op = read_orb_mat(self.datadir + config + '/orb_matrix.1.dat')

                orbgen.add_config(ov, op)
                dat = orbgen.config[iconf]
                njy = len(dat['lin2comp']) * dat['nbes']

                self.assertEqual(len(orbgen.config), iconf+1)
                self.assertEqual(dat['mo_mo'].shape, (2, ov['nk'], ov['nbands']))
                self.assertEqual(dat['mo_jy'].shape, (2, ov['nk'], ov['nbands'], njy))
                self.assertEqual(dat['jy_jy'].shape, (2, ov['nk'], njy, njy))

                nbes = dat['nbes']
                S = np.diag(dat['jy_jy'][0, 0])
                T = np.diag(dat['jy_jy'][1, 0])

                self.assertLess(np.linalg.norm(S - np.ones(S.shape)), 1e-6)

                if orbgen == self.orbgen_raw:
                    T_ref_0 = (JLZEROS[0][:nbes]/dat['rcut'])**2
                    self.assertLess(np.linalg.norm(T[:nbes] - T_ref_0), 1e-2)

                    T_ref_1 = (JLZEROS[1][:nbes]/dat['rcut'])**2
                    self.assertLess(np.linalg.norm(T[nbes:2*nbes] - T_ref_1), 1e-2)

            self.assertEqual(self.orbgen_raw.config[iconf]['nbes'],
                             self.orbgen_reduced.config[iconf]['nbes'] + 1)


    def test_tab_frozen(self):
        for iconf, config in enumerate(self.config):
            ov = read_orb_mat(self.datadir + config + '/orb_matrix.0.dat')
            op = read_orb_mat(self.datadir + config + '/orb_matrix.1.dat')
            self.orbgen_raw.add_config(ov, op)
            self.orbgen_reduced.add_config(ov, op)

        nzeta = [2, 1, 0]
        lmax = len(nzeta) - 1
        nbes = min(dat['nbes'] for dat in self.orbgen_reduced.config)

        coef_frozen = [[np.eye(nzeta[l], nbes).tolist() for l in range(lmax+1)]]

        for orbgen in [self.orbgen_raw, self.orbgen_reduced]:
            orbgen._tab_frozen(coef_frozen)
            self.assertEqual(len(orbgen.mo_Pfrozen_jy), len(self.config))
            self.assertEqual(len(orbgen.spill_frozen), len(self.config))

            for iconf, config in enumerate(self.config):
                dat = orbgen.config[iconf]
                njy = len(dat['lin2comp']) * dat['nbes']
                self.assertEqual(orbgen.spill_frozen[iconf].shape,
                                 (dat['nbands'],))
                self.assertEqual(orbgen.mo_Pfrozen_jy[iconf].shape,
                                 (2, dat['nk'], dat['nbands'], njy))


    def test_tab_deriv(self):
        for iconf, config in enumerate(self.config):
            ov = read_orb_mat(self.datadir + config + '/orb_matrix.0.dat')
            op = read_orb_mat(self.datadir + config + '/orb_matrix.1.dat')
            self.orbgen_raw.add_config(ov, op)
            self.orbgen_reduced.add_config(ov, op)

        nzeta = [2, 1, 0]
        lmax = len(nzeta) - 1
        nbes = min(dat['nbes'] for dat in self.orbgen_reduced.config)

        coef = [[np.random.randn(nzeta[l], nbes).tolist() for l in range(lmax + 1)]]

        ncoef = len(flatten(coef))
        nao_type = [sum(len(coef_tl) * (2*l+1) for l, coef_tl in enumerate(coef_t))
                    for coef_t in coef]

        for orbgen in [self.orbgen_raw, self.orbgen_reduced]:
            orbgen._tab_deriv(coef)
            self.assertEqual(len(orbgen.dao_jy), len(self.config))
            self.assertEqual(len(orbgen.mo_Qfrozen_dao), len(self.config))

            for iconf, config in enumerate(self.config):
                dat = orbgen.config[iconf]
                njy = len(dat['lin2comp']) * dat['nbes']
                nao = sum(nao*natom for nao, natom in zip(nao_type, dat['natom']))
                self.assertEqual(orbgen.dao_jy[iconf].shape,
                                 (2, ncoef, dat['nk'], nao, njy))


    def test_overlap_spillage(self):
        '''
        Verifies that the generalized spillage with op=I recovers the overlap spillage

        '''
        ibands = range(5)
        #FIXME for this and the next test, the frozen & raw part might be put together
        # to reduced duplicate code
        for orbgen in [self.orbgen_raw, self.orbgen_reduced]:
            for iconf, config in enumerate(self.config):
                ov = read_orb_mat(self.datadir + config + '/orb_matrix.0.dat')
                orbgen.add_config(ov, ov)

                nbes = orbgen.config[iconf]['nbes']
                nzeta = [2, 2, 1]
                lmax = len(nzeta) - 1
                coef = [[np.random.randn(nzeta[l], nbes).tolist() for l in range(lmax + 1)]]

                ov['nbes'] = nbes
                ov['mo_jy'] = orbgen.config[iconf]['mo_jy'][0]
                ov['jy_jy'] = orbgen.config[iconf]['jy_jy'][0]

                spill_ref = _overlap_spillage(ov, coef, ibands) 
                spill = orbgen._generalize_spillage(iconf, coef, ibands, False)
                self.assertAlmostEqual(spill, spill_ref, places=10)

            nbes = min(dat['nbes'] for dat in orbgen.config)
            nzeta_frozen = [2, 1, 0]
            lmax = len(nzeta_frozen) - 1
            coef_frozen = [[np.random.randn(nzeta_frozen[l], nbes).tolist() for l in range(lmax + 1)]]

            orbgen._tab_frozen(coef_frozen)

            for iconf, config in enumerate(self.config):
                ov = read_orb_mat(self.datadir + config + '/orb_matrix.0.dat')

                nbes = orbgen.config[iconf]['nbes']
                nzeta = [2, 2, 1]
                lmax = len(nzeta) - 1
                coef = [[np.random.randn(nzeta[l], nbes).tolist() for l in range(lmax + 1)]]

                ov['nbes'] = nbes
                ov['mo_jy'] = orbgen.config[iconf]['mo_jy'][0]
                ov['jy_jy'] = orbgen.config[iconf]['jy_jy'][0]

                spill_ref = _overlap_spillage(ov, coef, ibands, coef_frozen) 
                spill = orbgen._generalize_spillage(iconf, coef, ibands, False)
                self.assertAlmostEqual(spill, spill_ref, places=10)


    def test_finite_difference(self):
        for orbgen in [self.orbgen_raw, self.orbgen_reduced]:
            for iconf, config in enumerate(self.config):
                ov = read_orb_mat(self.datadir + config + '/orb_matrix.0.dat')
                op = read_orb_mat(self.datadir + config + '/orb_matrix.1.dat')
                orbgen.add_config(ov, op)

            nbes = min(dat['nbes'] for dat in orbgen.config)

            nzeta = [1, 2, 1]
            lmax = len(nzeta) - 1
            ibands = range(6)
            coef = [[np.random.randn(nzeta[l], nbes).tolist() for l in range(lmax + 1)]]

            orbgen._tab_deriv(coef)

            for iconf, dat in enumerate(orbgen.config):

                dspill = orbgen._generalize_spillage(iconf, coef, ibands, True)[1]
                dspill = np.array(flatten(dspill))

                pat = nestpat(coef)
                sz = len(flatten(coef))

                dspill_fd = np.zeros(sz)
                dc = 1e-6
                for i in range(sz):
                    coef_p = flatten(deepcopy(coef))
                    coef_p[i] += dc
                    coef_p = nest(coef_p, pat)
                    spill_p = orbgen._generalize_spillage(iconf, coef_p, ibands, False)

                    coef_m = flatten(deepcopy(coef))
                    coef_m[i] -= dc
                    coef_m = nest(coef_m, pat)
                    spill_m = orbgen._generalize_spillage(iconf, coef_m, ibands, False)

                    dspill_fd[i] = (spill_p - spill_m) / (2 * dc)

                self.assertTrue(np.allclose(dspill, dspill_fd, atol=1e-7))


            nzeta_frozen = [2, 2, 1]
            lmax_frozen = len(nzeta_frozen) - 1
            coef_frozen = [[np.random.randn(nzeta_frozen[l], nbes).tolist()
                            for l in range(lmax_frozen + 1)]]

            orbgen._tab_frozen(coef_frozen)
            orbgen._tab_deriv(coef)

            for iconf, dat in enumerate(orbgen.config):
                dspill = orbgen._generalize_spillage(iconf, coef, ibands, True)[1]
                dspill = np.array(flatten(dspill))

                pat = nestpat(coef)
                sz = len(flatten(coef))

                dspill_fd = np.zeros(sz)
                dc = 1e-6
                for i in range(sz):
                    coef_p = flatten(deepcopy(coef))
                    coef_p[i] += dc
                    coef_p = nest(coef_p, pat)
                    spill_p = orbgen._generalize_spillage(iconf, coef_p, ibands, False)

                    coef_m = flatten(deepcopy(coef))
                    coef_m[i] -= dc
                    coef_m = nest(coef_m, pat)
                    spill_m = orbgen._generalize_spillage(iconf, coef_m, ibands, False)

                    dspill_fd[i] = (spill_p - spill_m) / (2 * dc)

                self.assertTrue(np.allclose(dspill, dspill_fd, atol=1e-7))


    def est_opt(self):
        nbes_min = 1000
        for iconf, config in enumerate(self.config):
            ov = read_orb_mat(self.datadir + config + '/orb_matrix.0.dat')
            op = read_orb_mat(self.datadir + config + '/orb_matrix.1.dat')
            self.orbgen.add_config(ov, op)
            nbes_min = min(nbes_min, ov['nbes'])

        coef_frozen = [[np.random.randn(1, nbes_min-1).tolist(),
                        np.random.randn(2, nbes_min-1).tolist(),
                        np.random.randn(1, nbes_min-1).tolist()]]

        coef0 = [[np.random.randn(1, nbes_min-1).tolist(),
                  np.random.randn(2, nbes_min-1).tolist(),
                  np.random.randn(1, nbes_min-1).tolist()]]

        ibands = range(6)

        self.orbgen._tab_frozen(coef_frozen)
        self.orbgen._tab_deriv(coef0)

        nthreads = 4
        options = {'maxiter': 5, 'disp': False, 'maxcor': 20}

        # use all configs and all bands
        coef_opt = self.orbgen.opt(coef0, coef_frozen, 'all', ibands, options, nthreads)

        # selected configs and bands
        iconfs = [1, 2]
        ibands = [range(4), range(6)]
        coef_opt = self.orbgen.opt(coef0, coef_frozen, iconfs, ibands, options, nthreads)

        pass


    def est_orbgen(self):
        nbes_min = 1000
        config = ['Si-dimer-1.62', 'Si-dimer-1.82', 'Si-dimer-2.22', 'Si-dimer-2.72', 'Si-dimer-3.22']
        datadir = '/home/zuxin/abacus-community/abacus_orbital_generation/tmp/'
        for iconf, config in enumerate(config):
            ov = read_orb_mat(datadir + config + '/orb_matrix.0.dat')
            op = read_orb_mat(datadir + config + '/orb_matrix.1.dat')
            self.orbgen.add_config(ov, op)
            nbes_min = min(nbes_min, ov['nbes'])

        #coef0 = [[np.random.randn(1, nbes_min-1).tolist(),
        #          np.random.randn(1, nbes_min-1).tolist(),
        #          ]]
        coef0 = [[np.eye(1, nbes_min-1).tolist(),
                  np.eye(1, nbes_min-1).tolist(),
                  ]]

        ibands = range(4)

        self.orbgen._tab_deriv(coef0)

        nthreads = 5
        options = {'maxiter': 1000, 'disp': True, 'maxcor': 20}

        # use all configs and all bands
        coef_opt = self.orbgen.opt(coef0, None, 'all', ibands, options, nthreads)

        from radial import build_reduced
        import matplotlib.pyplot as plt

        rcut = ov['rcut']
        dr = 0.01
        r = np.linspace(0, rcut, int(rcut/dr)+1)
        chi = build_reduced(coef_opt[0], rcut, r, True, True)


        lmax = len(chi)-1
        nzeta = [len(chi_l) for chi_l in chi]
        nzetamax = max(nzeta)
        chimax = np.max([np.max(np.abs(chi_l)) for chi_l in chi])

        fig, ax = plt.subplots(nzetamax, lmax+1, figsize=((lmax+1)*6, nzetamax*5),
                               layout='tight', squeeze=False)

        for l in range(lmax+1):
            for zeta in range(nzeta[l]):
                # adjust the sign so that the largest value is positive
                if chi[l][zeta][np.argmax(np.abs(chi[l][zeta]))] < 0:
                    chi[l][zeta] *= -1

                ax[zeta, l].plot(r, chi[l][zeta])
                #ax[zeta, l].plot(r, (r*nao['chi'][l][zeta])**2)

                ax[zeta, l].axhline(0, color='black', linestyle=':')
                #ax[zeta, l].axvline(2.0, color='black', linestyle=':')

                # title
                ax[zeta, l].set_title('l=%d, zeta=%d' % (l, zeta), fontsize=20)

                ax[zeta, l].set_xlim([0, rcut])
                ax[zeta, l].set_ylim([-0.4*chimax, chimax*1.1])

        plt.show()

if __name__ == '__main__':
    unittest.main()


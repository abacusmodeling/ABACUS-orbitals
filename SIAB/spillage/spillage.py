from datparse import read_orb_mat, _assert_consistency
from radial import jl_reduce, JL_REDUCE
from listmanip import flatten, nest, nestpat

import numpy as np
from scipy.optimize import minimize

def _mrdiv(X, S):
    '''
    Right matrix division.

    Given two 3-d arrays X and S, returns a 3-d array X_dual such that

        X_dual[k] = X[k] @ inv(S[k])

    '''
    assert len(X.shape) == 3 and len(S.shape) == 3
    return np.array([np.linalg.solve(Sk.T, Xk.T).T for Xk, Sk in zip(X, S)])


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
    Basis transformation matrix from the truncated spherical Bessel
    function to the pseudo-atomic orbital.

    This function constructs the transformation matrix from some jY
    ([spherical Bessel] x [spherical harmonics]) basis arranged in the
    lexicographic order of (itype, iatom, l, m, q) (q being the index
    for spherical Bessel functions) to some pseudo-atomic orbital basis
    arranged in the lexicographic order of (itype, iatom, l, m, zeta).

    Given "lin2comp" that provides the linear-to-composite index map for
    (itype, iatom, l, m), the entire transformation matrix is a block
    diagonal matrix, with each block corresponding to a specific q->zeta.


    Parameters
    ----------
        coef : nested list
            The coefficients for the orthonormal end-smoothed mixed
            spherical Bessel basis. coef[itype][l][zeta] gives a list of
            coefficients that specifies an orbital.
            Note that the length of this coefficient list is allowed to
            be smaller than nbes-1; the list will be padded with zeros
            to make it of length nbes-1.
        lin2comp : dict
            linear-to-composite index map (not including the spherical
            Bessel index q):

                    mu -> (itype, iatom, l, 0, m).

            NOTE: zeta is supposed to be always 0 in this function.
        nbes : int
            Number of truncated spherical Bessel functions.
        rcut : float
            Cutoff radius.

    Notes
    -----
    This function makes use of JL_REDUCE[rcut][l] in radial.py. One should
    make sure JL_REDUCE[rcut][l] is properly tabulated before calling
    this function.

    '''
    from scipy.linalg import block_diag

    def _gen_q2zeta(coef, lin2comp, nbes, rcut):
        if coef is None:
            return

        for mu in lin2comp:
            itype, _, l, _, _ = lin2comp[mu]
            if l >= len(coef[itype]) or len(coef[itype][l]) == 0:
                # The yielded matrices will be diagonally concatenated
                # by scipy.linalg.block_diag. Therefore, even if the
                # coefficient is not provided, the generator should
                # yield the zero matrix with the appropriate size
                yield np.zeros((nbes, 0))
            else:
                C = np.zeros((nbes-1, len(coef[itype][l])))
                C[:len(coef[itype][l][0])] = np.array(coef[itype][l]).T
                yield JL_REDUCE[rcut][l][:nbes,:nbes-1] @ C

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


class Spillage:
    '''
    Generalized spillage function and its optimization.

    Attributes
    ----------
        config : list
            A list of 2-tuples like (ovlp_dat, op_dat). Each pair corresponds
            to a geometric configuration, where ovlp_dat and op_dat are data
            read from orb_matrix_rcutXderiv0.dat and orb_matrix_rcutXderiv1.dat
            (subject to minor changes, e.g., permutation of ndarrays).
            NOTE: data files are subject to change in the future.

    '''

    def __init__(self):
        self.reset()


    def reset(self):
        self.config = []
        self.rcut = None

        self._reset_frozen()
        self._reset_deriv()


    def _reset_frozen(self):
        self.spill_frozen = None
        self.mo_Pfrozen_jy = None


    def _reset_deriv(self):
        self.dV = []
        self.dao_jy = []


    def add_config(self, ovlp_dat, op_dat):
        '''
        '''
        _assert_consistency(ovlp_dat, op_dat)

        dat = deepcopy(ovlp_dat)
        dat['mo_mo'] = np.array([ovlp_dat['mo_mo'], op_dat['mo_mo']])
        dat['mo_jy'] = np.array([ovlp_dat['mo_jy'], op_dat['mo_jy']])
        dat['jy_jy'] = np.array([ovlp_dat['jy_jy'], op_dat['jy_jy']])

        self.config.append(dat)

        # NOTE currently a dataset merely contains one atom type
        # and one rcut. This may change in the future.
        if self.rcut is None:
            self.rcut = ovlp_dat['rcut']
        else:
            assert self.rcut == ovlp_dat['rcut']

        # transformation matrices from the truncated spherical Bessel functions
        # to the orthonormal end-smoothed mixed spherical Bessel basis
        if self.rcut not in JL_REDUCE:
            JL_REDUCE[self.rcut] = [jl_reduce(l, 100, self.rcut) for l in range(8)]


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
        This function tabulates the derivatives of

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
        The only useful information of coef is its nesting pattern.

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
        # at this stage, the index for each config follows [deriv][0/1][k][mo][ao]
        self.dV = [np.array([dat['mo_jy'] @ jy2dao_i for jy2dao_i in jy2dao])
                   for dat, jy2dao in zip(self.config, jy2dao_all)]

        if self.spill_frozen is not None:
            # if frozen orbitals are present, subtract from the previous results
            # <mo|frozen_dual><frozen|ao> and <mo|frozen_dual><frozen|op|ao>
            self.dV = [dV - np.array([M @ jy2dao_i for jy2dao_i in jy2dao])
                       for dV, M, jy2dao in zip(self.dV, self.mo_Pfrozen_jy, jy2dao_all)]

        # transpose to [0/1][deriv][k][mo][ao]
        self.dV = [dV.transpose(1,0,2,3,4) for dV in self.dV]


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
            dV = self.dV[iconf][:,:,:,ibands,:]

            spill_grad = (_rfrob(dW[1], VdaggerV)
                          - 2.0 * _rfrob(V_dual, dV[1])
                          + 2.0 * _rfrob(dV[0] - V_dual @ dW[0],
                                        _mrdiv(V_dual @ W[1] - V[1], W[0]))
                          ) @ dat['wk']

            spill_grad /= len(ibands)
            spill_grad = nest(spill_grad.tolist(), nestpat(coef))

        return (spill, spill_grad) if with_grad else spill



    def opt(self, coef_init, coef_frozen, ibands):
        '''
        '''
        self._tab_frozen(coef_frozen)
        self._tab_deriv(coef_init)

        pat = nestpat(coef_init)
        nconf = len(self.config)

        def f(c):
            spill, spill_grad = zip(*[
                self._generalize_spillage(iconf, nest(c.tolist(), pat), ibands, True)
                for iconf in range(nconf)])
            return (sum(spill) / nconf,
                    sum(np.array(flatten(g)) for g in spill_grad) / nconf)

        options = {'maxiter': 1000, 'disp': True}

        c0 = flatten(coef_init)
        bounds = [(-1.0, 1.0) for _ in c0]

        res = minimize(f, c0, jac=True, method='L-BFGS-B', bounds=bounds, options=options)
        coef_opt = nest(res.x.tolist(), pat)
        print(coef_opt)

        coef_opt = [[[np.array(coef_tlz) / np.linalg.norm(np.array(coef_tlz))
                      for coef_tlz in coef_tl] for coef_tl in coef_t] for coef_t in coef_opt]
        print(coef_opt)


############################################################
#                           Test
############################################################
import unittest

from indexmap import _index_map

import time
from copy import deepcopy
import matplotlib.pyplot as plt

class _TestSpillage(unittest.TestCase):

    def test_mrdiv(self):
        nk = 3
        nbands = 5
        nao = 6

        # make each slice of S orthogonal to make it easier to verify
        S = np.array([np.linalg.qr(np.random.randn(nao, nao))[0] for _ in range(nk)])

        X = np.random.randn(nk, nbands, nao)
        X_dual = _mrdiv(X, S)

        self.assertEqual(X_dual.shape, X.shape)
        for i in range(nk):
            self.assertTrue( np.allclose(X_dual[i], X[i] @ S[i].T) )


    def test_rfrob(self):
        nk = 5
        nrow = 3
        ncol = 4
        w = np.random.rand(nk)
        X = np.random.randn(nk, nrow, ncol) + 1j * np.random.randn(nk, nrow, ncol)
        Y = np.random.randn(nk, nrow, ncol) + 1j * np.random.randn(nk, nrow, ncol)

        wsum = 0.0
        for wk, Xk, Yk in zip(w, X, Y):
            wsum += wk * np.trace(Xk @ Yk.T.conj()).sum()

        self.assertAlmostEqual(w @ _rfrob(X, Y), wsum.real)

        wsum = np.zeros(nrow, dtype=complex)
        for i in range(nrow):
            for k in range(nk):
                wsum[i] += w[k] * (X[k,i] @ Y[k,i].T.conj())

        self.assertTrue( np.allclose(w @ _rfrob(X, Y, rowwise=True), wsum.real) )


    def test_jy2ao(self):
        ntype = 3
        natom = [1, 2, 3]
        lmax = [2, 1, 0]
        nzeta = [[1, 1, 1], [2, 2], [3]]
        _, lin2comp = _index_map(ntype, natom, lmax, nzeta)

        nbes = 5
        rcut = 6.0

        # NOTE the list of coefficients as given by coef[itype][l][zeta] is w.r.t
        # the end-smoothed mixed spherical Bessel basis, rather than the truncated
        # spherical Bessel functions, which differs by a transformation matrix
        # as given by jl_reduce
        coef = [ [np.random.randn(nzeta[itype][l], nbes-1).tolist()
                  for l in range(lmax[itype]+1)]
                for itype in range(ntype) ]

        M = _jy2ao(coef, lin2comp, nbes, rcut)

        icol = 0
        for mu, (itype, iatom, l, _, m) in lin2comp.items():
            nzeta = len(coef[itype][l])
            self.assertTrue(np.allclose(\
                    M[mu*nbes:(mu+1)*nbes, icol:icol+nzeta], \
                    jl_reduce(l, nbes, rcut) @ np.array(coef[itype][l]).T))
            icol += nzeta


    def est_add_config(self):
        orbgen = Spillage()

        folder = '/home/zuxin/abacus-community/abacus_orbital_generation/tmp/Si-dimer-2.0/'

        mat = read_orb_mat(folder + 'orb_matrix.0.dat')
        dmat = read_orb_mat(folder + 'orb_matrix.1.dat')
        rcut = mat['rcut']

        orbgen.add_config(mat, dmat)

        self.assertEqual(len(orbgen.config), 1)
        self.assertDictEqual(orbgen.config[0][0], mat)
        self.assertDictEqual(orbgen.config[0][1], dmat)

        orbgen.reset()

        mat = read_orb_mat(folder + 'orb_matrix.0.dat')
        dmat = read_orb_mat(folder + 'orb_matrix.1.dat')
        rcut = mat['rcut']

        orbgen.add_config(mat, dmat)

        self.assertEqual(len(orbgen.config), 1)
        self.assertDictEqual(orbgen.config[0][0], mat)
        self.assertDictEqual(orbgen.config[0][1], dmat)


    def test_tab_frozen(self):
        orbgen = Spillage()

        folder = '/home/zuxin/abacus-community/abacus_orbital_generation/tmp/Si-dimer-2.0/'

        mat = read_orb_mat(folder + 'orb_matrix.0.dat')
        dmat = read_orb_mat(folder + 'orb_matrix.1.dat')

        orbgen.add_config(mat, dmat)
        orbgen._tab_frozen([[np.eye(2, mat['nbes']-1).tolist() for l in range(3)]])


    def test_spillage(self):
        orbgen = Spillage()

        topdir = '/home/zuxin/abacus-community/abacus_orbital_generation/tmp/'
        confdir = ['Si-dimer-2.0', 'Si-dimer-2.2', 'Si-trimer-2.1', 'Si-trimer-2.3']
        #confdir = ['Si-dimer-2.0', 'Si-dimer-2.2']

        print('read to add config...')

        for conf in confdir:
            folder = topdir + conf + '/'
            mat = read_orb_mat(folder + 'orb_matrix.0.dat')
            dmat = read_orb_mat(folder + 'orb_matrix.1.dat')
            orbgen.add_config(mat, dmat)
            print('config added: ', conf)

        coef = [[np.eye(2, mat['nbes']-1).tolist(), \
                np.eye(2, mat['nbes']-1).tolist(), \
                np.eye(1, mat['nbes']-1).tolist()]]

        ibands = range(5)

        np.random.seed(0)

        #coef_frozen = [[np.eye(2, mat['nbes']-1).tolist(), \
        #        np.eye(2, mat['nbes']-1).tolist(), \
        #        np.eye(1, mat['nbes']-1).tolist()]]

        coef_frozen = [[np.random.randn(2, mat['nbes']-1).tolist(), \
                np.random.randn(2, mat['nbes']-1).tolist(), \
                np.random.randn(1, mat['nbes']-1).tolist()]]

        print('tab frozen...')
        orbgen._tab_frozen(coef_frozen)

        print('tab deriv...')
        orbgen._tab_deriv(coef)

        dc = 1e-6
        coef_p = deepcopy(coef)
        coef_p[0][0][0][3] += dc
        spill_p = orbgen._generalize_spillage(1, coef_p, ibands, False)

        coef_m = deepcopy(coef)
        coef_m[0][0][0][3] -= dc
        spill_m = orbgen._generalize_spillage(1, coef_m, ibands, False)

        dspill_fd = (spill_p - spill_m) / (2 * dc)

        print('')
        start = time.time()
        dspill = orbgen._generalize_spillage(1, coef, ibands, True)[1]
        print('time = ', time.time() - start)

        print('dspill  ( analytic  ) = ', dspill[0][0][0][3])
        print('dspill  (finite diff) = ', dspill_fd)

        #coef0 = [[np.random.rand(2, mat['nbes']-1).tolist(),
        #          np.random.rand(2, mat['nbes']-1).tolist(),
        #          np.random.rand(1, mat['nbes']-1).tolist()]]
        coef0 = [[np.eye(2, mat['nbes']-1).tolist(),
                  np.eye(2, mat['nbes']-1).tolist(),
                  np.eye(1, mat['nbes']-1).tolist()]]

        orbgen.opt(coef0, coef_frozen, range(4))


        #orbgen.reset()
        #orbgen.add_config(mat, mat)
        #orbgen._tab_frozen(coef_frozen)
        #spill_ref = _overlap_spillage(mat, coef, ibands, coef_frozen)
        #spill = orbgen._generalize_spillage(0, coef, ibands, False)

        #print('spill ref = ', spill_ref)
        #print('spill     = ', spill)



if __name__ == '__main__':
    unittest.main()


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


def _sum_fro(w, A, B, return_real=True, rowwise=False):
    '''
    Weighted sum of Frobenius inner products.

    The Frobenius inner product between two matrices or vectors is defined as

        <X, Y> \equiv Tr(X @ Y.T.conj()) = (X * Y.conj()).sum()

    NOTE: the inner product is assumed to have the Hermitian conjugate on the
    second argument, not the first.

    Given a 1-d array w and two 3-d arrays A and B, if `rowwise` is False,
    this function computes the weighted sum of the slice-wise Frobenius
    inner product:

        res = \sum_k w[k] * <A[k], B[k]>

    If `rowwise` is True, the returned value will be a 1-d array
    computed as

        res[i] = \sum_k w[k] * <A[k,i], B[k,i]>

    '''
    tmp = w.reshape(w.size, 1, 1) * A * B.conj()
    tmp = tmp.real if return_real else tmp
    return tmp.sum((-3,-1) if rowwise else (-3,-2,-1))


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

# TODO should get rid of this function
# and use the one above instead
def _obsolete(coef, arr, lin2comp, rcut):
    '''
    Basis transformation from jY to pseudo-atomic orbitals.

    This function transforms a 2 or 3d array with some jY
    ([spherical Bessel] x [spherical harmonics]) basis to the
    corresponding array with pseudo-atomic orbital basis specified
    by the given coefficients.

    The arrangement of jY basis is assumed
    to follow a lexicographic order of (itype, iatom, l, m, q) where
    q is the label for the spherical Bessel functions.

    Parameters
    ----------
        coef : list or tuple
            The coefficients of pseudo-atomic orbitals in the
            orthonormal end-smoothed mixed spherical Bessel basis,
            where coef[itype][l][zeta] is a list of float that
            specifies a pseudo-atomic orbital.
            coef could also be a tuple like (coef_bra, coef_ket),
            in which case the ket and bra will be transformed by
            the respective coefficients. If either is None, the
            corresponding transformation will be skipped.
        arr : np.ndarray
            A 2 or 3d array. The transformation will be applied
            to the last two dimensions. If arr is 3d, the first
            dimension is broadcasted.
        lin2comp : dict
            linear-to-composite index map (not including the spherical
            Bessel index q):

                    mu -> (itype, iatom, l, 0, m).

            NOTE: zeta is supposed to be always 0 in this function.
        rcut : float
            Cutoff radius.

    Notes
    -----
    The raw output of ABACUS corresponds to a 5-d array of shape
    (nk, nao, nao, nbes, nbes). It shall be permuted and reshaped
    before being passed to this function. Currently read_orb_mat
    in datparse.py takes care of these operations.

    '''
    if isinstance(coef, list):
        nbes = arr.shape[-1] // len(lin2comp)
        M = _jy2ao(coef, lin2comp, nbes, rcut)
        arr = M.T @ arr @ M
    else:
        coef_bra, coef_ket = coef

        if coef_bra is not None:
            nbes = arr.shape[-2] // len(lin2comp)
            arr = _jy2ao(coef_bra, lin2comp, nbes, rcut).T @ arr

        if coef_ket is not None:
            nbes = arr.shape[-1] // len(lin2comp)
            arr = arr @ _jy2ao(coef_ket, lin2comp, nbes, rcut)

    return arr


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
        spill -= _sum_fro(ovlp['wk'], X_dual, X)

        V -= X_dual @ jy2frozen.T @ ovlp['jy_jy'] @ jy2ao

    spill -= _sum_fro(ovlp['wk'], _mrdiv(V, W), V)

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
        self.M = None


    def _reset_deriv(self):
        self.dV = []
        self.dJ = []


    def add_config(self, ovlp_dat, op_dat):
        '''
        '''
        _assert_consistency(ovlp_dat, op_dat)

        # NOTE currently a dataset merely contains one atom type
        # and one rcut. This may change in the future.
        if self.rcut is None:
            self.rcut = ovlp_dat['rcut']
        else:
            assert self.rcut == ovlp_dat['rcut']

        self.config.append((ovlp_dat, op_dat))

        # transformation matrices from the truncated spherical Bessel functions
        # to the orthonormal end-smoothed mixed spherical Bessel basis
        if self.rcut not in JL_REDUCE:
            JL_REDUCE[self.rcut] = [jl_reduce(l, 100, self.rcut) for l in range(8)]


    def _tab_frozen(self, coef_frozen):
        '''
        Tabulates the band-wise spillage contribution from frozen orbitals and

                        <mo|frozen_dual><frozen|jy>
                        <mo|frozen_dual><frozen|op|jy>

        for each configuration.

        '''
        jy2frozen_all = [_jy2ao(coef_frozen, ox['lin2comp'], ox['nbes'], ox['rcut'])
                         for ox,_ in self.config]

        frozen_frozen, frozen_op_frozen = zip(*[
            [jy2frozen.T @ ox['jy_jy'] @ jy2frozen for ox in dat]
            for dat, jy2frozen in zip(self.config, jy2frozen_all)])

        mo_frozen, mo_op_frozen = zip(*[
            [ox['mo_jy'] @ jy2frozen for ox in dat]
            for dat, jy2frozen in zip(self.config, jy2frozen_all)])

        mo_frozen_dual = [_mrdiv(X, S) for X, S in zip(mo_frozen, frozen_frozen)]

        self.M = [[X_dual @ jy2frozen.T @ ox['jy_jy'] for ox in dat]
                  for X_dual, dat, jy2frozen in \
                          zip(mo_frozen_dual, self.config, jy2frozen_all)]

        wks = [op['wk'] for _, op in self.config]
        self.spill_frozen = [_sum_fro(wk, X_dual @ S_op, X_dual, rowwise=True)
                             - 2.0 * _sum_fro(wk, X_dual, X_op, rowwise=True)
                             for wk, X_dual, X_op, S_op in
                             zip(wks, mo_frozen_dual, mo_op_frozen, frozen_op_frozen)]


    def _tab_deriv(self, coef):
        '''
        This function includes two parts:

        1. Tabulates the derivatives of

                        <mo|ao>
                        <mo|op|ao>

        or, if frozen orbitals are present, the derivatives of

                <mo|( 1 - |frozen_dual><frozen| )|ao>
                <mo|( 1 - |frozen_dual><frozen| )|op|ao>

        with respect to the coefficients that specifies |ao>.

        2. Tabulates the derivatives of <ao|jy> with respect to
        the coefficients.


        Note
        ----
        The only useful information of coef is its nesting pattern.

        '''
        sz = len(flatten(coef))
        _c = [nest(ci.tolist(), nestpat(coef)) for ci in np.eye(sz)]

        # Part-1: derivatives of <mo|ao> and <mo|op|ao>
        self.dV = [[np.array([_obsolete((None, ci), ox['mo_jy'], ox['lin2comp'], ox['rcut'])
                              for ci in _c]) # loop coef
                    for ox in dat] # loop ov & op
                   for dat in self.config] # loop config

        # if frozen orbitals are present, subtract from the previous results
        # <mo|frozen_dual><frozen|ao> and <mo|frozen_dual><frozen|op|ao>
        if self.spill_frozen is not None:
            self.dV = [[dV_ox - np.array([_obsolete((None, ci), M_ox, ox['lin2comp'], ox['rcut'])
                                          for ci in _c]) # loop coef
                        for dV_ox, ox, M_ox in zip(dV, dat, M)] # loop ov & op
                       for dV, dat, M in zip(self.dV, self.config, self.M)] # loop config

        # Part-2: derivatives of <ao|jy>
        self.dJ = [[np.array([_obsolete((ci, None), ox['jy_jy'], ox['lin2comp'], ox['rcut'])
                              for ci in _c]) for ox in dat] for dat in self.config]


    def _generalize_spillage(self, iconf, coef, ibands, with_grad=False):
        '''
        Generalized spillage function and its gradient.

        '''
        ov, op = self.config[iconf]

        spill = (op['wk'] @ op['mo_mo'][:,ibands]).real.sum()

        # <mo|ao> and <mo|op|ao>
        V_ov = _obsolete((None, coef), ov['mo_jy'][:,ibands,:], ov['lin2comp'], ov['rcut'])
        V_op = _obsolete((None, coef), op['mo_jy'][:,ibands,:], op['lin2comp'], op['rcut'])

        if self.spill_frozen is not None:
            spill += self.spill_frozen[iconf][ibands].sum()

            # <mo|frozen_dual><frozen|jy> and <mo|frozen_dual><frozen|op|jy>
            M_ov, M_op = self.M[iconf]

            # if frozen orbitals are present, subtract from the previous V
            # <mo|frozen_dual><frozen|ao> and <mo|frozen_dual><frozen|op|ao> respectively
            V_ov -= _obsolete((None, coef), M_ov[:,ibands,:], ov['lin2comp'], ov['rcut'])
            V_op -= _obsolete((None, coef), M_op[:,ibands,:], op['lin2comp'], op['rcut'])

        # <ao|ao> and <ao|op|ao>
        W_ov = _obsolete(coef, ov['jy_jy'], ov['lin2comp'], ov['rcut'])
        W_op = _obsolete(coef, op['jy_jy'], op['lin2comp'], op['rcut'])

        V_ov_dual = _mrdiv(V_ov, W_ov)
        VV = V_ov_dual.transpose((0,2,1)).conj() @ V_ov_dual

        spill += _sum_fro(op['wk'], V_ov_dual @ W_op, V_ov_dual) \
                - 2.0 * _sum_fro(op['wk'], V_ov_dual, V_op)
        spill /= len(ibands)

        if with_grad:
            dJ_ov, dJ_op = self.dJ[iconf]

            j2zeta = _jy2ao(coef, ov['lin2comp'], ov['nbes'], ov['rcut'])
            dW_ov = dJ_ov @ j2zeta
            dW_op = dJ_op @ j2zeta
            dW_ov += dW_ov.transpose((0,1,3,2)).conj()
            dW_op += dW_op.transpose((0,1,3,2)).conj()

            dV_ov = self.dV[iconf][0][:,:,ibands,:]
            dV_op = self.dV[iconf][1][:,:,ibands,:]

            spill_grad = _sum_fro(op['wk'], dW_op, VV) \
                 - 2.0 * _sum_fro(op['wk'], V_ov_dual, dV_op) \
                 + 2.0 * _sum_fro(op['wk'], dV_ov - V_ov_dual @ dW_ov,
                                  _mrdiv(V_ov_dual @ W_op - V_op, W_ov))

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


    def test_sum_fro(self):
        nk = 5
        nrow = 3
        ncol = 4
        w = np.random.rand(nk)
        X = np.random.randn(nk, nrow, ncol) + 1j * np.random.randn(nk, nrow, ncol)
        Y = np.random.randn(nk, nrow, ncol) + 1j * np.random.randn(nk, nrow, ncol)

        wsum = 0.0
        for wk, Xk, Yk in zip(w, X, Y):
            wsum += wk * np.trace(Xk @ Yk.T.conj()).sum()

        self.assertAlmostEqual(_sum_fro(w, X, Y, False), wsum)
        self.assertAlmostEqual(_sum_fro(w, X, Y, True), wsum.real)

        wsum = np.zeros(nrow, dtype=complex)
        for i in range(nrow):
            for k in range(nk):
                wsum[i] += w[k] * (X[k,i] @ Y[k,i].T.conj())

        self.assertTrue( np.allclose(_sum_fro(w, X, Y, False, True), wsum) )
        self.assertTrue( np.allclose(_sum_fro(w, X, Y, True, True), wsum.real) )


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


    def test_obsolete(self):
        ovlp = read_orb_mat('./testfiles/orb_matrix_rcut6deriv0.dat')

        # 2s2p1d
        coef = [[np.eye(2, ovlp['nbes']-1).tolist(),
                 np.eye(2, ovlp['nbes']-1).tolist(),
                 np.eye(1, ovlp['nbes']-1).tolist()]]

        nao = sum(len(coef_tl) * (2*l+1) * ovlp['natom'][it]
                  for it, coef_t in enumerate(coef) for l, coef_tl in enumerate(coef_t))

        S = _obsolete(coef, ovlp['jy_jy'], ovlp['lin2comp'], ovlp['rcut'])
        self.assertEqual(S.shape, (ovlp['nk'], nao, nao))

        # overlap matrix should be hermitian
        for Sk in S:
            self.assertLess(np.linalg.norm(Sk-Sk.T.conj(), np.inf), 1e-12)

        # ensure one-sided transformations do not alter the shape of the other side
        tmp = _obsolete((coef, None), ovlp['jy_jy'], ovlp['lin2comp'], ovlp['rcut'])
        self.assertEqual((ovlp['nk'], nao, ovlp['jy_jy'].shape[2]), tmp.shape)

        tmp = _obsolete((None, coef), ovlp['jy_jy'], ovlp['lin2comp'], ovlp['rcut'])
        self.assertEqual((ovlp['nk'], ovlp['jy_jy'].shape[1], nao), tmp.shape)


    def test_add_config(self):
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
        #confdir = ['Si-dimer-2.0', 'Si-dimer-2.2', 'Si-trimer-2.1', 'Si-trimer-2.3']
        confdir = ['Si-dimer-2.0', 'Si-dimer-2.2']

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


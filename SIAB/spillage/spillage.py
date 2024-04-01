from datparse import read_orb_mat, _assert_consistency
from indexmap import _index_map
from radial import jl_reduce, JL_REDUCE
from listmanip import flatten, nest, nestpat

import numpy as np
from scipy.linalg import block_diag
from copy import deepcopy


def _mrdivide(X, S):
    '''
    Right matrix division.

    Given two 3-d arrays X and S, returns a 3-d array X_dual such that

        X_dual[k] = X[k] @ inv(S[k])

    '''
    assert len(X.shape) == 3 and len(S.shape) == 3
    return np.array([np.linalg.solve(Sk.T, Xk.T).T for Xk, Sk in zip(X, S)])


def _wsum_fro(w, A, B, return_real=True, rowwise=False):
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
    return tmp.sum((0,2) if rowwise else None)


def _gen_q2zeta(coef, mu2comp, nbes, rcut):
    '''
    Basis transformation matrix from the truncated spherical Bessel
    function to the pseudo-atomic orbital.

    Given an index map "mu2comp" (see indexmap.py), the number of truncated
    spherical Bessel functions "nbes" and cutoff radius "rcut", this generator
    generates for each mu the transformation matrix from the truncated spherical
    Bessel function to the pseudo-atomic orbital, which is a linear combination of
    orthonormal end-smoothed mixed spherical Bessel basis specificied by coef.

    Parameters
    ----------
        coef : nested list
            The coefficients for the orthonormal end-smoothed mixed
            spherical Bessel basis. coef[itype][l][zeta] gives a list of
            coefficients that specifies an orbital.
            Note that the length of this coefficient list is allowed to
            be smaller than nbes-1; the list will be padded with zeros
            to make it of length nbes-1.
        mu2comp : dict
            Index map mu -> (itype, iatom, l, zeta, m).
            NOTE: zeta is supposed to be 0 for all mu.
        nbes : int
            Number of truncated spherical Bessel functions.
        rcut : float
            Cutoff radius.

    Notes
    -----
    This generator makes use of JL_REDUCE[rcut][l] in radial.py. One should
    make sure JL_REDUCE[rcut][l] is properly tabulated before calling
    this function.

    '''
    if coef is None:
        return

    for mu in mu2comp:
        itype, _, l, _, _ = mu2comp[mu]
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


def _jy2ao(coef, array, lin2comp, rcut):
    '''
    Basis transformation from jY to pseudo-atomic orbitals.

    This function transforms matrix elements in some jY basis to the
    matrix elements between pseudo-atomic orbitals specified by the
    given coefficients in the orthonormal end-smoothed mixed spherical
    Bessel basis.

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
        jy_jy : np.ndarray, shape (nk, nao*nbes, nao*nbes)
            The original matrix in jY basis as read from an
            orb_matrix_rcutXderivY.dat file. See also Notes.
        jy_mu2comp : dict
            Index map mu -> (itype, iatom, l, zeta, m).
            NOTE: zeta is supposed to be 0 for all mu.
        rcut : float
            Cutoff radius.

    Notes
    -----
    The raw output of ABACUS corresponds to a 5-d array of shape
    (nk, nao, nao, nbes, nbes). It shall be permuted before being
    passed to this function. Currently read_orb_mat in datparse.py
    takes care of this permutation.

    '''
    if isinstance(coef, list):
        nbes = array.shape[1] // len(lin2comp)
        M = block_diag(*_gen_q2zeta(coef, lin2comp, nbes, rcut))
        array = M.T @ array @ M
    else:
        coef_bra, coef_ket = coef

        if coef_bra is not None:
            nbes = array.shape[1] // len(lin2comp)
            array = block_diag(*_gen_q2zeta(coef_bra, lin2comp, nbes, rcut)).T @ array

        if coef_ket is not None:
            nbes = array.shape[2] // len(lin2comp)
            array = array @ block_diag(*_gen_q2zeta(coef_ket, lin2comp, nbes, rcut))

    return array


def _overlap_spillage(ovlp, coef, ibands, coef_frozen=None):
    '''
    Standard spillage function (overlap spillage).

    Note
    ----
    This function is not supposed to be used in the optimization.
    As a special case of the generalized spillage (op = I), it serves
    as a cross-check for the implementation of the generalized spillage.

    '''
    spill = (ovlp['wk'] @ ovlp['mo_mo'][:, ibands]).real.sum()

    mo_jy = ovlp['mo_jy'][:, ibands, :]

    V = _jy2ao((None, coef), mo_jy, ovlp['mu2comp'], ovlp['rcut'])
    W = _jy2ao(coef, ovlp['jy_jy'], ovlp['mu2comp'], ovlp['rcut'])

    if coef_frozen is not None:
        X = _jy2ao((None, coef_frozen), mo_jy, ovlp['mu2comp'], ovlp['rcut'])
        S = _jy2ao(coef_frozen, ovlp['jy_jy'], ovlp['mu2comp'], ovlp['rcut'])

        X_dual = _mrdivide(X, S)
        spill -= _wsum_fro(ovlp['wk'], X_dual, X)

        V -= X_dual @ _jy2ao((coef_frozen, coef), ovlp['jy_jy'],
                             ovlp['mu2comp'], ovlp['rcut'])

    spill -= _wsum_fro(ovlp['wk'], _mrdivide(V, W), V)

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

        self.mo_projfrozen_jy = None
        self.mo_projfrozen_op_jy = None

    
    def _reset_deriv(self):
        self.dV = ()
        self.dV_op = ()

        self.dao_jy = ()
        self.dao_op_jy = ()


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
        Tabulates quantities related to the frozen-orbitals, including

        <mo|frozen's dual>      

        '''
        frozen_frozen, frozen_op_frozen = zip(*[(
            _jy2ao(coef_frozen, ovlp['jy_jy'], ovlp['mu2comp'], ovlp['rcut']),
            _jy2ao(coef_frozen, op['jy_jy'], op['mu2comp'], op['rcut'])
            ) for ovlp, op in self.config])

        mo_frozen, mo_op_frozen = zip(*[(
            _jy2ao((None, coef_frozen), ovlp['mo_jy'], ovlp['mu2comp'], ovlp['rcut']),
            _jy2ao((None, coef_frozen), op['mo_jy'], op['mu2comp'], op['rcut'])
            ) for ovlp, op in self.config])

        mo_frozen_dual = [_mrdivide(X, S) for X, S in zip(mo_frozen, frozen_frozen)]

        self.mo_projfrozen_jy, self.mo_projfrozen_op_jy = zip(*[(
            X_dual @ _jy2ao((coef_frozen, None), ovlp['jy_jy'], ovlp['mu2comp'], ovlp['rcut']),
            X_dual @ _jy2ao((coef_frozen, None), op['jy_jy'], op['mu2comp'], op['rcut'])
            ) for X_dual, (ovlp, op) in zip(mo_frozen_dual, self.config)])

        wks = [op['wk'] for _, op in self.config]
        self.spill_frozen = [_wsum_fro(wk, X_dual @ S_op, X_dual, rowwise=True)
                             - 2.0 * _wsum_fro(wk, X_dual, X_op, rowwise=True)
                             for wk, X_dual, X_op, S_op in
                             zip(wks, mo_frozen_dual, mo_op_frozen, frozen_op_frozen)]


    def _tab_deriv(self, coef):
        '''
        Tabulates the derivatives of

                        <mo|ao>
                        <mo|op|ao>

        or, if frozen orbitals are present, the derivatives of

                <mo|( 1 - |frozen_dual><frozen| )|ao>
                <mo|( 1 - |frozen_dual><frozen| )|op|ao>

        with respect to the coefficients.


        Note
        ----
        The only useful information of coef is its nesting pattern.
        
        '''
        sz = len(flatten(coef))
        _c = [nest(ci.tolist(), nestpat(coef)) for ci in np.eye(sz)]

        _dY = lambda op: \
                np.array([_jy2ao((None, ci), op['mo_jy'], op['mu2comp'], op['rcut'])
                          for ci in _c])

        _XdZ = lambda op, mo_projfrozen_op_jy: \
                np.array([_jy2ao((None, ci), mo_projfrozen_op_jy,
                                 op['mu2comp'], op['rcut']) for ci in _c])

        dV = [(_dY(ovlp), _dY(op)) for ovlp, op in self.config]

        if self.spill_frozen is not None:
            dV = [(dY - _XdZ(ovlp, mo_projfrozen_jy), dY_op - _XdZ(op, mo_projfrozen_op_jy))
                  for (dY, dY_op), (ovlp, op), mo_projfrozen_jy, mo_projfrozen_op_jy
                  in zip(dV, self.config, self.mo_projfrozen_jy, self.mo_projfrozen_op_jy)]

        self.dV, self.dV_op = zip(*dV)


        _dao_jy = lambda op: \
                np.array([_jy2ao((ci, None), op['jy_jy'], op['mu2comp'], op['rcut'])
                          for ci in _c])

        self.dao_jy, self.dao_op_jy = zip(*[(_dao_jy(ovlp), _dao_jy(op))
                                            for (ovlp, op) in self.config])


    def _generalize_spillage(self, iconf, coef, ibands, with_grad=False):
        '''
        Generalized spillage function and its gradient.

        '''
        ovlp, op = self.config[iconf]

        spill = (op['wk'] @ op['mo_mo'][:,ibands]).real.sum()

        V = _jy2ao((None, coef), ovlp['mo_jy'][:,ibands,:], ovlp['mu2comp'], ovlp['rcut'])
        V_op = _jy2ao((None, coef), op['mo_jy'][:,ibands,:], op['mu2comp'], op['rcut'])

        if self.spill_frozen is not None:
            spill += self.spill_frozen[iconf][ibands].sum()

            V -= _jy2ao((None, coef), self.mo_projfrozen_jy[iconf][:,ibands,:],
                        ovlp['mu2comp'], ovlp['rcut'])
            V_op -= _jy2ao((None, coef), self.mo_projfrozen_op_jy[iconf][:,ibands,:],
                           op['mu2comp'], op['rcut'])

        W = _jy2ao(coef, ovlp['jy_jy'], ovlp['mu2comp'], ovlp['rcut'])
        W_op = _jy2ao(coef, op['jy_jy'], op['mu2comp'], op['rcut'])

        V_dual = _mrdivide(V, W)
        spill += _wsum_fro(op['wk'], V_dual @ W_op, V_dual) \
                - 2.0 * _wsum_fro(op['wk'], V_dual, V_op)

        spill /= len(ibands)

        if with_grad:
            sz = len(flatten(coef))
            spill_grad = np.zeros(sz)

            for i in range(sz):
                # (d/dcoef[i])<ao|ao>
                dW = _jy2ao((None, coef), self.dao_jy[iconf][i],
                            ovlp['mu2comp'], ovlp['rcut'])
                dW += dW.transpose((0,2,1)).conj()

                # (d/dcoef[i])<ao|op|ao>
                dW_op = _jy2ao((None, coef), self.dao_op_jy[iconf][i],
                               ovlp['mu2comp'], ovlp['rcut'])
                dW_op += dW_op.transpose((0,2,1)).conj()

                spill_grad[i] = _wsum_fro(op['wk'], V_dual @ dW_op, V_dual) \
                        - 2.0 * _wsum_fro(op['wk'], V_dual, self.dV_op[iconf][i][:,ibands,:]) \
                        + 2.0 * _wsum_fro(op['wk'], _mrdivide(self.dV[iconf][i][:,ibands,:]
                                                              - V_dual @ dW, W),
                                          V_dual @ W_op - V_op)

            spill_grad /= len(ibands)
            spill_grad = nest(spill_grad.tolist(), nestpat(coef))

        return (spill, spill_grad) if with_grad else spill


    def opt(self, coef_init):
        '''
        '''
        pass


############################################################
#                           Test
############################################################
import unittest

import matplotlib.pyplot as plt
import time

class _TestSpillage(unittest.TestCase):

    def test_mrdivide(self):
        nk = 3
        nbands = 5
        nao = 6

        # make each slice of S orthogonal to make it easier to verify
        S = np.array([np.linalg.qr(np.random.randn(nao, nao))[0] for _ in range(nk)])

        X = np.random.randn(nk, nbands, nao)
        X_dual = _mrdivide(X, S)

        self.assertEqual(X_dual.shape, X.shape)
        for i in range(nk):
            self.assertTrue( np.allclose(X_dual[i], X[i] @ S[i].T) )


    def test_wsum_fro(self):
        nk = 5
        nrow = 3
        ncol = 4
        w = np.random.rand(nk)
        X = np.random.randn(nk, nrow, ncol)
        Y = np.random.randn(nk, nrow, ncol)

        wsum = 0.0
        for wk, Xk, Yk in zip(w, X, Y):
            wsum += wk * np.trace(Xk @ Yk.T.conj()).sum()

        self.assertAlmostEqual(_wsum_fro(w, X, Y, False), wsum)
        self.assertAlmostEqual(_wsum_fro(w, X, Y, True), wsum.real)

        wsum = np.zeros(nrow)
        for i in range(nrow):
            for k in range(nk):
                wsum[i] += w[k] * (X[k,i] @ Y[k,i].T.conj())

        self.assertTrue( np.allclose(_wsum_fro(w, X, Y, False, True), wsum) )
        self.assertTrue( np.allclose(_wsum_fro(w, X, Y, True, True), wsum.real) )


    def test_gen_q2zeta(self):
        ntype = 3
        natom = [1, 2, 3]
        lmax = [2, 1, 0]
        nzeta = [[1, 1, 1], [2, 2], [3]]
        _, mu2comp = _index_map(ntype, natom, lmax, nzeta)

        nbes = 5
        rcut = 6.0

        # NOTE the list of coefficients as given by coef[itype][l][zeta] is w.r.t
        # the end-smoothed mixed spherical Bessel basis, rather than the truncated
        # spherical Bessel functions, which differs by a transformation matrix
        # as given by jl_reduce
        coef = [ [np.random.randn(nzeta[itype][l], nbes-1).tolist()
                  for l in range(lmax[itype]+1)]
                for itype in range(ntype) ]

        for mu, q2zeta in enumerate(_gen_q2zeta(coef, mu2comp, nbes, rcut)):
            itype, iatom, l, zeta, m = mu2comp[mu]
            self.assertEqual(q2zeta.shape, (nbes, nzeta[itype][l]))
            self.assertTrue( np.allclose(q2zeta, \
                    jl_reduce(l, nbes, rcut) @ np.array(coef[itype][l]).T))


    def test_jy2ao(self):
        ovlp = read_orb_mat('./testfiles/orb_matrix_rcut6deriv0.dat')

        # 2s2p1d
        coef = [[np.eye(2, ovlp['nbes']-1).tolist(),
                 np.eye(2, ovlp['nbes']-1).tolist(),
                 np.eye(1, ovlp['nbes']-1).tolist()]]

        nao = sum(len(coef_tl) * (2*l+1) * ovlp['natom'][it]
                  for it, coef_t in enumerate(coef) for l, coef_tl in enumerate(coef_t))

        S = _jy2ao(coef, ovlp['jy_jy'], ovlp['mu2comp'], ovlp['rcut'])
        self.assertEqual(S.shape, (ovlp['nk'], nao, nao))

        # overlap matrix should be hermitian
        for Sk in S:
            self.assertLess(np.linalg.norm(Sk-Sk.T.conj(), np.inf), 1e-12)

        # ensure one-sided transformations do not alter the shape of the other side
        tmp = _jy2ao((coef, None), ovlp['jy_jy'], ovlp['mu2comp'], ovlp['rcut'])
        self.assertEqual((ovlp['nk'], nao, ovlp['jy_jy'].shape[2]), tmp.shape)

        tmp = _jy2ao((None, coef), ovlp['jy_jy'], ovlp['mu2comp'], ovlp['rcut'])
        self.assertEqual((ovlp['nk'], ovlp['jy_jy'].shape[1], nao), tmp.shape)


    def test_add_config(self):
        orbgen = Spillage()

        folder = '/home/zuxin/abacus-community/abacus_orbital_generation/tmp/Si-dimer-2.0/'

        mat = read_orb_mat(folder + 'orb_matrix_rcut6deriv0.dat')
        dmat = read_orb_mat(folder + 'orb_matrix_rcut6deriv1.dat')
        rcut = mat['rcut']

        orbgen.add_config(mat, dmat)

        self.assertEqual(len(orbgen.config), 1)
        self.assertDictEqual(orbgen.config[0][0], mat)
        self.assertDictEqual(orbgen.config[0][1], dmat)

        orbgen.reset()

        mat = read_orb_mat(folder + 'orb_matrix_rcut7deriv0.dat')
        dmat = read_orb_mat(folder + 'orb_matrix_rcut7deriv1.dat')
        rcut = mat['rcut']

        orbgen.add_config(mat, dmat)

        self.assertEqual(len(orbgen.config), 1)
        self.assertDictEqual(orbgen.config[0][0], mat)
        self.assertDictEqual(orbgen.config[0][1], dmat)


    def test_tab_frozen(self):
        orbgen = Spillage()

        folder = '/home/zuxin/abacus-community/abacus_orbital_generation/tmp/Si-dimer-2.0/'

        mat = read_orb_mat(folder + 'orb_matrix_rcut6deriv0.dat')
        dmat = read_orb_mat(folder + 'orb_matrix_rcut6deriv1.dat')

        orbgen.add_config(mat, dmat)
        orbgen._tab_frozen([[np.eye(2, mat['nbes']-1).tolist() for l in range(3)]])


    def test_spillage(self):
        orbgen = Spillage()

        folder = '/home/zuxin/abacus-community/abacus_orbital_generation/tmp/Si-dimer-2.0/'

        mat = read_orb_mat(folder + 'orb_matrix_rcut6deriv0.dat')
        dmat = read_orb_mat(folder + 'orb_matrix_rcut6deriv1.dat')

        orbgen.add_config(mat, dmat)

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

        orbgen._tab_frozen(coef_frozen)
        orbgen._tab_deriv(coef)

        dc = 1e-6
        coef_p = deepcopy(coef)
        coef_p[0][0][0][3] += dc
        spill_p = orbgen._generalize_spillage(0, coef_p, ibands, False)

        coef_m = deepcopy(coef)
        coef_m[0][0][0][3] -= dc
        spill_m = orbgen._generalize_spillage(0, coef_m, ibands, False)

        dspill_fd = (spill_p - spill_m) / (2 * dc)

        print('')
        start = time.time()        
        dspill = orbgen._generalize_spillage(0, coef, ibands, True)[1]
        print('time = ', time.time() - start)

        print('dspill  ( analytic  ) = ', dspill[0][0][0][3])
        print('dspill  (finite diff) = ', dspill_fd)


        orbgen.reset()
        orbgen.add_config(mat, mat)
        orbgen._tab_frozen(coef_frozen)
        spill_ref = _overlap_spillage(mat, coef, ibands, coef_frozen)
        spill = orbgen._generalize_spillage(0, coef, ibands, False)

        print('spill ref = ', spill_ref)
        print('spill     = ', spill)







if __name__ == '__main__':
    unittest.main()


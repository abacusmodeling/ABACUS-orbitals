from datparse import read_orb_mat, _assert_consistency
from indexmap import _index_map
from radial import jl_reduce, JL_REDUCE
from listmanip import flatten, nest, nestpat

import numpy as np
from scipy.linalg import block_diag
from copy import deepcopy


def _mrdivide(X, S):
    '''
    Given two 3-d arrays X and S, returns a 3-d array X_dual such that

        X_dual[k] = X[k] @ inv(S[k])

    '''
    assert len(X.shape) == 3 and len(S.shape) == 3
    return np.array([np.linalg.solve(Sk.T, Xk.T).T for Xk, Sk in zip(X, S)])


def _wsum_fro(w, A, B, return_real=True, rowwise=False):
    '''
    Weighted sum of Frobenius inner products.

    The Frobenius inner product can be defined as

        <X, Y> \equiv Tr(X @ Y.T.conj()) = (X * Y.conj()).sum()

    Given a 1-d array w and two 3-d arrays A and B, if `rowwise` is False,
    this function computes the weighted sum of the slice-wise Frobenius
    inner product:

        res = \sum_k w[k] * <A[k], B[k]>

    If `rowwise` is True, the returned value will be a 1-d array
    computed as

        res[i] = \sum_k w[k] * <A[k,i], B[k,i]>

    in which case the returned value will be a 1-d array.

    Note
    ----
    The inner product is assumed to have the Hermitian conjugate
    on B, rather than on A.

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


def _ao_ao(coef, jy_jy, jy_mu2comp, rcut):
    '''
    Matrix elements between pseudo-atomic orbitals.

    This function transforms matrix elements in the jY basis to the
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
        jy_jy : np.ndarray, shape (nk, nao, nbes, nao, nbes)
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
    nk, nao, nbes = jy_jy.shape[0], jy_jy.shape[1], jy_jy.shape[-1]
    tmp = jy_jy.reshape(nk, nao*nbes, nao*nbes)

    if isinstance(coef, list):
        M = block_diag(*_gen_q2zeta(coef, jy_mu2comp, nbes, rcut))
        tmp = M.T @ tmp @ M
    else:
        coef_bra, coef_ket = coef

        if coef_bra is not None:
            tmp = block_diag(*_gen_q2zeta(coef_bra, jy_mu2comp, nbes, rcut)).T @ tmp

        if coef_ket is not None:
            tmp = tmp @ block_diag(*_gen_q2zeta(coef_ket, jy_mu2comp, nbes, rcut))

    return tmp


def _mo_ao(coef, mo_jy, jy_mu2comp, rcut, ibands=None):
    '''
    Matrix elements between MO and pseudo-atomic orbitals.

    This function transforms matrix elements between MO and jY to the
    matrix elements between MO and pseudo-atomic orbitals specified by
    the given coefficients in the orthonormal end-smoothed mixed spherical
    Bessel basis.

    Parameters
    ----------
        coef : nested list
            The coefficients of pseudo-atomic orbitals in the
            orthonormal end-smoothed mixed spherical Bessel basis,
            where coef[itype][l][zeta] is a list of float that
            specifies a pseudo-atomic orbital.
        mo_jy : np.ndarray
            The original matrix elements evaluated between MO and jY
            as read from the orb_matrix_rcutXderivY.dat file.
            Shape: (nk, nbands, nao, nbes)
        jy_mu2comp : dict
            Index map mu -> (itype, iatom, l, zeta, m).
            NOTE: zeta is supposed to be 0 for all mu.
        rcut : float
            Cutoff radius.
        ibands : list or range
            Indices for the bands (MO) to be considered.

    '''
    nk, nbands, nao, nbes = mo_jy.shape
    ibands = range(nbands) if ibands is None else ibands

    M = block_diag(*_gen_q2zeta(coef, jy_mu2comp, nbes, rcut))

    return mo_jy.reshape(nk, nbands, nao*nbes)[:,ibands,:] @ M


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
        coef_frozen: list
            Coefficients in terms of the end-smoothed mixed spherical Bessel basis
            for the "frozen" orbitals that do not participate in the optimization.
            coef_frozen[itype][l][zeta] is a list of floats that specifies an orbital.
        frozen_frozen : list
            A list of 2-tuples (np.ndarray, np.ndarray). Each pair corresponds to a
            configuration; the arrays are the overlap and operator matrix elements
            between the frozen orbitals.
        mo_frozen: list
            A list of 2-tuples (np.ndarray, np.ndarray). Each pair corresponds to a
            configuration; the arrays are the overlap and operator matrix elements
            between the MOs and frozen orbitals.
        mo_frozen_dual : list of np.ndarray
            Similar to mo_frozen, but the matrix elements are transformed to between
            the MOs and the dual of frozen orbitals.

    '''

    def __init__(self):
        self.reset()


    def reset(self):
        self.config = []
        self.rcut = None

        self._reset_frozen()
        self._reset_dV()


    def _reset_frozen(self):
        self.coef_frozen = None
        self.spill_frozen = None
        self.frozen_frozen = None
        self.mo_frozen = None
        self.mo_frozen_dual = None

    
    def _reset_dV(self):
        self.dV = ()
        self.dV_op = ()


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

        <frozen|frozen>         <frozen|op|frozen>
        <mo|frozen>             <mo|op|frozen>
        <mo|frozen's dual>      

        '''
        self.coef_frozen = coef_frozen

        frozen_frozen, frozen_op_frozen = zip(*[(
            _ao_ao(coef_frozen, ovlp['jy_jy'], ovlp['mu2comp'], ovlp['rcut']),
            _ao_ao(coef_frozen, op['jy_jy'], op['mu2comp'], op['rcut'])
            ) for ovlp, op in self.config])

        mo_frozen, mo_op_frozen = zip(*[(
            _mo_ao(coef_frozen, ovlp['mo_jy'], ovlp['mu2comp'], ovlp['rcut']),
            _mo_ao(coef_frozen, op['mo_jy'], op['mu2comp'], op['rcut'])
            ) for ovlp, op in self.config])

        mo_frozen_dual = [_mrdivide(X, S) for X, S in zip(mo_frozen, frozen_frozen)]

        wks = [op['wk'] for _, op in self.config]

        self.mo_frozen_dual = mo_frozen_dual
        self.spill_frozen = [_wsum_fro(wk, X_dual @ S_op, X_dual, rowwise=True)
                             - 2.0 * _wsum_fro(wk, X_dual, X_op, rowwise=True)
                             for wk, X_dual, X_op, S_op in
                             zip(wks, mo_frozen_dual, mo_op_frozen, frozen_op_frozen)]


    def _tab_dV(self, coef):
        '''
        Tabulates the derivatives of

                        <mo|ao>
                        <mo|op|ao>

        or, if frozen orbitals are present, the derivatives of

                <mo|(1-|frozen_dual><frozen|)|ao>
                <mo|(1-|frozen_dual><frozen|)|op|ao>

        with respect to the coefficients.


        Note
        ----
        The only useful information of coef is its nesting pattern.
        
        '''
        pat = nestpat(coef) # nesting pattern
        sz = len(flatten(coef))
        _c = np.eye(sz)

        def _dV(op, X_dual): # here `op` could also take an ovlp
            dV = np.array([_mo_ao(nest(ci.tolist(), pat), op['mo_jy'],
                                  op['mu2comp'], op['rcut']) for ci in _c])

            if self.coef_frozen is not None:
                dV -= np.array([X_dual @ _ao_ao((self.coef_frozen, nest(ci.tolist(), pat)),
                                                op['jy_jy'], op['mu2comp'], op['rcut'])
                                for ci in _c])
            return dV

        self.dV, self.dV_op = zip(*[(_dV(ovlp, X_dual), _dV(op, X_dual)) \
                for X_dual, (ovlp, op) in zip(self.mo_frozen_dual, self.config)])


    def _overlap_spillage(self, iconf, coef, ibands):
        '''
        Standard spillage function (overlap spillage) and its gradient.

        Note
        ----
        This function is not supposed to be used in the optimization.
        As a specific case of the generalized spillage (op = I), it
        provides a cross-check for the implementation of the generalized
        spillage.

        '''
        ovlp = self.config[iconf][0]
        wk = ovlp['wk']

        spill = (wk @ ovlp['mo_mo'][:, ibands]).real.sum()

        W = _ao_ao(coef, ovlp['jy_jy'], ovlp['mu2comp'], ovlp['rcut'])
        V = _mo_ao(coef, ovlp['mo_jy'], ovlp['mu2comp'], ovlp['rcut'], ibands)

        if self.coef_frozen is not None:
            X = _mo_ao(self.coef_frozen, ovlp['mo_jy'], ovlp['mu2comp'], ovlp['rcut'], ibands)
            X_dual = self.mo_frozen_dual[iconf][:, ibands, :]
            spill -= _wsum_fro(wk, X_dual, X)

            V -= X_dual @ _ao_ao((self.coef_frozen, coef), ovlp['jy_jy'],
                                 ovlp['mu2comp'], ovlp['rcut'])

        V_dual = _mrdivide(V, W)
        spill -= _wsum_fro(wk, V_dual, V)

        return spill / len(ibands)


    def _generalize_spillage(self, iconf, coef, ibands, with_grad=False):
        '''
        Generalized spillage function and its gradient.

        '''
        ovlp, op = self.config[iconf]
        wk = op['wk']

        spill = (wk @ op['mo_mo'][:, ibands]).real.sum()

        V = _mo_ao(coef, ovlp['mo_jy'], ovlp['mu2comp'], ovlp['rcut'], ibands)
        V_op = _mo_ao(coef, op['mo_jy'], op['mu2comp'], op['rcut'], ibands)

        if self.coef_frozen is not None:
            spill += self.spill_frozen[iconf][ibands].sum()

            X_dual = self.mo_frozen_dual[iconf][:, ibands, :]

            # TODO the left half of the _ao_ao below might be tabulated,
            # thereby reducing to an mo_ao calculation
            V -= X_dual @ _ao_ao((self.coef_frozen, coef),
                                 ovlp['jy_jy'], ovlp['mu2comp'], ovlp['rcut'])
            V_op -= X_dual @ _ao_ao((self.coef_frozen, coef),
                                    op['jy_jy'], op['mu2comp'], op['rcut'])

        W = _ao_ao(coef, ovlp['jy_jy'], ovlp['mu2comp'], ovlp['rcut'])
        W_op = _ao_ao(coef, op['jy_jy'], op['mu2comp'], op['rcut'])

        V_dual = _mrdivide(V, W)
        spill += _wsum_fro(wk, V_dual @ W_op, V_dual) - 2.0 * _wsum_fro(wk, V_dual, V_op)

        spill /= len(ibands)

        if with_grad:
            pattern = nestpat(coef) # nesting pattern
            sz = len(flatten(coef))
            spill_grad = np.zeros(sz)
            _c = np.eye(sz)

            for i in range(sz):
                dV = self.dV[iconf][i][:,ibands,:]
                dV_op = self.dV_op[iconf][i][:,ibands,:]

                coef_d = nest(_c[i].tolist(), pattern)

                # TODO the left half of the _ao_ao below might be tabulated,
                # thereby reducing to an mo_ao calculation and completely get
                # rid of the meaningless coef_d
                dW = _ao_ao((coef_d, coef), ovlp['jy_jy'], ovlp['mu2comp'], ovlp['rcut'])
                dW += dW.transpose((0,2,1)).conj()

                dW_op = _ao_ao((coef_d, coef), op['jy_jy'], op['mu2comp'], op['rcut'])
                dW_op += dW_op.transpose((0,2,1)).conj()

                spill_grad[i] = _wsum_fro(wk, V_dual @ dW_op, V_dual) \
                        - 2.0 * _wsum_fro(wk, V_dual, dV_op) \
                        + 2.0 * _wsum_fro(wk, _mrdivide(dV - V_dual @ dW, W),
                                              V_dual @ W_op - V_op)
            
            spill_grad /= len(ibands)
            spill_grad = nest(spill_grad.tolist(), pattern)

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


    def test_ao_ao(self):
        ovlp = read_orb_mat('./testfiles/orb_matrix_rcut6deriv0.dat')

        # 2s2p1d
        coef = [[np.eye(2, ovlp['nbes']-1).tolist(),
                 np.eye(2, ovlp['nbes']-1).tolist(),
                 np.eye(1, ovlp['nbes']-1).tolist()]]

        nao = sum(len(coef_tl) * (2*l+1) * ovlp['natom'][it]
                  for it, coef_t in enumerate(coef) for l, coef_tl in enumerate(coef_t))

        S = _ao_ao(coef, ovlp['jy_jy'], ovlp['mu2comp'], ovlp['rcut'])
        self.assertEqual(S.shape, (ovlp['nk'], nao, nao))

        # overlap matrix should be hermitian
        for Sk in S:
            self.assertLess(np.linalg.norm(Sk-Sk.T.conj(), np.inf), 1e-12)

        # check one-sided transformation
        njy = ovlp['jy_jy'][0,0,0].size

        tmp = _ao_ao((coef, None), ovlp['jy_jy'], ovlp['mu2comp'], ovlp['rcut'])
        self.assertEqual(tmp.shape, (ovlp['nk'], nao, njy))

        tmp = _ao_ao((None, coef), ovlp['jy_jy'], ovlp['mu2comp'], ovlp['rcut'])
        self.assertEqual(tmp.shape, (ovlp['nk'], njy, nao))


    def test_mo_ao(self):
        ovlp = read_orb_mat('./testfiles/orb_matrix_rcut6deriv0.dat')

        # 2s2p1d
        coef = [[np.eye(2, ovlp['nbes']-1).tolist(),
                 np.eye(2, ovlp['nbes']-1).tolist(),
                 np.eye(1, ovlp['nbes']-1).tolist()]]

        nao = sum(len(coef_tl) * (2*l+1) * ovlp['natom'][it]
                  for it, coef_t in enumerate(coef) for l, coef_tl in enumerate(coef_t))

        ibands = range(1, 4)
        X = _mo_ao(coef, ovlp['mo_jy'], ovlp['mu2comp'], ovlp['rcut'], ibands)
        self.assertEqual(X.shape, (ovlp['nk'], len(ibands), nao))

        X = _mo_ao(coef, ovlp['mo_jy'], ovlp['mu2comp'], ovlp['rcut'])
        self.assertEqual(X.shape, (ovlp['nk'], ovlp['nbands'], nao))


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
        orbgen._tab_dV(coef)

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
        spill = orbgen._generalize_spillage(0, coef, ibands, False)
        spill_ref = orbgen._overlap_spillage(0, coef, ibands)
        print('spill ref = ', spill_ref)
        print('spill     = ', spill)







if __name__ == '__main__':
    unittest.main()


from datparse import read_orb_mat, _assert_consistency
from indexmap import _index_map
from radial import jl_reduce, JL_REDUCE
from listmanip import flatten, nest, nestpat

import numpy as np
from scipy.linalg import block_diag
from copy import deepcopy

import matplotlib.pyplot as plt
import time


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

    Given matrix elements evaluated between jY, builds the matrix
    elements between pseudo-atomic orbitals specified by the given
    coefficients in the orthonormal end-smoothed mixed spherical
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
            the respective coefficients.
        jy_jy : np.ndarray
            The original matrix in jY basis as read from an
            orb_matrix_rcutXderivY.dat file.
            Shape: (nk, nao, nbes, nao, nbes)
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
        return M.T @ tmp @ M
    else:
        coef_bra, coef_ket = coef
        assert coef_bra is not None or coef_ket is not None

        M_bra = block_diag(*_gen_q2zeta(coef_bra, jy_mu2comp, nbes, rcut))
        M_ket = block_diag(*_gen_q2zeta(coef_ket, jy_mu2comp, nbes, rcut))

        if coef_bra is not None and coef_ket is not None:
            return M_bra.T @ tmp @ M_ket
        elif coef_bra is not None:
            return (M_bra.T @ tmp).reshape(nk, -1, nao, nbes)
        else:
            return (tmp @ M_ket).reshape(nk, nao, nbes, -1)


#def _ao_ao(coef, jy_jy, jy_mu2comp, rcut):
#    '''
#    Matrix elements between pseudo-atomic orbitals.
#
#    Given matrix elements evaluated between jY, builds the matrix
#    elements between pseudo-atomic orbitals specified by the given
#    coefficients in the orthonormal end-smoothed mixed spherical
#    Bessel basis.
#
#    Parameters
#    ----------
#        coef : list or tuple
#            The coefficients of pseudo-atomic orbitals in the
#            orthonormal end-smoothed mixed spherical Bessel basis,
#            where coef[itype][l][zeta] is a list of float that
#            specifies a pseudo-atomic orbital.
#            coef could also be a tuple like (coef_bra, coef_ket),
#            in which case the ket and bra will be transformed by
#            the respective coefficients.
#        jy_jy : np.ndarray
#            The original matrix in jY basis as read from an
#            orb_matrix_rcutXderivY.dat file.
#            Shape: (nk, nao, nbes, nao, nbes)
#        jy_mu2comp : dict
#            Index map mu -> (itype, iatom, l, zeta, m).
#            NOTE: zeta is supposed to be 0 for all mu.
#        rcut : float
#            Cutoff radius.
#
#    Notes
#    -----
#    The raw output of ABACUS corresponds to a 5-d array of shape
#    (nk, nao, nao, nbes, nbes). It shall be permuted before being
#    passed to this function. Currently read_orb_mat in datparse.py
#    takes care of this permutation.
#
#    '''
#    nk, nao, nbes = jy_jy.shape[0], jy_jy.shape[1], jy_jy.shape[-1]
#
#    coef_bra, coef_ket = coef if isinstance(coef, tuple) else (coef, None)
#
#    # basis transformation matrix from the truncated spherical Bessel
#    # function to the pseudo-atomic orbital
#    M_bra = block_diag(*_gen_q2zeta(coef_bra, jy_mu2comp, nbes, rcut))
#    M_ket = block_diag(*_gen_q2zeta(coef_ket, jy_mu2comp, nbes, rcut)) \
#            if coef_ket is not None else M_bra
#
#    return M_bra.T @ jy_jy.reshape(nk, nao*nbes, nao*nbes) @ M_ket


def _mo_ao(coef, mo_jy, jy_mu2comp, rcut, ibands=None):
    '''
    Matrix elements between MO and pseudo-atomic orbitals.

    Given matrix elements evaluated between MO and jY, builds the matrix
    elements between MO and pseudo-atomic orbitals specified by the given
    coefficients in the orthonormal end-smoothed mixed spherical Bessel
    basis.

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

    # basis transformation matrix from the truncated spherical Bessel
    # function to the pseudo-atomic orbital
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

        self._clear_frozen()


    def _clear_frozen(self):
        self.coef_frozen = None
        self.spill_frozen = None
        self.frozen_frozen = None
        self.mo_frozen = None
        self.mo_frozen_dual = None


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
        
        #self.dV.append()

    #def _dV(self, mo_jy, jy_mu2comp, rcut):



    def _tab_frozen(self, coef_frozen):
        '''
        Tabulates quantities related to the frozen-orbitals, including

        <frozen|frozen>         <frozen|op|frozen>
        <mo|frozen>             <mo|op|frozen>
        <mo|frozen's dual>      

        '''
        self.coef_frozen = coef_frozen

        self.frozen_frozen = [(
            _ao_ao(coef_frozen, ovlp['jy_jy'], ovlp['mu2comp'], ovlp['rcut']),
            _ao_ao(coef_frozen, op['jy_jy'], op['mu2comp'], op['rcut'])
            ) for ovlp, op in self.config]

        self.mo_frozen = [(
            _mo_ao(coef_frozen, ovlp['mo_jy'], ovlp['mu2comp'], ovlp['rcut']),
            _mo_ao(coef_frozen, op['mo_jy'], op['mu2comp'], op['rcut'])
            ) for ovlp, op in self.config]

        self.mo_frozen_dual = [(
            _mrdivide(X[0], S[0]),
            None #_mrdivide(X[1], S[0])
            ) for X, S in zip(self.mo_frozen, self.frozen_frozen)]

        #############################################################
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

        self.spill_frozen = [_wsum_fro(wk, X_dual @ S_op, X_dual, rowwise=True)
                             - 2.0 * _wsum_fro(wk, X_dual, X_op, rowwise=True)
                             for wk, X_dual, X_op, S_op in
                             zip(wks, mo_frozen_dual, mo_op_frozen, frozen_op_frozen)]

    
    def _tab_dV(self, coef):
        pass


    def _spillage_0(self, coef, ibands):
        '''
        Standard spillage function.

        Note
        ----
        This function is not supposed to be used in the optimization;
        It merely provides a cross-check for the correctness of the
        implementation of the generalized spillage function (_spillage),
        i.e., the generalized spillage should reduce to the standard
        spillage when op = I, in which case op_dat = ovlp_dat.

        '''
        spill_0 = 0.0
        nbands = len(ibands)

        for i, (ovlp, _) in enumerate(self.config):
            # weight of k-points, reshaped to enable broadcasting
            wk = ovlp['wk'].reshape(ovlp['nk'], 1, 1)

            W = _ao_ao(coef, ovlp['jy_jy'], ovlp['mu2comp'], ovlp['rcut'])
            V = _mo_ao(coef, ovlp['mo_jy'], ovlp['mu2comp'], ovlp['rcut'], ibands)

            if self.coef_frozen is not None:
                X = self.mo_frozen[i][0][:, ibands, :]
                X_dual = self.mo_frozen_dual[i][0][:, ibands, :]
                spill_0 -= (wk * X_dual * X.conj()).real.sum() / nbands 

                V -= X_dual @ _ao_ao((self.coef_frozen, coef), ovlp['jy_jy'],
                                          ovlp['mu2comp'], ovlp['rcut'])

            V_dual = _mrdivide(V, W)
            spill_0 += 1.0 - (wk * V_dual * V.conj()).real.sum() / nbands

        # averaged by the number of configurations
        return spill_0 / len(self.config)


    def _spillage_0_grad(self, coef, ibands):
        '''
        Gradient of the standard spillage function.

        '''
        nbands = len(ibands)

        patn = nestpat(coef) # nesting pattern
        sz = len(flatten(coef))
        spill_0_grad = np.zeros(sz)

        for i, (ovlp, _) in enumerate(self.config):
            # weight of k-points, reshaped to enable broadcasting
            wk = ovlp['wk'].reshape(ovlp['nk'], 1, 1)

            W = _ao_ao(coef, ovlp['jy_jy'], ovlp['mu2comp'], ovlp['rcut'])
            V = _mo_ao(coef, ovlp['mo_jy'], ovlp['mu2comp'], ovlp['rcut'], ibands)

            if self.coef_frozen is not None:
                X_dual = self.mo_frozen_dual[i][0][:, ibands, :]
                V -= X_dual @ _ao_ao((self.coef_frozen, coef), ovlp['jy_jy'],
                                     ovlp['mu2comp'], ovlp['rcut'])
            V_dual = _mrdivide(V, W)

            for ic in range(sz):
                c = np.zeros(sz)
                c[ic] = 1
                coef_d = nest(c.tolist(), patn)

                dV = _mo_ao(coef_d, ovlp['mo_jy'], ovlp['mu2comp'], ovlp['rcut'], ibands)

                dW = _ao_ao((coef_d, coef), ovlp['jy_jy'], ovlp['mu2comp'], ovlp['rcut'])
                dW += dW.transpose((0,2,1)).conj()

                if self.coef_frozen is not None:
                    dV -= X_dual @ _ao_ao((self.coef_frozen, coef_d), ovlp['jy_jy'],
                                               ovlp['mu2comp'], ovlp['rcut'])

                # FIXME dV should be tabulated
                spill_0_grad[ic] += (wk * (V_dual @ dW) * V_dual.conj()).real.sum() / nbands \
                        - 2.0 * (wk * V_dual * dV.conj()).real.sum() / nbands

        return nest((spill_0_grad/len(self.config)).tolist(), patn)


    def _ovlp_spillage(self, iconf, coef, ibands, with_grad=False):
        '''
        Standard spillage function (overlap spillage) and its gradient.

        Note
        ----
        This function is not supposed to be used in the optimization;
        It merely provides a cross-check for the correctness of the
        implementation of the generalized spillage function (_spillage),
        i.e., the generalized spillage should reduce to the overlap
        spillage when op = I, in which case op_dat = ovlp_dat.

        '''
        ovlp = self.config[iconf][0]
        nbands = len(ibands)

        # weight of k-points, reshaped to enable broadcasting
        wk = ovlp['wk'].reshape(ovlp['nk'], 1, 1)

        W = _ao_ao(coef, ovlp['jy_jy'], ovlp['mu2comp'], ovlp['rcut'])
        V = _mo_ao(coef, ovlp['mo_jy'], ovlp['mu2comp'], ovlp['rcut'], ibands)

        if self.coef_frozen is not None:
            X = self.mo_frozen[i][0][:, ibands, :]
            X_dual = self.mo_frozen_dual[i][0][:, ibands, :]
            spill_0 -= (wk * X_dual * X.conj()).real.sum() / nbands 

            V -= X_dual @ _ao_ao((self.coef_frozen, coef), ovlp['jy_jy'],
                                      ovlp['mu2comp'], ovlp['rcut'])

        V_dual = _mrdivide(V, W)
        spill += 1.0 - (wk * V_dual * V.conj()).real.sum() / nbands



    def _spillage2(self, iconf, coef, ibands, with_grad=False):
        '''
        Generalized spillage function and its gradient.

        '''
        ovlp, op = self.config[iconf]
        nbands = len(ibands)
        wk = op['wk']

        spill = (wk @ op['mo_mo']).real.sum()

        V = _mo_ao(coef, ovlp['mo_jy'], ovlp['mu2comp'], ovlp['rcut'], ibands)
        V_op = _mo_ao(coef, op['mo_jy'], op['mu2comp'], op['rcut'], ibands)

        if self.coef_frozen is not None:
            spill += self.spill_frozen[iconf][ibands].sum()

            X_dual = self.mo_frozen_dual[iconf][0][:, ibands, :]

            V -= X_dual @ _ao_ao((self.coef_frozen, coef),
                                 ovlp['jy_jy'], ovlp['mu2comp'], ovlp['rcut'])
            V_op -= X_dual @ _ao_ao((self.coef_frozen, coef),
                                    op['jy_jy'], op['mu2comp'], op['rcut'])

        W = _ao_ao(coef, ovlp['jy_jy'], ovlp['mu2comp'], ovlp['rcut'])
        W_op = _ao_ao(coef, op['jy_jy'], op['mu2comp'], op['rcut'])

        V_dual = _mrdivide(V, W)

        spill += _wsum_fro(wk, V_dual @ W_op, V_dual) - 2.0 * _wsum_fro(wk, V_dual, V_op)

        spill /= nbands

        if with_grad:
            patn = nestpat(coef) # nesting pattern
            sz = len(flatten(coef))
            spill_grad = np.zeros(sz)

            for i in range(sz):
                c = np.zeros(sz)
                c[i] = 1
                coef_d = nest(c.tolist(), patn)

                # FIXME dV & dV_op should be tabulated
                dV = _mo_ao(coef_d, ovlp['mo_jy'], ovlp['mu2comp'], ovlp['rcut'], ibands)
                dV_op = _mo_ao(coef_d, op['mo_jy'], op['mu2comp'], op['rcut'], ibands)

                if i == sz-1:
                    print('============')
                    print('dV     = ', dV[0])
                    coef_tmp = deepcopy(coef_d)
                    coef_tmp[0][0]=[]
                    coef_tmp[0][1]=[]
                    dV_tmp = _mo_ao(coef_tmp, ovlp['mo_jy'], ovlp['mu2comp'], ovlp['rcut'], ibands)
                    print('dV_tmp = ', dV_tmp[0])
                    print('============')

                if self.coef_frozen is not None:
                    dV -= X_dual @ _ao_ao((self.coef_frozen, coef_d), ovlp['jy_jy'],
                                          ovlp['mu2comp'], ovlp['rcut'])
                    dV_op -= X_dual @ _ao_ao((self.coef_frozen, coef_d), op['jy_jy'],
                                             op['mu2comp'], op['rcut'])

                # FIXME dW & dW_op can be reduced to bes_ao calculation, instead of ao_ao
                dW = _ao_ao((coef_d, coef), ovlp['jy_jy'], ovlp['mu2comp'], ovlp['rcut'])
                dW += dW.transpose((0,2,1)).conj()

                dW_op = _ao_ao((coef_d, coef), op['jy_jy'], op['mu2comp'], op['rcut'])
                dW_op += dW_op.transpose((0,2,1)).conj()

                spill_grad[i] = _wsum_fro(wk, V_dual @ dW_op, V_dual) \
                        - 2.0 * _wsum_fro(wk, V_dual, dV_op) \
                        + 2.0 * _wsum_fro(wk, _mrdivide(dV - V_dual @ dW, W),
                                          V_dual @ W_op - V_op)
            
            spill_grad /= nbands
            spill_grad = nest(spill_grad.tolist(), patn)

        return (spill, spill_grad) if with_grad else spill



    def _spillage(self, coef, ibands):
        '''
        Generalized spillage function.

        '''
        spill = 0.0
        nbands = len(ibands)

        for i, (ovlp, op) in enumerate(self.config):
            # the number & weight of k-points
            nk, wk = op['nk'], op['wk']

            # <mo|op|mo> / nbands
            spill += (wk @ op['mo_mo']).real.sum() / nbands

            W = _ao_ao(coef, ovlp['jy_jy'], ovlp['mu2comp'], ovlp['rcut'])
            V = _mo_ao(coef, ovlp['mo_jy'], ovlp['mu2comp'], ovlp['rcut'], ibands)

            W_op = _ao_ao(coef, op['jy_jy'], op['mu2comp'], op['rcut'])
            V_op = _mo_ao(coef, op['mo_jy'], op['mu2comp'], op['rcut'], ibands)

            # reshape the weight of k-points to enable broadcasting
            wk = wk.reshape(nk, 1, 1)

            if self.coef_frozen is not None:
                X_dual = self.mo_frozen_dual[i][0][:, ibands, :]
                X_op = self.mo_frozen[i][1][:, ibands, :]
                S_op = self.frozen_frozen[i][1]
                #spill += (wk * (X_dual @ S_op) * X_dual.conj()).real.sum() / nbands \
                #        - 2.0 * (wk * X_dual * X_op.conj()).real.sum() / nbands
                spill += self.spill_frozen[i][ibands].sum() / nbands

                V_op -= X_dual @ _ao_ao((self.coef_frozen, coef), op['jy_jy'],
                                             op['mu2comp'], op['rcut'])
                V -= X_dual @ _ao_ao((self.coef_frozen, coef), ovlp['jy_jy'],
                                          ovlp['mu2comp'], ovlp['rcut'])

            V_dual = _mrdivide(V, W)

            spill += (wk * (V_dual @ W_op) * V_dual.conj()).real.sum() / nbands \
                    - 2.0 * (wk * V_dual * V_op.conj()).real.sum() / nbands

        return spill / len(self.config)


    def _spillage_grad(self, coef, ibands):
        '''
        Gradient of the generalized spillage function.

        '''
        nbands = len(ibands)

        patn = nestpat(coef) # nesting pattern
        sz = len(flatten(coef))
        spill_grad = np.zeros(sz)

        for i, (ovlp, op) in enumerate(self.config):
            # weight of k-points, reshaped to enable broadcasting
            wk = ovlp['wk'].reshape(ovlp['nk'], 1, 1)

            W = _ao_ao(coef, ovlp['jy_jy'], ovlp['mu2comp'], ovlp['rcut'])
            V = _mo_ao(coef, ovlp['mo_jy'], ovlp['mu2comp'], ovlp['rcut'], ibands)

            W_op = _ao_ao(coef, op['jy_jy'], op['mu2comp'], op['rcut'])
            V_op = _mo_ao(coef, op['mo_jy'], op['mu2comp'], op['rcut'], ibands)

            if self.coef_frozen is not None:
                X_dual = self.mo_frozen_dual[i][0][:, ibands, :]
                V -= X_dual @ _ao_ao((self.coef_frozen, coef), ovlp['jy_jy'],
                                          ovlp['mu2comp'], ovlp['rcut'])
                V_op -= X_dual @ _ao_ao((self.coef_frozen, coef), op['jy_jy'],
                                             op['mu2comp'], op['rcut'])
            V_dual = _mrdivide(V, W)

            for ic in range(sz):
                c = np.zeros(sz)
                c[ic] = 1
                coef_d = nest(c.tolist(), patn)

                dV = _mo_ao(coef_d, ovlp['mo_jy'], ovlp['mu2comp'], ovlp['rcut'], ibands)
                dW = _ao_ao((coef_d, coef), ovlp['jy_jy'], ovlp['mu2comp'], ovlp['rcut'])
                dW += dW.transpose((0,2,1)).conj()

                dV_op = _mo_ao(coef_d, op['mo_jy'], op['mu2comp'], op['rcut'], ibands)
                dW_op = _ao_ao((coef_d, coef), op['jy_jy'], op['mu2comp'], op['rcut'])
                dW_op += dW_op.transpose((0,2,1)).conj()

                if self.coef_frozen is not None:
                    dV -= X_dual @ _ao_ao((self.coef_frozen, coef_d), ovlp['jy_jy'], 
                                               ovlp['mu2comp'], ovlp['rcut'])
                    dV_op -= X_dual @ _ao_ao((self.coef_frozen, coef_d), op['jy_jy'],
                                                  op['mu2comp'], op['rcut'])

                # FIXME dV & dV_op should be tabulated
                # FIXME dW & dW_op can be reduced to mo_ao calls, instead of ao_ao calls
                spill_grad[ic] += (wk * (V_dual @ dW_op) * V_dual.conj()).sum().real / nbands \
                        - 2.0 * (wk * V_dual * dV_op.conj()).sum().real / nbands \
                        + 2.0 * (wk * _mrdivide(dV - V_dual @ dW, W) 
                                 * (V_dual @ W_op - V_op).conj()).real.sum() / nbands

        return nest((spill_grad/len(self.config)).tolist(), patn)




    def opt(self, coef_init):
        '''
        '''
        pass











############################################################
#                           Test
############################################################
import unittest

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

        # coef[itype][l][zeta] gives a list of coefficients that specifies an orbital
        # NOTE it's the coefficients for the end-smoothed mixed spherical Bessel basis,
        # not the coefficients for the truncated spherical Bessel function.
        coef = [ [np.random.randn(nzeta[itype][l], nbes-1).tolist()
                  for l in range(lmax[itype]+1)]
                for itype in range(ntype) ]

        for mu, q2zeta in enumerate(_gen_q2zeta(coef, mu2comp, nbes, rcut)):
            itype, iatom, l, zeta, m = mu2comp[mu]
            self.assertEqual(q2zeta.shape, (nbes, nzeta[itype][l]))
            self.assertTrue( np.allclose(q2zeta, \
                    jl_reduce(l, nbes, rcut) @ np.array(coef[itype][l]).T))


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


    def test_ao_ao(self):
        mat = read_orb_mat('./testfiles/orb_matrix_rcut6deriv0.dat')
        #mat = read_orb_mat('./testfiles/orb_matrix_rcut7deriv1.dat')

        # 2s2p1d
        coef = [[np.eye(2, mat['nbes']-1).tolist(), \
                np.eye(2, mat['nbes']-1).tolist(), \
                np.eye(1, mat['nbes']-1).tolist()]]

        S = _ao_ao(coef, mat['jy_jy'], mat['mu2comp'], mat['rcut'])

        # overlap matrix should be hermitian
        for ik in range(mat['nk']):
            self.assertLess(np.linalg.norm(S[ik]-S[ik].T.conj(), np.inf), 1e-12)

        #plt.imshow(S[0])
        #plt.show()


    def test_mo_ao(self):
        mat = read_orb_mat('./testfiles/orb_matrix_rcut6deriv0.dat')

        # 2s2p1d
        coef = [[np.eye(2, mat['nbes']-1).tolist(), \
                np.eye(2, mat['nbes']-1).tolist(), \
                np.eye(1, mat['nbes']-1).tolist()]]

        # mo-basis overlap matrix
        X = _mo_ao(coef, mat['mo_jy'], mat['mu2comp'], mat['rcut'], range(1,4))

        #fig, ax = plt.subplots(1, 2)
        #im = ax[0].imshow(np.real(np.log(np.abs(X[0]))))
        #fig.colorbar(im, ax=ax[0], location='bottom')

        #im = ax[1].imshow(np.imag(np.log(np.abs(X[0]))))
        #fig.colorbar(im, ax=ax[1], location='bottom')

        #plt.show()

    def test_spillage_0(self):
        orbgen = Spillage()

        folder = '/home/zuxin/abacus-community/abacus_orbital_generation/tmp/Si-dimer-2.0/'

        mat = read_orb_mat(folder + 'orb_matrix_rcut6deriv0.dat')
        dmat = read_orb_mat(folder + 'orb_matrix_rcut6deriv1.dat')

        orbgen.add_config(mat, dmat)

        coef = [[np.eye(2, mat['nbes']-1).tolist(), \
                np.eye(2, mat['nbes']-1).tolist(), \
                np.eye(1, mat['nbes']-1).tolist()]]

        ibands = range(5)
        print(orbgen._spillage_0(coef, ibands))

        np.random.seed(0)

        #coef_frozen = [[np.eye(2, mat['nbes']-1).tolist(), \
        #        np.eye(2, mat['nbes']-1).tolist(), \
        #        np.eye(1, mat['nbes']-1).tolist()]]
        coef_frozen = [[np.random.randn(2, mat['nbes']-1).tolist(), \
                np.random.randn(2, mat['nbes']-1).tolist(), \
                np.random.randn(1, mat['nbes']-1).tolist()]]
        orbgen._tab_frozen(coef_frozen)
        #print(orbgen._spillage_0(coef, ibands))

        spill = orbgen._spillage_0(coef, ibands)

        dc = 1e-6
        coef_p = deepcopy(coef)
        coef_p[0][0][0][3] += dc
        spill_p = orbgen._spillage_0(coef_p, ibands)

        coef_m = deepcopy(coef)
        coef_m[0][0][0][3] -= dc
        spill_m = orbgen._spillage_0(coef_m, ibands)

        dspill = (spill_p - spill_m) / (2 * dc)

        print('')
        print('dspill0 (finite diff) = ', dspill)

        print('dspill0 ( analytic  ) = ', orbgen._spillage_0_grad(coef, ibands)[0][0][0][3])

        dc = 1e-6
        coef_p = deepcopy(coef)
        coef_p[0][0][0][3] += dc
        spill_p = orbgen._spillage(coef_p, ibands)

        coef_m = deepcopy(coef)
        coef_m[0][0][0][3] -= dc
        spill_m = orbgen._spillage(coef_m, ibands)

        dspill_fd = (spill_p - spill_m) / (2 * dc)

        print('')
        print('dspill  (finite diff) = ', dspill_fd)

        dspill = orbgen._spillage_grad(coef, ibands)
        
        print('dspill  ( analytic  ) = ', dspill[0][0][0][3])

        start = time.time()        
        res = orbgen._spillage2(0, coef, ibands, True)
        print('time = ', time.time() - start)

        self.assertLess(
                np.linalg.norm(np.array(flatten(dspill)) 
                               - np.array(flatten(res[1])), np.inf),
                1e-12)
        
        print('_spillage           = ', orbgen._spillage(coef, ibands))
        print('_spillage2 res[0]   = ', res[0])
        print('_spillage2 no grad  = ', orbgen._spillage2(0, coef, ibands, False))

        #print('generalized spillage = ', orbgen._spillage(coef, ibands))

        #from scipy.optimize import minimize
        #pat = nestpat(coef)
        #def f(c):
        #    tmp = orbgen._spillage_0(nest(c.tolist(), pat), ibands)
        #    print('spillage = ', tmp)
        #    return tmp

        #def df(c):
        #    tmp = orbgen._spillage_0_grad(nest(c.tolist(), pat), ibands)
        #    print('max grad = ', np.linalg.norm(flatten(tmp), np.inf))
        #    return np.array(flatten(tmp))

        #c0 = flatten(coef)
        #res = minimize(f, np.array(c0), jac=df, method='BFGS', tol=1e-6)
        #print(res.x)



if __name__ == '__main__':
    unittest.main()


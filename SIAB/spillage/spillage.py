from datparse import read_orb_mat, are_consistent
from indexmap import _index_map
from radial import jl_reduce
from listmanip import flatten, nest, nestpat

import numpy as np
from scipy.linalg import block_diag
from copy import deepcopy

import matplotlib.pyplot as plt


def _make_dual(X, S):
    '''
    Given two 3-d arrays X and S, returns a 3-d array X_dual such that

        X_dual[i] = X[i] * inv(S[i])

    '''
    assert len(X.shape) == 3 and len(S.shape) == 3
    return np.array([np.linalg.solve(S[i].T, X[i].T).T for i in range(X.shape[0])])


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

    '''
    for mu in mu2comp:
        itype, _, l, _, _ = mu2comp[mu]
        nzeta = len(coef[itype][l])
        if l >= len(coef[itype]) or nzeta == 0:
            # The yielded matrices will be diagonally concatenated
            # by scipy.linalg.block_diag. Therefore, even if the
            # coefficient is not provided, the generator should
            # yield the zero matrix with the appropriate size
            yield np.zeros((nbes, 0))
        else:
            C = np.zeros((nbes-1, nzeta))
            C[:len(coef[itype][l][0])] = np.array(coef[itype][l]).T
            yield jl_reduce(l, nbes, rcut) @ C


def _ao_ao(coef, jy_jy, jy_mu2comp, rcut):
    '''
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

    coef_bra, coef_ket = coef if isinstance(coef, tuple) else (coef, None)

    # basis transformation matrix from the truncated spherical Bessel
    # function to the pseudo-atomic orbital
    M_bra = block_diag(*_gen_q2zeta(coef_bra, jy_mu2comp, nbes, rcut))
    M_ket = block_diag(*_gen_q2zeta(coef_ket, jy_mu2comp, nbes, rcut)) \
            if coef_ket is not None else M_bra

    return M_bra.T @ jy_jy.reshape((nk, nao*nbes, nao*nbes)) @ M_ket


def _mo_ao(coef, mo_jy, jy_mu2comp, rcut, ibands=None):
    '''
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
        T : list
            A list of transformation matrices (corresponding to different order)
            from the truncated spherical Bessel functions to the orthonormal
            end-smoothed mixed spherical Bessel basis. 
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
        self.T = []

        self.coef_frozen = None
        self.frozen_frozen = None
        self.mo_frozen = None
        self.mo_frozen_dual = None


    def add_config(self, ovlp_dat, op_dat):
        '''
        '''
        # ensures consistency between two datasets
        assert are_consistent(ovlp_dat, op_dat)

        if self.rcut is None:
            self.rcut = ovlp_dat['rcut']
        else:
            assert self.rcut == ovlp_dat['rcut']

        self.config.append((ovlp_dat, op_dat))

        # The table of jl transformation matrix should cover the largest lmax and nbes
        # among all configurations. NOTE: the value of 'lmax' is a list.
        self._update_transform_table(max(ovlp_dat['nbes'], op_dat['nbes']),
                                     max(ovlp_dat['lmax'] + op_dat['lmax']))


    def _tab_frozen(self, coef_frozen):
        '''
        Tabulates matrix elements related to the frozen-orbitals, including

        <frozen|frozen>         <frozen|op|frozen>
        <mo|frozen>             <mo|op|frozen>
        <mo|frozen's dual>      <mo|op|frozen's dual>

        '''
        self.coef_frozen = coef_frozen

        self.frozen_frozen = [(
            _ao_ao(coef_frozen, ovlp_dat['jy_jy'], ovlp_dat['mu2comp'], ovlp_dat['rcut']),
            _ao_ao(coef_frozen, op_dat['jy_jy'], op_dat['mu2comp'], op_dat['rcut'])
            ) for ovlp_dat, op_dat in self.config]

        self.mo_frozen = [(
            _mo_ao(coef_frozen, ovlp_dat['mo_jy'], ovlp_dat['mu2comp'],
                        ovlp_dat['rcut']),
            _mo_ao(coef_frozen, op_dat['mo_jy'], op_dat['mu2comp'],
                        op_dat['rcut'])
            ) for ovlp_dat, op_dat in self.config]

        self.mo_frozen_dual = [(
            _make_dual(X[0], S[0]),
            _make_dual(X[1], S[0])
            ) for X, S in zip(self.mo_frozen, self.frozen_frozen)]

    
    def _update_transform_table(self, nbes, lmax):
        '''
        Updates the list of jl transformation matrix.

        The list, indexed by l, stores transformation matrices from the truncated
        spherical Bessel function to the orthonormal end-smoothed mixed spherical
        Bessel basis. Given rcut and l, the transformation is guaranteed to be
        consistent with respect to different nbes, i.e., the transformation
        matrix for nbes = N is a submatrix of the transformation matrix for
        nbes = M, if N < M.

        '''
        _lmax = len(self.T) - 1
        _nbes = self.T[0].shape[0] if self.T else 0 # tabulated matrix size

        if not self.T or _nbes < nbes:
            # If the list is empty, or the tabulated matrix size is too small,
            # tabulate the whole list.
            self.T = [jl_reduce(l, nbes, self.rcut) for l in range(max(lmax, _lmax)+1)]
        else:
            # If the tabulated matrix size is large enough, append to the existing
            # list if the new lmax is larger.
            self.T += [jl_reduce(l, _nbes, self.rcut) for l in range(_lmax+1, lmax+1)]


    # FIXME 
    # this should be a special case of the generalized spillage function
    # with op = ovlp
    # instead of a separate function
    def _spillage_0(self, coef, ibands):
        '''
        Standard spillage function.

        '''
        spill_0 = 0.0
        nbands = len(ibands)

        for i, (ovlp, _) in enumerate(self.config):
            # weight of k-points, reshaped to enable broadcasting
            wk = ovlp['wk'].reshape(ovlp['nk'], 1, 1)

            W = _ao_ao(coef, ovlp['jy_jy'], ovlp['mu2comp'], ovlp['rcut'])
            Y = _mo_ao(coef, ovlp['mo_jy'], ovlp['mu2comp'], ovlp['rcut'], ibands)

            V = Y
            if self.coef_frozen is not None:
                X = self.mo_frozen[i][0][:, ibands, :]
                X_dual = self.mo_frozen_dual[i][0][:, ibands, :]
                spill_0 -= (wk * X_dual * X.conj()).real.sum() / nbands 

                V -= X_dual @ _ao_ao((self.coef_frozen, coef), ovlp['jy_jy'],
                                          ovlp['mu2comp'], ovlp['rcut'])

            V_dual = _make_dual(V, W)
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
            Y = _mo_ao(coef, ovlp['mo_jy'], ovlp['mu2comp'], ovlp['rcut'], ibands)

            V = Y
            if self.coef_frozen is not None:
                X_dual = self.mo_frozen_dual[i][0][:, ibands, :]
                V -= X_dual @ _ao_ao((self.coef_frozen, coef), ovlp['jy_jy'],
                                     ovlp['mu2comp'], ovlp['rcut'])
            V_dual = _make_dual(V, W)

            for ic in range(sz):
                c = np.zeros(sz)
                c[ic] = 1
                coef_d = nest(c.tolist(), patn)

                dY = _mo_ao(coef_d, ovlp['mo_jy'], ovlp['mu2comp'], ovlp['rcut'], ibands)

                dW = _ao_ao((coef_d, coef), ovlp['jy_jy'], ovlp['mu2comp'], ovlp['rcut'])
                dW += dW.transpose((0,2,1)).conj()

                dV = dY
                if self.coef_frozen is not None:
                    dV -= X_dual @ _ao_ao((self.coef_frozen, coef_d), ovlp['jy_jy'],
                                               ovlp['mu2comp'], ovlp['rcut'])

                # FIXME dV should be tabulated
                spill_0_grad[ic] += (wk * (V_dual @ dW) * V_dual.conj()).real.sum() / nbands \
                        - 2.0 * (wk * V_dual * dV.conj()).real.sum() / nbands

        return nest((spill_0_grad/len(self.config)).tolist(), patn)


    def _spillage2(self, iconf, coef, ibands, with_grad=False):
        '''
        Generalized spillage function and its gradient.

        '''
        ovlp, op = self.config[iconf]
        nbands = len(ibands)

        # the number & weight of k-points
        nk, wk = op['nk'], op['wk']

        # <mo|op|mo> / nbands
        spill = (wk @ op['mo_mo']).real.sum() / nbands

        W = _ao_ao(coef, ovlp['jy_jy'], ovlp['mu2comp'], ovlp['rcut'])
        Y = _mo_ao(coef, ovlp['mo_jy'], ovlp['mu2comp'], ovlp['rcut'], ibands)

        W_op = _ao_ao(coef, op['jy_jy'], op['mu2comp'], op['rcut'])
        Y_op = _mo_ao(coef, op['mo_jy'], op['mu2comp'], op['rcut'], ibands)

        # reshape the weight of k-points to enable broadcasting
        wk = wk.reshape(nk, 1, 1)

        V = Y
        V_op = Y_op
        if self.coef_frozen is not None:
            X_dual = self.mo_frozen_dual[iconf][0][:, ibands, :]
            X_op = self.mo_frozen[iconf][1][:, ibands, :]
            S_op = self.frozen_frozen[iconf][1]
            spill += (wk * (X_dual @ S_op) * X_dual.conj()).real.sum() / nbands \
                    - 2.0 * (wk * X_dual * X_op.conj()).real.sum() / nbands

            V_op -= X_dual @ _ao_ao((self.coef_frozen, coef), op['jy_jy'],
                                         op['mu2comp'], op['rcut'])
            V -= X_dual @ _ao_ao((self.coef_frozen, coef), ovlp['jy_jy'],
                                      ovlp['mu2comp'], ovlp['rcut'])

        V_dual = _make_dual(V, W)

        spill += (wk * (V_dual @ W_op) * V_dual.conj()).real.sum() / nbands \
                - 2.0 * (wk * V_dual * V_op.conj()).real.sum() / nbands

        if with_grad:
            patn = nestpat(coef) # nesting pattern
            sz = len(flatten(coef))
            spill_grad = np.zeros(sz)

            for i in range(sz):
                c = np.zeros(sz)
                c[i] = 1
                coef_d = nest(c.tolist(), patn)

                dY = _mo_ao(coef_d, ovlp['mo_jy'], ovlp['mu2comp'], ovlp['rcut'], ibands)
                dW = _ao_ao((coef_d, coef), ovlp['jy_jy'], ovlp['mu2comp'], ovlp['rcut'])
                dW += dW.transpose((0,2,1)).conj()

                dY_op = _mo_ao(coef_d, op['mo_jy'], op['mu2comp'], op['rcut'], ibands)
                dW_op = _ao_ao((coef_d, coef), op['jy_jy'], op['mu2comp'], op['rcut'])
                dW_op += dW_op.transpose((0,2,1)).conj()

                dV = dY
                dV_op = dY_op
                if self.coef_frozen is not None:
                    dV -= X_dual @ _ao_ao((self.coef_frozen, coef_d), ovlp['jy_jy'], 
                                               ovlp['mu2comp'], ovlp['rcut'])
                    dV_op -= X_dual @ _ao_ao((self.coef_frozen, coef_d), op['jy_jy'],
                                                  op['mu2comp'], op['rcut'])

                # FIXME dV & dV_op should be tabulated
                # FIXME dW & dW_op can be reduced to mo_ao calls, instead of ao_ao calls
                spill_grad[i] += (wk * (V_dual @ dW_op) * V_dual.conj()).sum().real / nbands \
                        - 2.0 * (wk * V_dual * dV_op.conj()).sum().real / nbands \
                        + 2.0 * (wk * _make_dual(dV - V_dual @ dW, W) 
                                 * (V_dual @ W_op - V_op).conj()).real.sum() / nbands

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
            Y = _mo_ao(coef, ovlp['mo_jy'], ovlp['mu2comp'], ovlp['rcut'], ibands)

            W_op = _ao_ao(coef, op['jy_jy'], op['mu2comp'], op['rcut'])
            Y_op = _mo_ao(coef, op['mo_jy'], op['mu2comp'], op['rcut'], ibands)

            # reshape the weight of k-points to enable broadcasting
            wk = wk.reshape(nk, 1, 1)

            V = Y
            V_op = Y_op
            if self.coef_frozen is not None:
                X_dual = self.mo_frozen_dual[i][0][:, ibands, :]
                X_op = self.mo_frozen[i][1][:, ibands, :]
                S_op = self.frozen_frozen[i][1]
                spill += (wk * (X_dual @ S_op) * X_dual.conj()).real.sum() / nbands \
                        - 2.0 * (wk * X_dual * X_op.conj()).real.sum() / nbands

                V_op -= X_dual @ _ao_ao((self.coef_frozen, coef), op['jy_jy'],
                                             op['mu2comp'], op['rcut'])
                V -= X_dual @ _ao_ao((self.coef_frozen, coef), ovlp['jy_jy'],
                                          ovlp['mu2comp'], ovlp['rcut'])

            V_dual = _make_dual(V, W)

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
            Y = _mo_ao(coef, ovlp['mo_jy'], ovlp['mu2comp'], ovlp['rcut'], ibands)

            W_op = _ao_ao(coef, op['jy_jy'], op['mu2comp'], op['rcut'])
            Y_op = _mo_ao(coef, op['mo_jy'], op['mu2comp'], op['rcut'], ibands)

            V = Y
            V_op = Y_op
            if self.coef_frozen is not None:
                X_dual = self.mo_frozen_dual[i][0][:, ibands, :]
                V -= X_dual @ _ao_ao((self.coef_frozen, coef), ovlp['jy_jy'],
                                          ovlp['mu2comp'], ovlp['rcut'])
                V_op -= X_dual @ _ao_ao((self.coef_frozen, coef), op['jy_jy'],
                                             op['mu2comp'], op['rcut'])
            V_dual = _make_dual(V, W)

            for ic in range(sz):
                c = np.zeros(sz)
                c[ic] = 1
                coef_d = nest(c.tolist(), patn)

                dY = _mo_ao(coef_d, ovlp['mo_jy'], ovlp['mu2comp'], ovlp['rcut'], ibands)
                dW = _ao_ao((coef_d, coef), ovlp['jy_jy'], ovlp['mu2comp'], ovlp['rcut'])
                dW += dW.transpose((0,2,1)).conj()

                dY_op = _mo_ao(coef_d, op['mo_jy'], op['mu2comp'], op['rcut'], ibands)
                dW_op = _ao_ao((coef_d, coef), op['jy_jy'], op['mu2comp'], op['rcut'])
                dW_op += dW_op.transpose((0,2,1)).conj()

                dV = dY
                dV_op = dY_op
                if self.coef_frozen is not None:
                    dV -= X_dual @ _ao_ao((self.coef_frozen, coef_d), ovlp['jy_jy'], 
                                               ovlp['mu2comp'], ovlp['rcut'])
                    dV_op -= X_dual @ _ao_ao((self.coef_frozen, coef_d), op['jy_jy'],
                                                  op['mu2comp'], op['rcut'])

                # FIXME dV & dV_op should be tabulated
                # FIXME dW & dW_op can be reduced to mo_ao calls, instead of ao_ao calls
                spill_grad[ic] += (wk * (V_dual @ dW_op) * V_dual.conj()).sum().real / nbands \
                        - 2.0 * (wk * V_dual * dV_op.conj()).sum().real / nbands \
                        + 2.0 * (wk * _make_dual(dV - V_dual @ dW, W) 
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

    def test_update_transform_table(self):
        orbgen = Spillage()

        orbgen.rcut = 5.0
        nbes = 30
        lmax = 2

        orbgen._update_transform_table(nbes, lmax)
        self.assertEqual(len(orbgen.T), lmax+1)
        
        # a smaller lmax & nbes would not alter the table
        orbgen._update_transform_table(nbes, 1)
        self.assertEqual(len(orbgen.T), lmax+1)

        orbgen._update_transform_table(10, lmax)
        self.assertEqual(orbgen.T[0].shape[0], nbes)

        orbgen._update_transform_table(10, 1)
        self.assertEqual(len(orbgen.T), lmax+1)
        self.assertEqual(orbgen.T[0].shape[0], nbes)

        # a larger lmax/nbes would extend the table
        lmax = 4
        orbgen._update_transform_table(nbes, lmax)
        self.assertEqual(len(orbgen.T), lmax+1)

        nbes = 40
        orbgen._update_transform_table(nbes, 2)
        self.assertEqual(len(orbgen.T), lmax+1)
        self.assertEqual(orbgen.T[0].shape[0], nbes)

        nbes = 50
        lmax = 7
        orbgen._update_transform_table(nbes, lmax)
        self.assertEqual(len(orbgen.T), lmax+1)
        self.assertEqual(orbgen.T[0].shape[0], nbes)

    
    def test_add_config(self):
        orbgen = Spillage()

        folder = '/home/zuxin/abacus-community/abacus_orbital_generation/tmp/Si-dimer-2.0/'

        mat = read_orb_mat(folder + 'orb_matrix_rcut6deriv0.dat')
        dmat = read_orb_mat(folder + 'orb_matrix_rcut6deriv1.dat')

        orbgen.add_config(mat, dmat)

        self.assertEqual(len(orbgen.config), 1)
        self.assertEqual(len(orbgen.T), max(mat['lmax'])+1)
        self.assertDictEqual(orbgen.config[0][0], mat)
        self.assertDictEqual(orbgen.config[0][1], dmat)

        orbgen.reset()

        mat = read_orb_mat(folder + 'orb_matrix_rcut7deriv0.dat')
        dmat = read_orb_mat(folder + 'orb_matrix_rcut7deriv1.dat')

        orbgen.add_config(mat, dmat)

        self.assertEqual(len(orbgen.config), 1)
        self.assertEqual(len(orbgen.T), max(mat['lmax'])+1)
        self.assertDictEqual(orbgen.config[0][0], mat)
        self.assertDictEqual(orbgen.config[0][1], dmat)


    def test_make_dual(self):
        nk = 3
        nbands = 5
        nao = 6

        X = np.random.randn(nk, nbands, nao)

        # make each slice of S orthogonal so that the case is easier to verify
        S = np.array([np.linalg.qr(np.random.randn(nao, nao))[0] for _ in range(nk)])

        X_dual = _make_dual(X, S)

        self.assertEqual(X_dual.shape, X.shape)
        for i in range(nk):
            self.assertTrue( np.allclose(X_dual[i], X[i] @ S[i].T) )


    def test_tab_frozen(self):
        orbgen = Spillage()

        folder = '/home/zuxin/abacus-community/abacus_orbital_generation/tmp/Si-dimer-2.0/'

        mat = read_orb_mat(folder + 'orb_matrix_rcut6deriv0.dat')
        dmat = read_orb_mat(folder + 'orb_matrix_rcut6deriv1.dat')

        orbgen.add_config(mat, dmat)
        orbgen._tab_frozen([[np.eye(2, mat['nbes']-1).tolist() for l in range(3)]])


    def test_gen_q2zeta(self):
        ntype = 3
        natom = [1, 2, 3]
        lmax = [2, 1, 0]
        nzeta = [[1, 1, 1], [2, 2], [3]]
        _, mu2comp = _index_map(ntype, natom, lmax, nzeta)

        nbes = 5
        rcut = 6.0

        # coef[itype][l][zeta] gives a list of coefficients that specifies an orbital
        # NOTE it's the coefficients of end-smoothed mixed spherical Bessel basis,
        # not the coefficients of the truncated spherical Bessel function.
        coef = [[np.random.randn(nzeta[itype][l], nbes-1).tolist() \
                for l in range(lmax[itype]+1)] for itype in range(ntype)]

        for mu, mat in enumerate(_gen_q2zeta(coef, mu2comp, nbes, rcut)):
            itype, iatom, l, zeta, m = mu2comp[mu]
            self.assertEqual(mat.shape, (nbes, nzeta[itype][l]))
            self.assertTrue( np.allclose(mat, \
                    jl_reduce(l, nbes, rcut) @ np.array(coef[itype][l]).T))


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

        
        res = orbgen._spillage2(0, coef, ibands, True)
        print('spill = ', orbgen._spillage(coef, ibands), res[0])
        print('dspill = ', dspill)
        print(res[1])
        
        res = orbgen._spillage2(0, coef, ibands, False)
        print(res)

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


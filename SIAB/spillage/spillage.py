from datparse import read_orb_mat, _assert_consistency
from radial import jl_reduce, JL_REDUCE
from listmanip import flatten, nest, nestpat

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
    Basis transformation matrix from truncated spherical Bessel functions
    to pseudo-atomic orbitals.

    This function constructs the transformation matrix from some jY
    ([spherical Bessel] x [spherical harmonics]) basis arranged in the
    lexicographic order of (itype, iatom, l, m, q) (q being the index
    for spherical Bessel functions) to some pseudo-atomic orbital basis
    arranged in the lexicographic order of (itype, iatom, l, m, zeta).
    The entire transformation matrix is block-diagonal, with each block
    corresponding to a specific q->zeta.


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
        for mu in lin2comp:
            itype, _, l, _, _ = lin2comp[mu]
            if l >= len(coef[itype]) or len(coef[itype][l]) == 0:
                # The generator should yield a zero matrix with the
                # appropriate size when no coefficient is provided.
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


def _list_normalize(coef):
    '''
    Returns the normalized list of coefficients.

    '''
    c = np.array(coef)
    return (c / np.linalg.norm(c)).tolist()


class Spillage:
    '''
    Generalized spillage function and its optimization.

    Attributes
    ----------
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


    def opt(self, coef_init, coef_frozen, iconf, ibands, options, nthreads=1):
        '''
        Spillage minimization w.r.t. end-smoothed mixed spherical Bessel coefficients.

        Parameters
        ----------
            coef_init : nested list
                Initial guess for the coefficients.
            coef_frozen : nested list
                Coefficients for the frozen orbitals.
            iconf : list of int or 'all'
                List of configuration indices to be included in the optimization.
                If 'all', all configurations are included.
            ibands : range/tuple or list of range/tuple
                Band indices to be included in the spillage calculation. If a range
                or tuple is given, the same indices are used for all configurations.
                If a list of range/tuple is given, each range/tuple will be applied
                to the configuration specified by iconf respectively.
            options : dict
                Options for the optimization.
            nthreads : int
                Number of threads for config-level parallellization.

        '''
        from multiprocessing.pool import ThreadPool
        pool = ThreadPool(nthreads)

        self._tab_frozen(coef_frozen)
        self._tab_deriv(coef_init)

        pat = nestpat(coef_init)

        iconf = range(len(self.config)) if iconf == 'all' else iconf
        nconf = len(iconf)

        ibands = [ibands] * nconf if not isinstance(ibands, list) else ibands
        assert len(ibands) == nconf

        # function to be minimized
        def f(c):
            s = lambda i: self._generalize_spillage(i, nest(c.tolist(), pat), ibands[i], True)
            spills, grads = zip(*pool.map(s, iconf))

            return (sum(spills) / nconf,
                    sum(np.array(flatten(g)) for g in grads) / nconf)

        c0 = np.array(flatten(coef_init))

        # Restricts the coefficients to [-1, 1] for better numerical stability
        # FIXME Is this necessary?
        bounds = [(-1.0, 1.0) for _ in c0]
        #bounds = None

        res = minimize(f, c0, jac=True, method='L-BFGS-B',
                       bounds=bounds, options=options)

        pool.close()

        coef_opt = nest(res.x.tolist(), pat)
        return [[[_list_normalize(coef_tlz) for coef_tlz in coef_tl]
                 for coef_tl in coef_t] for coef_t in coef_opt]


############################################################
#                           Test
############################################################
import unittest

class _TestSpillage(unittest.TestCase):

    def setUp(self):
        self.orbgen = Spillage()
        self.datadir = './testfiles/orb_matrix/'
        self.config = ['Si-dimer-2.0', 'Si-dimer-2.2',
                       'Si-trimer-2.1', 'Si-trimer-2.3',
                       ]

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


    def test_add_config(self):
        for iconf, config in enumerate(self.config):
            mat = read_orb_mat(self.datadir + config + '/orb_matrix.0.dat')
            dmat = read_orb_mat(self.datadir + config + '/orb_matrix.1.dat')
            self.orbgen.add_config(mat, dmat)

            self.assertEqual(len(self.orbgen.config), iconf+1)

            njy = len(mat['lin2comp']) * mat['nbes']
            dat = self.orbgen.config[iconf]
            self.assertEqual(dat['mo_mo'].shape, (2, mat['nk'], mat['nbands']))
            self.assertEqual(dat['mo_jy'].shape, (2, mat['nk'], mat['nbands'], njy))
            self.assertEqual(dat['jy_jy'].shape, (2, mat['nk'], njy, njy))

        self.assertTrue(self.orbgen.rcut in JL_REDUCE)


    def test_tab_frozen(self):
        for iconf, config in enumerate(self.config):
            mat = read_orb_mat(self.datadir + config + '/orb_matrix.0.dat')
            dmat = read_orb_mat(self.datadir + config + '/orb_matrix.1.dat')
            self.orbgen.add_config(mat, dmat)

        lmax = 2
        coef_frozen = [[np.eye(2, mat['nbes']-1).tolist() for l in range(lmax+1)]]
        self.orbgen._tab_frozen(coef_frozen)

        self.assertEqual(len(self.orbgen.mo_Pfrozen_jy), len(self.config))
        self.assertEqual(len(self.orbgen.spill_frozen), len(self.config))

        for iconf, config in enumerate(self.config):
            dat = self.orbgen.config[iconf]
            njy = len(dat['lin2comp']) * dat['nbes']
            self.assertEqual(self.orbgen.spill_frozen[iconf].shape,
                             (dat['nbands'],))
            self.assertEqual(self.orbgen.mo_Pfrozen_jy[iconf].shape,
                             (2, dat['nk'], dat['nbands'], njy))


    def test_tab_deriv(self):
        nbes_min = 1000
        for iconf, config in enumerate(self.config):
            ov = read_orb_mat(self.datadir + config + '/orb_matrix.0.dat')
            op = read_orb_mat(self.datadir + config + '/orb_matrix.1.dat')
            self.orbgen.add_config(ov, op)
            nbes_min = min(nbes_min, ov['nbes'])

        coef = [[np.eye(2, nbes_min-1).tolist(),
                 np.eye(2, nbes_min-1).tolist(),
                 np.eye(1, nbes_min-1).tolist()]]
        ncoef = len(flatten(coef))
        naos = [sum(len(coef_tl) * (2*l+1) for l, coef_tl in enumerate(coef_t))
                for coef_t in coef]
        self.orbgen._tab_deriv(coef)

        self.assertEqual(len(self.orbgen.dao_jy), len(self.config))
        self.assertEqual(len(self.orbgen.mo_Qfrozen_dao), len(self.config))

        for iconf, config in enumerate(self.config):
            dat = self.orbgen.config[iconf]
            njy = len(dat['lin2comp']) * dat['nbes']
            nao = sum(nao*natom for nao, natom in zip(naos, dat['natom']))
            self.assertEqual(self.orbgen.dao_jy[iconf].shape,
                             (2, ncoef, dat['nk'], nao, njy))


    def test_overlap_spillage(self):
        # generalized spillage with op=I should recover the overlap spillage

        ibands = range(5)
        nbes_min = 1000
        for iconf, config in enumerate(self.config):
            dat = read_orb_mat(self.datadir + config + '/orb_matrix.0.dat')
            self.orbgen.add_config(dat, dat)

            coef = [[np.random.randn(2, dat['nbes']-1).tolist(),
                     np.random.randn(2, dat['nbes']-1).tolist(),
                     np.random.randn(1, dat['nbes']-1).tolist()]]

            nbes_min = min(nbes_min, dat['nbes'])

            spill_ref = _overlap_spillage(dat, coef, ibands) 
            spill = self.orbgen._generalize_spillage(iconf, coef, ibands, False)
            self.assertAlmostEqual(spill, spill_ref, places=10)

        coef_frozen = [[np.random.randn(2, nbes_min-1).tolist(),
                        np.random.randn(1, nbes_min-1).tolist(),
                        np.random.randn(1, nbes_min-1).tolist()]]
        self.orbgen._tab_frozen(coef_frozen)

        for iconf, config in enumerate(self.config):
            dat = read_orb_mat(self.datadir + config + '/orb_matrix.0.dat')
            coef = [[np.random.randn(2, dat['nbes']-1).tolist(),
                     np.random.randn(2, dat['nbes']-1).tolist(),
                     np.random.randn(1, dat['nbes']-1).tolist()]]

            spill_ref = _overlap_spillage(dat, coef, ibands, coef_frozen) 
            spill = self.orbgen._generalize_spillage(iconf, coef, ibands, False)
            self.assertAlmostEqual(spill, spill_ref, places=10)


    def test_finite_difference(self):
        nbes_min = 1000
        for iconf, config in enumerate(self.config):
            ov = read_orb_mat(self.datadir + config + '/orb_matrix.0.dat')
            op = read_orb_mat(self.datadir + config + '/orb_matrix.1.dat')
            self.orbgen.add_config(ov, op)
            nbes_min = min(nbes_min, ov['nbes'])

        ibands = range(6)
        coef = [[np.random.randn(1, nbes_min-1).tolist(),
                 np.random.randn(2, nbes_min-1).tolist(),
                 np.random.randn(1, nbes_min-1).tolist()]]

        self.orbgen._tab_deriv(coef)

        for iconf, dat in enumerate(self.orbgen.config):

            dspill = self.orbgen._generalize_spillage(iconf, coef, ibands, True)[1]
            dspill = np.array(flatten(dspill))

            pat = nestpat(coef)
            sz = len(flatten(coef))

            dspill_fd = np.zeros(sz)
            dc = 1e-6
            for i in range(sz):
                coef_p = flatten(deepcopy(coef))
                coef_p[i] += dc
                coef_p = nest(coef_p, pat)
                spill_p = self.orbgen._generalize_spillage(iconf, coef_p, ibands, False)

                coef_m = flatten(deepcopy(coef))
                coef_m[i] -= dc
                coef_m = nest(coef_m, pat)
                spill_m = self.orbgen._generalize_spillage(iconf, coef_m, ibands, False)

                dspill_fd[i] = (spill_p - spill_m) / (2 * dc)

            self.assertTrue(np.allclose(dspill, dspill_fd, atol=1e-7))

        coef_frozen = [[np.random.randn(2, nbes_min-1).tolist(),
                        np.random.randn(1, nbes_min-1).tolist(),
                        np.random.randn(1, nbes_min-1).tolist()]]

        self.orbgen._tab_frozen(coef_frozen)
        self.orbgen._tab_deriv(coef)

        for iconf, dat in enumerate(self.orbgen.config):
            dspill = self.orbgen._generalize_spillage(iconf, coef, ibands, True)[1]
            dspill = np.array(flatten(dspill))

            pat = nestpat(coef)
            sz = len(flatten(coef))

            dspill_fd = np.zeros(sz)
            dc = 1e-6
            for i in range(sz):
                coef_p = flatten(deepcopy(coef))
                coef_p[i] += dc
                coef_p = nest(coef_p, pat)
                spill_p = self.orbgen._generalize_spillage(iconf, coef_p, ibands, False)

                coef_m = flatten(deepcopy(coef))
                coef_m[i] -= dc
                coef_m = nest(coef_m, pat)
                spill_m = self.orbgen._generalize_spillage(iconf, coef_m, ibands, False)

                dspill_fd[i] = (spill_p - spill_m) / (2 * dc)

            self.assertTrue(np.allclose(dspill, dspill_fd, atol=1e-7))


    def test_opt(self):
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
        options = {'maxiter': 100, 'disp': True, 'maxcor': 20}

        coef_opt = self.orbgen.opt(coef0, coef_frozen, 'all', ibands, options, nthreads)

        ibands = [range(4), range(4), range(6), range(6)]
        coef_opt = self.orbgen.opt(coef0, coef_frozen, [0,1,2,3], ibands, options, nthreads)

        pass


if __name__ == '__main__':
    unittest.main()


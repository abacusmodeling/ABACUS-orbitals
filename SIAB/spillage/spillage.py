from SIAB.spillage.datparse import read_orb_mat, _assert_consistency
from SIAB.spillage.radial import jl_reduce, jl_raw_norm, coeff_normalized2raw
from SIAB.spillage.listmanip import flatten, nest, nestpat
from SIAB.spillage.jlzeros import JLZEROS

import numpy as np
from scipy.optimize import minimize, basinhopping

from copy import deepcopy

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


def _nbes(l, rcut, ecut):
    '''
    Calculates the number of normalized truncated spherical Bessel functions
    whose kinetic energy is below the energy cutoff.

    Note
    ----
    1. The kinetic energy of a normalized truncated spherical Bessel basis
       j_l(k*r) * Y_{lm}(r) is k^2
    
    2. The wavenumbers of truncated spherical Bessel functions are chosen such
       that the function is zero at rcut, i.e., JLZEROS/rcut

    '''
    # make sure the tabulated JLZEROS is sufficient
    assert (JLZEROS[l][-1]/rcut)**2 > ecut
    return sum((JLZEROS[l]/rcut)**2 < ecut)


def initgen(nzeta, ov, reduced=False):
    '''
    Generate an initial guess of the spherical Bessel coefficients from
    the single-atom overlap data (in raw basis) for the spillage optimization.

    '''
    assert ov['ntype'] == 1 and ov['natom'][0] == 1 and ov['ecutwfc'] <= ov['ecutjlq']

    lmax = len(nzeta) - 1
    assert lmax <= ov['lmax'][0]

    rcut = ov['rcut']
    ecut = ov['ecutwfc']
    nbes_ecut = [_nbes(l, rcut, ecut) for l in range(lmax+1)]
    if reduced:
        nbes_ecut = [n - 1 for n in nbes_ecut]

    mo_jy = ov['mo_jy']
    nbes = ov['nbes']
    rcut = ov['rcut']
    if reduced:
        coef = [[jl_reduce(l, nbes, rcut).T.tolist() for l in range(ov['lmax'][0] + 1)]]
        mo_jy = mo_jy @ _jy2ao(coef, ov['lin2comp'], nbes, rcut)
        nbes -= 1
    else: # normalized
        uvec = lambda v, k, n: [v if i == k else 0 for i in range(n)]
        coef = [[[uvec(1. / jl_raw_norm(l, q, rcut), q, nbes)
                  for q in range(nbes)]
                 for l in range(ov['lmax'][0] + 1)]]
        mo_jy = mo_jy @ _jy2ao(coef, ov['lin2comp'], nbes, rcut)

    # reshaped to [nk, nbands, nao, nbes]
    Y = mo_jy.reshape(ov['nk'], ov['nbands'], -1, nbes)

    assert all(n > 0 and n <= nbes for n in nbes_ecut)

    coef = []
    for l in range(lmax+1):
        idx_start = ov['comp2lin'][(0, 0, l, 0, 0)]
        Yl = Y[:, :, idx_start:idx_start+2*l+1, :].reshape(ov['nk'], -1, nbes)
        Yl = Yl[:,:,:nbes_ecut[l]]
        YdaggerY = (ov['wk'].reshape(-1,1,1) * (Yl.transpose((0, 2, 1)).conj() @ Yl)) \
                   .sum(0).real
        val, vec = np.linalg.eigh(YdaggerY)

        # eigenvectors corresponding to the largest nzeta eigenvalues
        coef.append(vec[:,-nzeta[l]:][:,::-1].T.tolist())

    #return coef
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


    def add_config(self, ov, op):
        '''
        '''
        # The overlap and operator data must be consistent except
        # for their matrix data (mo_mo, mo_jy and jy_jy).
        _assert_consistency(ov, op)

        # The dict to append to config is a new one consisting of
        # the common part of ovlp & op data plus their stacked
        # matrix data.
        dat = deepcopy(ov)

        ntype = ov['ntype']
        lmax = ov['lmax']
        nbes = ov['nbes']
        rcut = ov['rcut']

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

        C = _jy2ao(coef, ov['lin2comp'], nbes, rcut)

        dat['mo_mo'] = np.array([ov['mo_mo'], op['mo_mo']])
        dat['mo_jy'] = np.array([ov['mo_jy'] @ C, op['mo_jy'] @ C])
        dat['jy_jy'] = np.array([C.T @ ov['jy_jy'] @ C, C.T @ op['jy_jy'] @ C])

        self.config.append(dat)

        # NOTE currently a dataset merely contains one atom type
        # and one rcut. This may change in the future.
        if self.rcut is None:
            self.rcut = ov['rcut']
        else:
            assert self.rcut == ov['rcut']


    def _tab_frozen(self, coef_frozen):
        '''
        Tabulates for each configuration the band-wise spillage contribution
        from frozen orbitals and

                            <mo|P_frozen   |jy>
                            <mo|P_frozen op|jy>

        where P_frozen is the projection operator onto the frozen subspace:

                        P_frozen = |frozen_dual><frozen|

        '''
        if coef_frozen is None:
            self.spill_frozen = None
            self.mo_Pfrozen_jy = None
            return

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

        #minimizer_kwargs = {"method": "L-BFGS-B", "jac": True, "bounds": bounds}
        #res = basinhopping(f, c0, minimizer_kwargs=minimizer_kwargs, niter=20, disp=True)

        pool.close()

        coef_opt = nest(res.x.tolist(), pat)
        return [[np.linalg.qr(np.array(coef_tl).T)[0].T.tolist()
                 for coef_tl in coef_t] for coef_t in coef_opt]


############################################################
#                           Test
############################################################
import unittest

from SIAB.spillage.radial import build_reduced, build_raw, coeff_reduced2raw
from SIAB.spillage.plot import plot_chi

import matplotlib.pyplot as plt


class _TestSpillage(unittest.TestCase):

    def setUp(self):
        self.orbgen_reduced = Spillage(True)
        self.orbgen_raw = Spillage(False)

        self.datadir = './testfiles/Si/'
        self.config = ['Si-dimer-1.8', 'Si-dimer-2.8', 'Si-dimer-3.8',
                       'Si-trimer-1.7', 'Si-trimer-2.7',
                       ]


    def read_config(self):
        self.ov = [read_orb_mat(self.datadir + config + '/orb_matrix.0.dat')
                   for config in self.config]
        self.op = [read_orb_mat(self.datadir + config + '/orb_matrix.1.dat')
                   for config in self.config]


    def add_config(self):
        for iconf, _ in enumerate(self.config):
            self.orbgen_reduced.add_config(self.ov[iconf], self.op[iconf])
            self.orbgen_raw.add_config(self.ov[iconf], self.op[iconf])


    def coefgen(self, nzeta, nbes):
        '''
        Generates some random coefficients for unit tests.

        Parameters
        ----------
            nzeta : nested list of int
                nzeta[itype][l] gives the number of zeta.
            nbes : int
                Number of spherical Bessel basis functions for each zeta.

        '''
        return [[np.random.randn(nzeta_tl, nbes).tolist()
                 for l, nzeta_tl in enumerate(nzeta_t)]
                for it, nzeta_t in enumerate(nzeta)]


    def test_initgen(self):
        '''
        checks whether initgen generates the correct number of coefficients

        '''
        reduced = False

        ov = read_orb_mat('./testfiles/Si/Si-monomer/orb_matrix.0.dat')
        nzeta = [2, 2, 1]

        coef = initgen(nzeta, ov, reduced)

        self.assertEqual(len(coef), len(nzeta))
        self.assertEqual([len(coef[l]) for l in range(len(nzeta))], nzeta)

        return

        rcut = ov['rcut']
        dr = 0.01
        r = np.linspace(0, rcut, int(rcut/dr)+1)

        if reduced:
            coef_raw = coeff_reduced2raw(coef, rcut)
            chi = build_reduced(coef, rcut, r, True)
        else:
            chi = build_raw(coeff_normalized2raw(coef, rcut), rcut, r, True)

        plot_chi(chi, r)
        plt.show()


    def test_mrdiv(self):
        '''
        checks mrdiv with orthogonal matrices

        '''
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

        coef = self.coefgen(nzeta, nbes)
        M = _jy2ao(coef, lin2comp, nbes, rcut)

        icol = 0
        for mu, (itype, iatom, l, _, m) in lin2comp.items():
            nzeta = len(coef[itype][l])
            self.assertTrue(np.allclose(
                M[mu*nbes:(mu+1)*nbes, icol:icol+nzeta],
                np.array(coef[itype][l]).T
            ))
            icol += nzeta


    def test_add_config(self):
        '''
        checks if add_config loads & transform data correctly

        '''
        self.read_config()
        self.add_config()

        for iconf, config in enumerate(self.config):
            for orbgen in [self.orbgen_raw, self.orbgen_reduced]:
                ov = self.ov[iconf]
                op = self.op[iconf]

                dat = orbgen.config[iconf]
                nbes = dat['nbes']
                njy = len(dat['lin2comp']) * nbes

                self.assertEqual(dat['mo_mo'].shape, (2, ov['nk'], ov['nbands']))
                self.assertEqual(dat['mo_jy'].shape, (2, ov['nk'], ov['nbands'], njy))
                self.assertEqual(dat['jy_jy'].shape, (2, ov['nk'], njy, njy))

                # add_config not only load data, but also performs normalization/reduction
                # here we check them by looking at the overlap matrix on the first atom
                nao_0 = dat['comp2lin'][(0, 1, 0, 0, 0)]
                S = dat['jy_jy'][0, 0][:nao_0, :nao_0]
                self.assertLess(np.linalg.norm(S - np.eye(nao_0), np.inf), 1e-6)

            self.assertEqual(self.orbgen_raw.config[iconf]['nbes'],
                             self.orbgen_reduced.config[iconf]['nbes'] + 1)


    def test_tab_frozen(self):
        '''
        checks if data tabulated by tab_frozen have the correct shape

        '''
        self.read_config()
        self.add_config()

        nbes = min(dat['nbes'] for dat in self.orbgen_reduced.config)
        coef_frozen = self.coefgen([[2, 1, 0]], nbes)

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
        '''
        checks if data tabulated by tab_deriv have the correct shape

        '''
        self.read_config()
        self.add_config()

        nbes = min(dat['nbes'] for dat in self.orbgen_reduced.config)
        coef = self.coefgen([[2, 1, 0]], nbes)

        # number of coefficients
        ncoef = len(flatten(coef))

        # number of spherical Bessel basis related to coef
        njy_ao = [sum(len(coef_tl) * (2*l+1) for l, coef_tl in enumerate(coef_t))
                    for coef_t in coef]

        for orbgen in [self.orbgen_raw, self.orbgen_reduced]:
            orbgen._tab_deriv(coef)

            self.assertEqual(len(orbgen.dao_jy), len(self.config))
            self.assertEqual(len(orbgen.mo_Qfrozen_dao), len(self.config))

            for iconf, config in enumerate(self.config):
                dat = orbgen.config[iconf]
                n_dao = np.dot(njy_ao, dat['natom'])
                njy = len(dat['lin2comp']) * dat['nbes']
                self.assertEqual(orbgen.dao_jy[iconf].shape,
                                 (2, ncoef, dat['nk'], n_dao, njy))


    def test_overlap_spillage(self):
        '''
        verifies that the generalized spillage with op=I recovers the overlap spillage

        '''
        self.read_config()

        # op=I implies op=ov
        for iconf, _ in enumerate(self.config):
            self.orbgen_reduced.add_config(self.ov[iconf], self.ov[iconf])
            self.orbgen_raw.add_config(self.ov[iconf], self.ov[iconf])

        ibands = range(5)
        nbes = min(dat['nbes'] for dat in self.orbgen_reduced.config)

        coef = self.coefgen([[2, 2, 1]], nbes)
        coef_frozen_list = [
                None,
                self.coefgen([[1, 1]], nbes),
                self.coefgen([[2, 1, 0]], nbes),
                self.coefgen([[0, 1, 1]], nbes),
                ]

        for orbgen in [self.orbgen_raw, self.orbgen_reduced]:
            for coef_frozen in coef_frozen_list:
                orbgen._tab_frozen(coef_frozen)
                for iconf, config in enumerate(self.config):
                    ov = deepcopy(self.ov[iconf])
                    ov['nbes'] = orbgen.config[iconf]['nbes']
                    ov['mo_jy'] = orbgen.config[iconf]['mo_jy'][0]
                    ov['jy_jy'] = orbgen.config[iconf]['jy_jy'][0]

                    spill_ref = _overlap_spillage(ov, coef, ibands, coef_frozen) 
                    spill = orbgen._generalize_spillage(iconf, coef, ibands, False)
                    self.assertAlmostEqual(spill, spill_ref, places=10)


    def test_finite_difference(self):
        '''
        checks the gradient of the generalized spillage with finite difference

        '''
        self.read_config()
        self.add_config()

        ibands = range(6)
        nbes = min(dat['nbes'] for dat in self.orbgen_reduced.config)

        coef = self.coefgen([[2, 1, 1]], nbes)
        coef_frozen_list = [None, self.coefgen([[1, 1]], nbes)]

        for orbgen in [self.orbgen_raw, self.orbgen_reduced]:
            for coef_frozen in coef_frozen_list:
                orbgen._tab_frozen(coef_frozen)
                orbgen._tab_deriv(coef)

                for iconf, _ in enumerate(orbgen.config):
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


    def test_opt(self):
        from listmanip import merge

        datadir = './testfiles/Si/'
        configs = ['Si-dimer-1.8',
                   'Si-dimer-2.8',
                   'Si-dimer-3.8',
                   'Si-trimer-1.7',
                   'Si-trimer-2.7']

        reduced = True
        orbgen = self.orbgen_reduced if reduced else self.orbgen_raw

        for iconf, config in enumerate(configs):
            ov = read_orb_mat(datadir + config + '/orb_matrix.0.dat')
            op = read_orb_mat(datadir + config + '/orb_matrix.1.dat')
            orbgen.add_config(ov, op)

        nthreads = 2
        options = {'ftol': 0, 'gtol': 1e-6, 'maxiter': 2000, 'disp': False, 'maxcor': 20}

        # initial guess
        ov = read_orb_mat('./testfiles/Si/Si-monomer/orb_matrix.0.dat')
        coef_init = initgen([3, 3, 2], ov, reduced)

        ibands = range(4)
        iconfs = [0, 1, 2]
        # coef_lvl1_init: [t][l][z][q]
        coef_lvl1_init = [[[coef_init[0][0]],
                           [coef_init[1][0]]]]
        coef_lvl1 = orbgen.opt(coef_lvl1_init, None, iconfs, ibands, options, nthreads)
        coef_tot = coef_lvl1

        ibands = range(8)
        iconfs = [0, 1, 2]
        coef_lvl2_init = [[[coef_init[0][1]],
                           [coef_init[1][1]],
                           [coef_init[2][0]]]]
        coef_lvl2 = orbgen.opt(coef_lvl2_init, coef_lvl1, iconfs, ibands, options, nthreads)
        coef_tot = merge(coef_tot, coef_lvl2, 2)

        ibands = range(12)
        iconfs = [3, 4]
        coef_lvl3_init = [[[coef_init[0][2]],
                           [coef_init[1][2]],
                           [coef_init[2][1]]]]
        coef_lvl3 = orbgen.opt(coef_lvl3_init, coef_tot, iconfs, ibands, options, nthreads)
        coef_tot = merge(coef_tot, coef_lvl3, 2)

        return

        rcut = ov['rcut']
        dr = 0.01
        r = np.linspace(0, rcut, int(rcut/dr)+1)

        if reduced:
            chi = build_reduced(coef_tot[0], rcut, r, True)
        else:
            coeff_raw = coeff_normalized2raw(coef_tot, rcut)
            chi = build_raw(coeff_raw[0], rcut, r, 0.0, True, True)

        plot_chi(chi, r)
        plt.show()


if __name__ == '__main__':
    unittest.main()


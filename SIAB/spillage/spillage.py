'''
Orbital generation by spillage optimization.

References
----------
[1] Chen, M., Guo, G. C., & He, L. (2010). Systematically improvable optimized
atomic basis sets for ab initio calculations.
Journal of Physics: Condensed Matter, 22(44), 445501.

[2] Lin, P., Ren, X., & He, L. (2021). Strategy for constructing compact
numerical atomic orbital basis sets by incorporating the gradients of
reference wavefunctions.
Physical Review B, 103(23), 235131.

'''
from SIAB.spillage.radial import _nbes, jl_reduce, jl_raw_norm,\
        coeff_normalized2raw, coeff_reduced2raw
from SIAB.spillage.listmanip import flatten, nest, nestpat
from SIAB.spillage.jlzeros import JLZEROS
from SIAB.spillage.index import _lin2comp, perm_zeta_m, _nao
from SIAB.spillage.linalg_helper import mrdiv, rfrob
from SIAB.spillage.basistrans import jy2ao
from SIAB.spillage.datparse import read_orb_mat, \
        read_wfc_lcao_txt, read_triu, read_running_scf_log

import numpy as np
from scipy.optimize import minimize, basinhopping
from copy import deepcopy


def _jy_data_extract(outdir):
    '''
    Extracts the data for spillage optimization with spherical-wave
    reference states from an OUT.{suffix} directory.

    This function looks for certain data from the following files:

        running_scf.log: natom, nzeta, wk, nspin
        data-*-S: overlap matrices
        data-*-T: kinetic energy matrices
        WFC_NAO-*.txt: LCAO wavefunction coefficients

    The extracted data is packed to a dict with the following key-value pairs:

        natom : list of int
            Number of atoms for each atom type.
        nzeta : list of list of int
            Number of zeta for each l of each atom type.
            nzeta[itype][l] -> int.
        wk : ndarray, shape (nk,)
            k-point weights. For nspin = 2, the weights are replicated
            for spin-down (so the first and second halves are the same).
        S : ndarray, shape (nk, nao, nao)
            Basis overlap matrices.
        T : ndarray, shape (nk, nao, nao)
            Kinetic energy matrices.
        C : ndarray, shape (nk, nao, nbands)
            LCAO wavefunction coefficients.

    '''
    info = read_running_scf_log(outdir + '/running_scf.log')
    nspin, wk, natom, nzeta = [info[key] for key in
                               ['nspin', 'wk', 'natom', 'nzeta']]

    nk = len(wk)
    S = [read_triu(f'{outdir}/data-{ik}-S') for ik in range(nk)]
    T = [read_triu(f'{outdir}/data-{ik}-T') for ik in range(nk)]

    wfc_suffix = 'GAMMA' if nk == 1 else 'K'
    C = [read_wfc_lcao_txt(f'{outdir}/WFC_NAO_{wfc_suffix}{ik+1}.txt')[0]
         for ik in range(nspin * nk)]

    if nspin == 2: # replicate for spin-down
        S = [*S, *S]
        T = [*T, *T]
        wk = [*wk, *wk]

    return {'natom': natom, 'nzeta': nzeta, 'wk': wk,
            'S': np.array(S), 'T': np.array(T), 'C': np.array(C)}


def _initgen_core(nzeta, nbes_data, ref_jy, wk, nbes_gen, diagosis):
    '''
    Computational core of initgen_jy and initgen_pw.

    Parameters
    ----------
        nzeta : list of int
            Target number of zeta for each l.
        nbes_data : list of int
            Number of spherical Bessel components for each l of the
            single-atom overlap data (ref_jy).
        ref_jy : ndarray, shape (nk, nbands, njy)
            Overlap between single-atom reference states and spherical
            waves (jy). The order of jy must be lexicographic in terms
            of (l, m, q) where l & m are the angular momentum indices
            and q is the radial index.
        wk : ndarray, shape (nk,)
            Weights of k points.
        nbes_gen : list of int
            Number of spherical Bessel components for each l to generate.
        diagosis : bool
            If true, print for each l the largest nzeta[l] eigenvalues of

                    sum_k wk * Y[l,k].T.conj() @ Y[l,k]

            where Y[l,k] is the subblock of <ref(k)|jy(k)> with spherical
            waves (jy) of angular momentum l and reorganized to a matrix of
            shape (nbands*(2*l+1), nbes_data[l]).
            (The data is assume for a single atom, so the number of spherical
            waves per l is simply (2l+1) * nbes_data[l].)

    Returns
    -------
        A nested list. coef[l][zeta][q] -> float.

    '''
    # number of spherical waves per l
    njy_l = [(2*l+1) * nbes_l for l, nbes_l in enumerate(nbes_data)]

    # delimiting indices for each l
    delim = [0] + np.cumsum(njy_l).tolist()

    nk, nbands, _ = ref_jy.shape

    coef = []
    for l, nzeta_l in enumerate(nzeta):
        if nzeta_l == 0:
            coef.append([])
            continue

        Y = ref_jy[:,:,range(delim[l], delim[l+1])] \
            .reshape(nk, nbands*(2*l+1), nbes_data[l]) \
            [:,:,:nbes_gen[l]]

        YdaggerY = ((Y.swapaxes(-2, -1).conj() @ Y)
                    * wk.reshape(-1, 1, 1)).sum(0).real

        val, vec = np.linalg.eigh(YdaggerY)

        # eigenvectors corresponding to the largest nzeta eigenvalues
        coef.append(vec[:,-nzeta_l:][:,::-1].T.tolist())

        if diagosis:
            print( "initgen: <jy|ref><ref|jy> eigval diagnosis:")
            print(f"         l = {l}: {val[-nzeta_l:][::-1]}")

    return coef


def initgen_jy(outdir, nzeta, ibands='all', nbes_gen=None, diagosis=False):
    '''
    Generates an initial guess of the spherical Bessel coefficients from
    single-atom overlap data.

    Parameters
    ----------
        outdir : str
            Path to the OUT.{suffix} directory generated by ABACUS.
        nzeta : list of int
            Target number of zeta for each l.
        ibands : list of int or range or 'all'
            Band indices to be included.
        nbes_gen : int or list of int or None
            Number of spherical Bessel components for each l to generate.
            This allows the user to generate a set of coefficients with
            smaller cutoff energy than the input data. For example, suppose
            the data has [21, 20, 20] spherical waves for l = 0, 1, 2,
            a user may specify a smaller nbes_gen like [16, 15, 15].
            If an int, the same number of spherical Bessel components
            is assumed for each l;
            If a list, nbes_gen[l] is the number corresponding to l;
            If None, nbes_gen is assumed to be the same as what is given
            in the single-atom data.
        diagosis : bool
            See _initgen_jy_core for details.

    Returns
    -------
        A nested list. coef[l][zeta][q] -> float.

    Note
    ----
    This function does not discriminate normalized or reduced spherical wave
    radial functions. The coefficients are generated in terms of whatever
    used to generate the reference state.

    The resulting coefficients are always "normalized" in the sense that
    norm(coef[l][zeta]) = 1 for any l and zeta. Since ABACUS always
    normalizes the numerical atomic orbitals in LCAO calculations,
    this should yield normalized orbitals in practice.

    '''
    dat = _jy_data_extract(outdir)
    natom, nbes_data, wk, S, T, C = \
            [dat[key] for key in ['natom', 'nzeta', 'wk', 'S', 'T', 'C']]

    ref_jy = C.swapaxes(-2, -1).conj() @ S
    p = perm_zeta_m(_lin2comp(natom, nzeta=nbes_data))
    ref_jy = ref_jy[:,:,p]

    # remove the 'itype' layer
    nbes_data = nbes_data[0]

    if nbes_gen is None:
        nbes_gen = nbes_data
    elif isinstance(nbes_gen, int):
        nbes_gen = [nbes_gen] * len(nzeta)
    else: # must be a list of int
        assert all(isinstance(n, int) for n in nbes_gen)

    if ibands == 'all':
        ibands = range(ref_jy.shape[1])

    # some sanity checks
    # 1. the data is indeed from a single atom
    assert natom == [1]
    assert(sum((2*l+1) * nbes_l for l, nbes_l in enumerate(nbes_data))
           == ref_jy.shape[-1])

    # 2. nzeta must not exceed nbes_gen
    assert len(nzeta) <= len(nbes_gen)
    assert all(nzeta[l] <= nbes_gen[l] for l in range(len(nzeta)))

    # 3. nbes_gen must not exceed nbes_data
    assert len(nbes_gen) <= len(nbes_data)
    assert all(nbes_gen[l] <= nbes_data[l] for l in range(len(nbes_gen)))

    # 4. band indices must be within the range
    assert all(0 <= ib < ref_jy.shape[1] for ib in ibands)

    return _initgen_core(nzeta, nbes_data, ref_jy[:,ibands,:],
                         wk, nbes_gen, diagosis)


def initgen_pw(orb_mat, nzeta, ibands='all', nbes_gen=None, diagosis=False):
    '''
    Generates an initial guess of the spherical Bessel coefficients from
    single-atom overlap data.

    Parameters
    ----------
        orb_mat: str
            Path to the orb_matrix data file generated by PW calculations.
            The file should contain overlap, not operator matrix elements.
        nzeta : list of int
            Target number of zeta for each l.
        ibands : list of int or range or 'all'
            Band indices to be included.
        nbes_gen : int or list of int or None
            Number of spherical Bessel components for each l to generate.
            This allows the user to generate a set of coefficients with
            smaller cutoff energy than the input data. For example, suppose
            the data has [21, 20, 20] spherical waves for l = 0, 1, 2,
            a user may specify a smaller nbes_gen like [16, 15, 15].
            If an int, the same number of spherical Bessel components
            is assumed for each l;
            If a list, nbes_gen[l] is the number corresponding to l;
            If None, nbes_gen is assumed to be the same as what is given
            in the single-atom data.
        diagosis : bool
            See _initgen_jy_core for details.

    Returns
    -------
        A nested list. coef[l][zeta][q] -> float.

    Note
    ----
    The coefficients are generated in terms of reduced spherical waves
    radial functions and are normalized.

    '''
    ov = read_orb_mat(orb_mat)

    ntype, natom, lmax, nbes, rcut, wk = \
        [ov[key] for key in ['ntype', 'natom', 'lmax', 'nbes', 'rcut', 'wk']]

    # The output of PW calculation is based on raw truncated spherical
    # waves, while the optimization will be performed w.r.t. the reduced
    # basis, so a basis transformation of ref_jy & jy_jy is needed.
    coef = [[jl_reduce(l, nbes, rcut).T.tolist()
             for l in range(lmax[itype]+1)]
            for itype in range(ntype)]

    # number of raw spherical wave radial functions per l
    # nbes[itype][l] -> int
    nbes_raw = [[nbes] * (lmax[itype] + 1) for itype in range(ntype)]

    ref_jy = ov['ref_jy'] @ jy2ao(coef, natom, nbes_raw)


    # number of reduced spherical wave radial functions per l
    nbes_data = [nbes - 1] * (lmax[0] + 1)

    if nbes_gen is None:
        nbes_gen = nbes_data
    elif isinstance(nbes_gen, int):
        nbes_gen = [nbes_gen] * len(nzeta)
    else: # must be a list of int
        assert all(isinstance(n, int) for n in nbes_gen)

    if ibands == 'all':
        ibands = range(ref_jy.shape[1])

    # some sanity checks
    # 1. the data is indeed from a single atom
    assert natom == [1]
    assert(sum((2*l+1) * nbes_l for l, nbes_l in enumerate(nbes_data))
           == ref_jy.shape[-1])

    # 2. nzeta must not exceed nbes_gen
    assert len(nzeta) <= len(nbes_gen)
    assert all(nzeta[l] <= nbes_gen[l] for l in range(len(nzeta)))

    # 3. nbes_gen must not exceed nbes_data
    assert len(nbes_gen) <= len(nbes_data)
    assert all(nbes_gen[l] <= nbes_data[l] for l in range(len(nbes_gen)))

    # 4. band indices must be within the range
    assert all(0 <= ib < ref_jy.shape[1] for ib in ibands)

    return _initgen_core(nzeta, nbes_data, ref_jy[:,ibands,:],
                         wk, nbes_gen, diagosis)


def _overlap_spillage(natom, nbes, jy_jy, ref_jy, ref_ref, wk,
                      coef, ibands, coef_frozen=None):
    '''
    Standard spillage function (overlap spillage).

    The spillage function is defined as

    S = (1/N) sum_{ik} w[k] <ref(i,k)|Q_frozen(k) Q(k) Q_frozen(k)|ref(i,k)>

    where N is the number of involved bands, w[k] is the k-point weight,
    Q (Q_frozen) is the projection operator onto the complement of the AO
    subspace specified by coef (coef_frozen).

    Parameters
    ----------
        natom : list of int
            Number of atoms for each atom type.
        nbes : list of int / list of list of int
            Number of spherical wave radial functions.
            If a list, nbes[l] specifies the number for angular momentum l,
            which is assumed to be the same for different atomic types.
            If a nested list, nbes[itype][l] specifies the number for angular
            momentum l of atomic type `itype`.
        jy_jy : ndarray, shape (nk, njy, njy)
            Overlap between spherical waves (jy).
        ref_jy : ndarray, shape (nk, nbands, njy)
            Overlap between reference states and spherical waves.
        ref_ref : ndarray, shape (nk, nbands)
            Overlap between reference states (diagonal terms only).
        wk : ndarray, shape (nk,)
            k-point weights.
        coef : nested list
            The coefficients of pseudo-atomic orbital basis orbitals
            in terms of the spherical wave basis. coef[itype][l][zeta]
            gives a list of spherical wave coefficients that specifies
            an orbital.
            Note that len(coef[itype][l][zeta]) must not be larger than
            nbes[itype][l]; coef[itype][l][zeta] will be padded with zeros
            if len(coef[itype][l][zeta]) < nbes[itype][l].
        ibands : list of int or range or 'all'
            Band indices to be included in the spillage calculation.
            If 'all', all bands are included.
        coef_frozen : nested list (optional)
            The coefficients of frozen pseudo-atomic orbitals in terms of
            the spherical wave basis. The format is the same as coef.


    Note
    ----
    This function is not supposed to be used in the optimization.
    As a special case of the generalized spillage (op = I), it serves
    as a cross-check for the implementation of the generalized spillage.

    '''
    if ibands == 'all':
        ibands = range(ref_ref.shape[1])

    spill = (wk @ ref_ref[:,ibands]).real.sum()

    ref_jy = ref_jy[:,ibands,:]
    _jy2ao = jy2ao(coef, natom, nbes)
    V = ref_jy @ _jy2ao
    W = _jy2ao.T @ jy_jy @ _jy2ao

    if coef_frozen is not None:
        jy2frozen = jy2ao(coef_frozen, natom, nbes)
        X = ref_jy @ jy2frozen
        S = jy2frozen.T @ jy_jy @ jy2frozen
        X_dual = mrdiv(X, S)

        V -= X_dual @ jy2frozen.T @ jy_jy @ _jy2ao
        spill -= wk @ rfrob(X_dual, X)

    spill -= wk @ rfrob(mrdiv(V, W), V)

    return spill / len(ibands)


class Spillage:
    '''
    Generalized spillage function and its optimization.

    Attributes
    ----------
        config : list of dict
            Each dict contains the data from a geometric configuration,
            including the following key-value pairs:

            natom : list of int
                Number of atoms for each atom type.
            nbes : list of int / list of list of int
                Number of spherical wave radial functions.
                If a list, nbes[l] specifies the number for angular momentum
                l, which is assumed to be the same for all atomic types.
                If a nested list, nbes[itype][l] specifies the number for
                angular momentum l of atomic type `itype`.
            jy_jy   : ndarray, shape (2, nk, njy, njy)
            ref_jy  : ndarray, shape (2, nk, nbands, njy)
            ref_ref : ndarray, shape (2, nk, nbands) (diagonal only!)
                Matrix elements between spherical waves (jy) or reference
                states. The overlap and operator matrix elements are stacked
                such that, e.g., jy_jy[0] refers to <jy|jy> and jy_jy[1]
                refers to <jy|op|jy>. The order of jy must be lexicographic
                in terms of (itype, iatom, l, mm, q).
            wk : ndarray, shape (nk,)
                k-point weights.

        spill_frozen : list of array of shape (nbands,)
            Band-wise spillage contribution from frozen orbitals. The shapes
            of arrays may vary among configurations due to different numbers
            of bands.
            spill_frozen[iconf][iband] -> float.
        ref_Pfrozen_jy : list of ndarray of shape (2, nk, nbands, njy)
            Stacked <ref|P_frozen|jy> and <ref|P_frozen op|jy> for each
            configuration, where P_frozen is the projection operator onto
            the frozen subspace.
            ref_Pfrozen_jy[iconf][0] and ref_Pfrozen_jy[iconf][1] correspond
            to <ref|P_frozen|jy> and <ref|P_frozen op|jy> respectively. Each
            of them has a shape of (nk, nbands, njy).
        ref_Qfrozen_dao : list of ndarray
            Derivatives of <ref|Q_frozen|ao> and <ref|Q_frozen op|ao> w.r.t.
            the spherical wave radial coefficients (Q_frozen is the projection
            operator onto the complement of the frozen subspace).
        dao_jy : list of ndarray
            The derivatives of <ao|jy> and <ao|op|jy> w.r.t. the coefficients
            for each configuration.

    '''
    def __init__(self):
        self.reset()


    def reset(self):
        self.config = []
        self.spill_frozen = None
        self.ref_Pfrozen_jy = None
        self.ref_Qfrozen_dao = None
        self.dao_jy = None


    def _tab_frozen(self, coef_frozen):
        '''
        Tabulates for each configuration the band-wise spillage contribution
        from frozen orbitals as well as

                            <ref|P_frozen   |jy>
                            <ref|P_frozen op|jy>

        where P_frozen is the projection operator onto the frozen subspace:

                        P_frozen = |frozen_dual><frozen|

        '''
        self.spill_frozen = [None] * len(self.config)
        self.ref_Pfrozen_jy = [None] * len(self.config)

        if coef_frozen is None:
            return

        for iconf, dat in enumerate(self.config):
            jy2frozen = jy2ao(coef_frozen, dat['natom'], dat['nbes'])

            frozen_frozen = jy2frozen.T @ dat['jy_jy'] @ jy2frozen
            ref_frozen = dat['ref_jy'] @ jy2frozen

            # no need to compute <ref|op|frozen_dual>
            ref_frozen_dual = mrdiv(ref_frozen[0], frozen_frozen[0])

            self.ref_Pfrozen_jy[iconf] = \
                    ref_frozen_dual @ jy2frozen.T @ dat['jy_jy']

            # spill_frozen before weighted sum over k
            tmp = rfrob(ref_frozen_dual @ frozen_frozen[1],
                        ref_frozen_dual, True) \
                    - 2.0 * rfrob(ref_frozen_dual, ref_frozen[1], True)

            self.spill_frozen[iconf] = dat['wk'] @ tmp


    def _tab_deriv(self, coef):
        '''
        Given coef which specifies a set of pseudo-atomic orbitals from
        spherical waves, this function tabulates for each configuration
        the derivatives of

                                <ao|jy>
                                <ao|op|jy>

                            <ref|Q_frozen   |ao>
                            <ref|Q_frozen op|ao>

        with respect to each element of coef, where Q_frozen is the
        projection operator onto the complement of the frozen subspace:

                        Q_frozen = 1 - |frozen_dual><frozen|

        (Q_frozen = 1 if there is no frozen orbitals)


        Note
        ----
        The only useful information of coef is its nesting pattern; its
        values are not used in this function.

        '''
        self.dao_jy = [None] * len(self.config)
        self.ref_Qfrozen_dao = [None] * len(self.config)

        for iconf, dat in enumerate(self.config):
            jy2dao = [jy2ao(nest(ci.tolist(), nestpat(coef)),
                            dat['natom'], dat['nbes'])
                      for ci in np.eye(len(flatten(coef)))]

            self.dao_jy[iconf] = \
                    np.array([jy2dao_i.T @ dat['jy_jy']
                              for jy2dao_i in jy2dao]) \
                    .transpose(1,0,2,3,4)

            self.ref_Qfrozen_dao[iconf] = \
                    np.array([dat['ref_jy'] @ jy2dao_i
                              for jy2dao_i in jy2dao])

            if self.spill_frozen is not None:
                self.ref_Qfrozen_dao[iconf] -= \
                        np.array([self.ref_Pfrozen_jy[iconf] @ jy2dao_i
                                  for jy2dao_i in jy2dao])

            self.ref_Qfrozen_dao[iconf] = \
                    self.ref_Qfrozen_dao[iconf].transpose(1,0,2,3,4)


    def _generalized_spillage(self, iconf, coef, ibands, with_grad=False):
        '''
        Generalized spillage function and its gradient with respect to
        spherical Bessel coefficients of a single configuration.

        '''
        dat = self.config[iconf]

        if ibands == 'all':
            ibands = range(dat['ref_ref'][1].shape[1])

        spill = (dat['wk'] @ dat['ref_ref'][1][:,ibands]).real.sum()
        _jy2ao = jy2ao(coef, dat['natom'], dat['nbes'])

        # <ref|Q_frozen|ao> and <ref|Q_frozen op|ao>
        V = dat['ref_jy'][:,:,ibands,:] @ _jy2ao
        if self.spill_frozen is not None:
            V -= self.ref_Pfrozen_jy[iconf][:,:,ibands,:] @ _jy2ao
            spill += self.spill_frozen[iconf][ibands].sum()

        # <ao|ao> and <ao|op|ao>
        W = _jy2ao.T @ dat['jy_jy'] @ _jy2ao

        V_dual = mrdiv(V[0], W[0]) # overlap only; no need for op
        VdaggerV = V_dual.transpose((0,2,1)).conj() @ V_dual

        spill += dat['wk'] @ (rfrob(W[1], VdaggerV)
                              - 2.0 * rfrob(V_dual, V[1]))
        spill /= len(ibands)

        if with_grad:
            # (d/dcoef)<ao|ao> and (d/dcoef)<ao|op|ao>
            dW = self.dao_jy[iconf] @ _jy2ao
            dW += dW.transpose((0,1,2,4,3)).conj()

            # (d/dcoef)<ref|Q_frozen|ao> and (d/dcoef)<ref|Q_frozen op|ao>
            dV = self.ref_Qfrozen_dao[iconf][:,:,:,ibands,:]

            grad = (rfrob(dW[1], VdaggerV)
                    - 2.0 * rfrob(V_dual, dV[1])
                    + 2.0 * rfrob(dV[0] - V_dual @ dW[0],
                                   mrdiv(V_dual @ W[1] - V[1], W[0]))
                    ) @ dat['wk']

            grad /= len(ibands)
            grad = nest(grad.tolist(), nestpat(coef))

        return (spill, grad) if with_grad else spill


    def opt(self, coef_init, coef_frozen, iconfs, ibands,
            options, nthreads=1):
        '''
        Spillage minimization w.r.t. spherical Bessel coefficients.

        Parameters
        ----------
            coef_init : nested list
                Initial guess for the coefficients.
                coef_init[itype][l][zeta][q] -> float.
            coef_frozen : nested list
                Coefficients for the frozen orbitals.
                coef_frozen[itype][l][zeta][q] -> float.
            iconfs : list of int or 'all'
                List of configuration indices to be included in the
                optimization. If 'all', all configurations are included.
            ibands : range/tuple or list of range/tuple
                Band indices to be included in the spillage calculation.
                If a range or tuple is given, the same indices are used
                for all configurations.
                If a list of range/tuple is given, each range/tuple will
                be applied to the configuration specified by iconfs
                respectively, in which case len(ibands) and len(iconfs)
                must agree.
            options : dict
                Options for the optimization.
            nthreads : int
                Number of threads for config-level parallelization.

        '''
        from multiprocessing.pool import ThreadPool
        pool = ThreadPool(nthreads)

        if coef_frozen is not None:
            self._tab_frozen(coef_frozen)

        self._tab_deriv(coef_init)

        if iconfs == 'all':
            iconfs = range(len(self.config))
        nconfs = len(iconfs)

        if not isinstance(ibands, list):
            ibands = [ibands] * nconfs

        assert len(ibands) == nconfs

        pat = nestpat(coef_init)
        def f(c): # function to be minimized
            s = lambda i: self._generalized_spillage(iconfs[i],
                                                     nest(c.tolist(), pat),
                                                     ibands[i],
                                                     with_grad=True)
            spills, grads = zip(*pool.map(s, range(nconfs)))
            return (sum(spills) / nconfs,
                    sum(np.array(flatten(g)) for g in grads) / nconfs)

        c0 = np.array(flatten(coef_init))

        bounds = [(-1.0, 1.0) for _ in c0]
        res = minimize(f, c0, jac=True, method='L-BFGS-B',
                       bounds=bounds, options=options)

        # to use basinhopping:
        #minimizer_kwargs = {"method": "L-BFGS-B", "jac": True,
        #                    "bounds": bounds}
        #res = basinhopping(f, c0, minimizer_kwargs=minimizer_kwargs,
        #                   niter=20, disp=True)

        pool.close()

        coef_opt = nest(res.x.tolist(), pat)
        return [[np.linalg.qr(np.array(coef_tl).T)[0].T.tolist()
                 if coef_tl else []
                 for coef_tl in coef_t] for coef_t in coef_opt]


class Spillage_jy(Spillage):
    '''
    Generalized spillage function and its optimization
    with spherical-wave reference states.

    '''
    def config_add(self, outdir, weight=(0.0, 1.0)):
        '''
        Adds a configuration by loading data from an OUT.{suffix} directory.

        The data will be processed and packed to a dict which is then
        appended to the config list. See the docstring of Spillage
        for details of the dict.

        '''
        raw = _jy_data_extract(outdir)
        C, S, T = raw['C'], raw['S'], raw['T']

        wov, wop = weight
        ref_ov_ref = np.sum(C.conj() * (S @ C), -2)
        ref_op_ref = np.sum(C.conj() * (T @ C), -2)

        ref_ov_jy = C.swapaxes(-2, -1).conj() @ S
        ref_op_jy = C.swapaxes(-2, -1).conj() @ T

        ref_ref = np.array([ref_ov_ref, wov*ref_ov_ref + wop*ref_op_ref])
        ref_jy  = np.array([ref_ov_jy , wov*ref_ov_jy  + wop*ref_op_jy ])
        jy_jy   = np.array([S, wov*S + wop*T])

        # NOTE: The LCAO basis in ABACUS follows a lexicographic order of
        # (itype, iatom, l, q, mm) where mm = 2*|m|-(m>0), which will be
        # transformed to a lexicographic order of (itype, iatom, l, mm, q)
        p = perm_zeta_m(_lin2comp(raw['natom'], nzeta=raw['nzeta']))
        ref_jy = ref_jy[:,:,:,p].copy()
        jy_jy = jy_jy[:,:,:,p][:,:,p,:].copy()

        self.config.append({
            'natom': raw['natom'],
            'nbes': raw['nzeta'],
            'wk': raw['wk'],
            'ref_ref': ref_ref,
            'ref_jy': ref_jy,
            'jy_jy': jy_jy,
            })


class Spillage_pw(Spillage):
    '''
    Generalized spillage function and its optimization
    with plane-wave reference states.

    '''

    def __init__(self, reduced=True):
        super().__init__()
        self.rcut = None


    def config_add(self, orb_matrix_0, orb_matrix_1, weight=(0.0, 1.0)):
        '''
        Adds a configuration by loading data from a pair of orb_matrix files.

        The data will be processed and packed to a dict which is then
        appended to the config list. See the docstring of Spillage
        for details of the dict.

        '''
        ov = read_orb_mat(orb_matrix_0)
        op = read_orb_mat(orb_matrix_1)

        wov, wop = weight

        ntype, natom, lmax, nbes, rcut = \
            [ov[key] for key in ['ntype', 'natom', 'lmax', 'nbes', 'rcut']]

        # sanity checks
        assert op['lin2comp'] == ov['lin2comp'] and \
                op['rcut'] == ov['rcut'] and \
                op['ecutwfc'] == ov['ecutwfc'] and \
                op['nbands'] == ov['nbands'] and \
                op['nbes'] == ov['nbes'] and \
                np.all(op['wk'] == ov['wk']) and \
                np.all(op['kpt'] == ov['kpt'])

        # NOTE currently a dataset merely contains one atom type
        # and one rcut. This may change in the future.
        if self.rcut is None:
            self.rcut = rcut
        else:
            assert self.rcut == rcut

        # The output of PW calculation is based on raw truncated spherical
        # waves, while the optimization will be performed w.r.t. the reduced
        # basis, so a basis transformation of ref_jy & jy_jy is needed.
        coef = [[jl_reduce(l, nbes, rcut).T.tolist()
                 for l in range(lmax[itype]+1)]
                for itype in range(ntype)]

        # number of spherical wave radial functions per l
        # nbes[itype][l] -> int
        nbes_raw = [[nbes] * (lmax[itype] + 1) for itype in range(ntype)]
        nbes_rdc = [[nbes - 1] * (lmax[itype] + 1) for itype in range(ntype)]

        # basis transformation matrix
        C = jy2ao(coef, natom, nbes_raw)

        ref_ref = np.array([ov['ref_ref'],
                            wov*ov['ref_ref'] + wop*op['ref_ref']])

        ref_jy = np.array([ov['ref_jy'] @ C,
                           (wov*ov['ref_jy'] + wop*op['ref_jy']) @ C])

        jy_jy = np.array([C.T @ ov['jy_jy'] @ C,
                          C.T @ (wov*ov['jy_jy'] + wop*op['jy_jy']) @ C])

        self.config.append({
            'natom': ov['natom'],
            'nbes': nbes_rdc,
            'wk': ov['wk'],
            'ref_ref': ref_ref,
            'ref_jy': ref_jy,
            'jy_jy': jy_jy,
            })


############################################################
#                           Test
############################################################
import unittest

from SIAB.spillage.radial import build_reduced, build_raw
from SIAB.spillage.plot import plot_chi

import matplotlib.pyplot as plt


class _TestSpillage(unittest.TestCase):

    def test_initgen_jy_gamma(self):
        outdir = './testfiles/Si/jy-7au/monomer-gamma/OUT.ABACUS/'

        nzeta = [3, 3, 2]
        ibands = range(19)
        coef = initgen_jy(outdir, nzeta, ibands=ibands, diagosis=False)

        self.assertEqual(len(coef), len(nzeta))
        self.assertEqual([len(coef_l) for coef_l in coef], nzeta)

        return # suppress the plot

        rcut = 7.0
        dr = 0.01
        r = np.linspace(0, rcut, int(rcut/dr)+1)
        chi = build_reduced(coef, rcut, r, True)

        plot_chi(chi, r)
        plt.show()


    def test_initgen_jy_k(self):
        outdir = './testfiles/Si/jy-7au/monomer-k/OUT.ABACUS/'

        nzeta = [3, 3, 2]
        ibands = range(22)
        coef = initgen_jy(outdir, nzeta, ibands=ibands, diagosis=False)

        self.assertEqual(len(coef), len(nzeta))
        self.assertEqual([len(coef_l) for coef_l in coef], nzeta)

        return # suppress the plot

        rcut = 7.0
        dr = 0.01
        r = np.linspace(0, rcut, int(rcut/dr)+1)
        chi = build_reduced(coef, rcut, r, True)

        plot_chi(chi, r)
        plt.show()


    def test_initgen_pw_gamma(self):
        orb_matrix = './testfiles/Si/pw/monomer-gamma/orb_matrix.0.dat'

        nzeta = [2, 2, 1]
        coef = initgen_pw(orb_matrix, nzeta, diagosis=False)

        self.assertEqual(len(coef), len(nzeta))
        self.assertEqual([len(coef[l]) for l in range(len(nzeta))], nzeta)

        return # suppress the plot

        dr = 0.01
        rcut = 7.0
        r = np.linspace(0, rcut, int(rcut/dr)+1)

        chi = build_reduced(coef, rcut, r, True)
        plot_chi(chi, r)
        plt.show()


    def test_jy_config_add(self):
        outdirs = [
                './testfiles/Si/jy-7au/monomer-gamma/OUT.ABACUS/',
                './testfiles/Si/jy-7au/dimer-1.8-k/OUT.ABACUS/',
                ]

        orbgen = Spillage_jy()

        for outdir in outdirs:
            orbgen.config_add(outdir)

        self.assertEqual(len(orbgen.config), 2)

        self.assertEqual(orbgen.config[0]['natom'], [1])
        self.assertEqual(len(orbgen.config[0]['nbes']), 1)
        self.assertEqual(len(orbgen.config[0]['nbes'][0]), 3)

        self.assertEqual(orbgen.config[1]['natom'], [2])
        self.assertEqual(len(orbgen.config[1]['nbes']), 1)
        self.assertEqual(len(orbgen.config[1]['nbes'][0]), 3)

        for conf in orbgen.config:
            njy = _nao(conf['natom'], conf['nbes'])
            nk = len(conf['wk'])
            nbands = conf['ref_ref'].shape[-1]

            self.assertEqual(conf['ref_ref'].shape, (2, nk, nbands))
            self.assertEqual(conf['ref_jy'].shape, (2, nk, nbands, njy))
            self.assertEqual(conf['jy_jy'].shape, (2, nk, njy, njy))


    def test_tab_frozen(self):
        outdirs = [
                './testfiles/Si/jy-7au/monomer-gamma/OUT.ABACUS/',
                './testfiles/Si/jy-7au/dimer-1.8-k/OUT.ABACUS/',
                ]

        orbgen = Spillage_jy()

        for outdir in outdirs:
            orbgen.config_add(outdir)

        coef_frozen = [[np.eye(3, 5).tolist(),
                        np.eye(3, 5).tolist(),
                        np.eye(2, 5).tolist()]]

        orbgen._tab_frozen(coef_frozen)

        self.assertEqual(len(orbgen.spill_frozen), len(outdirs))
        self.assertEqual(len(orbgen.ref_Pfrozen_jy), len(outdirs))

        for iconf, conf in enumerate(orbgen.config):
            njy = _nao(conf['natom'], conf['nbes'])
            nk = len(conf['wk'])
            nbands = conf['ref_ref'].shape[-1]

            self.assertEqual(orbgen.spill_frozen[iconf].shape,
                             (nbands,))
            self.assertEqual(orbgen.ref_Pfrozen_jy[iconf].shape,
                             (2, nk, nbands, njy))


    def test_tab_deriv(self):
        outdirs = [
                './testfiles/Si/jy-7au/monomer-k/OUT.ABACUS/',
                './testfiles/Si/jy-7au/dimer-1.8-gamma/OUT.ABACUS/',
                ]

        orbgen = Spillage_jy()

        for outdir in outdirs:
            orbgen.config_add(outdir)

        coef = [[np.eye(3, 5).tolist(),
                 np.eye(3, 5).tolist(),
                 np.eye(2, 5).tolist()]]

        orbgen._tab_deriv(coef)

        self.assertEqual(len(orbgen.dao_jy), len(outdirs))
        self.assertEqual(len(orbgen.ref_Qfrozen_dao), len(outdirs))

        for iconf, conf in enumerate(orbgen.config):
            ncoef = len(flatten(coef))
            njy = _nao(conf['natom'], conf['nbes'])
            nk = len(conf['wk'])
            nbands = conf['ref_ref'].shape[-1]

            nzeta = [[len(coef_tl) for coef_tl in coef_t] for coef_t in coef]
            nao = _nao(conf['natom'], nzeta)

            self.assertEqual(orbgen.dao_jy[iconf].shape,
                             (2, ncoef, nk, nao, njy))
            self.assertEqual(orbgen.ref_Qfrozen_dao[iconf].shape,
                             (2, ncoef, nk, nbands, nao))

        # verifies in the presence of frozen orbitals
        coef_frozen = [[np.eye(2, 5).tolist(),
                        np.eye(2, 5).tolist(),
                        np.eye(1, 5).tolist()]]
        orbgen._tab_frozen(coef_frozen)
        orbgen._tab_deriv(coef)

        self.assertEqual(len(orbgen.dao_jy), len(outdirs))
        self.assertEqual(len(orbgen.ref_Qfrozen_dao), len(outdirs))

        for iconf, conf in enumerate(orbgen.config):
            ncoef = len(flatten(coef))
            njy = _nao(conf['natom'], conf['nbes'])
            nk = len(conf['wk'])
            nbands = conf['ref_ref'].shape[-1]

            nzeta = [[len(coef_tl) for coef_tl in coef_t] for coef_t in coef]
            nao = _nao(conf['natom'], nzeta)

            self.assertEqual(orbgen.dao_jy[iconf].shape,
                             (2, ncoef, nk, nao, njy))
            self.assertEqual(orbgen.ref_Qfrozen_dao[iconf].shape,
                             (2, ncoef, nk, nbands, nao))


    def test_overlap_spillage_gamma(self):
        '''
        Verifies that generalized spillage with op = I
        recovers the overlap spillage.

        '''
        outdir = './testfiles/Si/jy-7au/dimer-1.8-gamma/OUT.ABACUS/'

        dat = _jy_data_extract(outdir)
        natom, nbes_data, wk, S, C = \
                [dat[key] for key in ['natom', 'nzeta', 'wk', 'S', 'C']]

        jy_jy = S
        ref_jy = C.swapaxes(-2, -1).conj() @ S
        ref_ref = np.sum(C.conj() * (S @ C), axis=1).real

        p = perm_zeta_m(_lin2comp(natom, nzeta=nbes_data))
        ref_jy = ref_jy[:,:,p].copy()
        jy_jy = jy_jy[:,:,p][:,p,:].copy()

        nzeta = [3, 3, 2]
        ibands = 'all'
        coef = [[np.random.randn(nzeta[l], nbes_tl).tolist()
                 for l, nbes_tl in enumerate(nbes_t)]
                for nbes_t in nbes_data]

        spill = _overlap_spillage(natom, nbes_data,
                                  jy_jy, ref_jy, ref_ref, wk,
                                  coef, ibands, None)

        orbgen = Spillage_jy()
        orbgen.config_add(outdir, (1.0, 0.0))
        spill2 = orbgen._generalized_spillage(0, coef, ibands)
        self.assertAlmostEqual(spill, spill2)

        # verifies in the presence of frozen orbitals
        coef_frozen = [[np.random.randn(nzeta[l], nbes_tl).tolist()
                        for l, nbes_tl in enumerate(nbes_t)]
                       for nbes_t in nbes_data]

        spill = _overlap_spillage(natom, nbes_data,
                                  jy_jy, ref_jy, ref_ref, wk,
                                  coef, ibands, coef_frozen)
        orbgen._tab_frozen(coef_frozen)
        spill2 = orbgen._generalized_spillage(0, coef, ibands)
        self.assertAlmostEqual(spill, spill2)


    def test_overlap_spillage_k(self):
        '''
        Verifies that generalized spillage with op = I
        recovers the overlap spillage.

        '''
        outdir = './testfiles/Si/jy-7au/dimer-1.8-k/OUT.ABACUS/'

        dat = _jy_data_extract(outdir)
        natom, nbes_data, wk, S, C = \
                [dat[key] for key in ['natom', 'nzeta', 'wk', 'S', 'C']]

        jy_jy = S
        ref_jy = C.swapaxes(-2, -1).conj() @ S
        ref_ref = np.sum(C.conj() * (S @ C), axis=1).real

        p = perm_zeta_m(_lin2comp(natom, nzeta=nbes_data))
        ref_jy = ref_jy[:,:,p].copy()
        jy_jy = jy_jy[:,:,p][:,p,:].copy()

        nzeta = [3, 3, 2]
        ibands = 'all'
        coef = [[np.random.randn(nzeta[l], nbes_tl).tolist()
                 for l, nbes_tl in enumerate(nbes_t)]
                for nbes_t in nbes_data]

        spill = _overlap_spillage(natom, nbes_data,
                                  jy_jy, ref_jy, ref_ref, wk,
                                  coef, ibands, None)

        orbgen = Spillage_jy()
        orbgen.config_add(outdir, (1.0, 0.0))
        spill2 = orbgen._generalized_spillage(0, coef, ibands)
        self.assertAlmostEqual(spill, spill2)

        # verifies in the presence of frozen orbitals
        coef_frozen = [[np.random.randn(nzeta[l], nbes_tl).tolist()
                        for l, nbes_tl in enumerate(nbes_t)]
                       for nbes_t in nbes_data]

        spill = _overlap_spillage(natom, nbes_data,
                                  jy_jy, ref_jy, ref_ref, wk,
                                  coef, ibands, coef_frozen)

        orbgen._tab_frozen(coef_frozen)
        spill2 = orbgen._generalized_spillage(0, coef, ibands)
        self.assertAlmostEqual(spill, spill2)


    def test_finite_difference(self):
        '''
        Checks the gradient of the generalized spillage
        with finite difference.

        '''
        outdir = './testfiles/Si/jy-7au/dimer-1.8-gamma/OUT.ABACUS/'
        orbgen = Spillage_jy()
        orbgen.config_add(outdir, (0.0, 1.0))

        nbes_data = read_running_scf_log(outdir + 'running_scf.log')['nzeta']

        nzeta = [2, 2, 1]
        ibands = 'all'
        coef = [[np.random.randn(nzeta[l], nbes_tl).tolist()
                 for l, nbes_tl in enumerate(nbes_t)]
                for nbes_t in nbes_data]

        coef_frozen = [[np.random.randn(nzeta[l], nbes_tl).tolist()
                        for l, nbes_tl in enumerate(nbes_t)]
                       for nbes_t in nbes_data]

        orbgen._tab_frozen(coef_frozen)
        orbgen._tab_deriv(coef)

        dspill = orbgen._generalized_spillage(0, coef, ibands, True)[1]
        dspill = np.array(flatten(dspill))

        pat = nestpat(coef)
        sz = len(flatten(coef))

        dspill_fd = np.zeros(sz)
        dc = 1e-6
        for i in range(sz):
            coef_p = flatten(deepcopy(coef))
            coef_p[i] += dc
            coef_p = nest(coef_p, pat)
            spill_p = orbgen._generalized_spillage(0, coef_p, ibands)

            coef_m = flatten(deepcopy(coef))
            coef_m[i] -= dc
            coef_m = nest(coef_m, pat)
            spill_m = orbgen._generalized_spillage(0, coef_m, ibands)

            dspill_fd[i] = (spill_p - spill_m) / (2 * dc)

        self.assertTrue(np.allclose(dspill, dspill_fd, atol=1e-7))


    def test_jy_opt(self):
        from listmanip import merge

        outdirs = [
                './testfiles/Si/jy-7au/dimer-1.8-gamma/OUT.ABACUS/',
                './testfiles/Si/jy-7au/dimer-2.8-gamma/OUT.ABACUS/',
                './testfiles/Si/jy-7au/dimer-3.8-gamma/OUT.ABACUS/',
                ]

        outdir_init = './testfiles/Si/jy-7au/monomer-gamma/OUT.ABACUS/'

        orbgen = Spillage_jy()
        for outdir in outdirs:
            orbgen.config_add(outdir)

        # coef_init[l][z][q] -> float
        coef_init = initgen_jy(outdir_init, [3, 3, 2], ibands='all')

        nthreads = 2
        options = {'ftol': 0, 'gtol': 1e-6, 'maxiter': 2000,
                   'disp': False, 'maxcor': 20}

        #======================================
        #               level-1
        #======================================
        ibands = range(4)
        iconfs = [0, 1, 2]
        # coef in optimization: coef[TYPE][l][z][q] -> float
        coef_lvl1_init = [[[coef_init[0][0]],
                           [coef_init[1][0]]]]
        coef_lvl1 = orbgen.opt(coef_lvl1_init, None, iconfs, ibands,
                               options, nthreads)
        coef_tot = coef_lvl1

        #======================================
        #               level-2
        #======================================
        ibands = range(8)
        iconfs = [0, 1, 2]
        coef_lvl2_init = [[[coef_init[0][1]],
                           [coef_init[1][1]],
                           [coef_init[2][0]]]]
        coef_lvl2 = orbgen.opt(coef_lvl2_init, coef_lvl1, iconfs, ibands,
                               options, nthreads)
        coef_tot = merge(coef_tot, coef_lvl2, 2)

        #======================================
        #               level-3
        #======================================
        ibands = range(12)
        iconfs = [0, 1, 2]
        coef_lvl3_init = [[[coef_init[0][2]],
                           [coef_init[1][2]],
                           [coef_init[2][1]]]]
        coef_lvl3 = orbgen.opt(coef_lvl3_init, coef_tot, iconfs, ibands,
                               options, nthreads)
        coef_tot = merge(coef_tot, coef_lvl3, 2)

        return # supress the plot 

        dr = 0.01
        rcut = 7.0
        r = np.linspace(0, rcut, int(rcut/dr)+1)
        chi = build_reduced(coef_tot[0], rcut, r, True)

        plot_chi(chi, r)
        plt.show()


    def test_pw_config_add_gamma(self):
        '''
        Only gamma-only calculation is supported at this stage;
        will be extended to multi-k calculations in the future.

        '''
        outdirs = [
                './testfiles/Si/pw/dimer-1.8-gamma/',
                './testfiles/Si/pw/dimer-2.8-gamma/',
                './testfiles/Si/pw/dimer-3.8-gamma/',
                ]

        orbgen = Spillage_pw()

        for outdir in outdirs:
            orbgen.config_add(outdir + 'orb_matrix.0.dat',
                              outdir + 'orb_matrix.1.dat')

        self.assertEqual(len(orbgen.config), 3)

        self.assertEqual(orbgen.config[0]['natom'], [2])
        self.assertEqual(len(orbgen.config[0]['nbes']), 1)
        self.assertEqual(len(orbgen.config[0]['nbes'][0]), 3)

        self.assertEqual(orbgen.config[1]['natom'], [2])
        self.assertEqual(len(orbgen.config[1]['nbes']), 1)
        self.assertEqual(len(orbgen.config[1]['nbes'][0]), 3)

        for conf in orbgen.config:
            njy = _nao(conf['natom'], conf['nbes'])
            nk = len(conf['wk'])
            nbands = conf['ref_ref'].shape[-1]

            self.assertEqual(conf['ref_ref'].shape, (2, nk, nbands))
            self.assertEqual(conf['ref_jy'].shape, (2, nk, nbands, njy))
            self.assertEqual(conf['jy_jy'].shape, (2, nk, njy, njy))


    def test_pw_opt(self):
        from listmanip import merge

        outdirs = [
                './testfiles/Si/pw/dimer-1.8-gamma/',
                './testfiles/Si/pw/dimer-2.8-gamma/',
                './testfiles/Si/pw/dimer-3.8-gamma/',
                ]

        orbgen = Spillage_pw()
        for outdir in outdirs:
            orbgen.config_add(outdir + 'orb_matrix.0.dat',
                              outdir + 'orb_matrix.1.dat')

        orb_matrix_init = './testfiles/Si/pw/monomer-gamma/orb_matrix.0.dat'
        nzeta = [2, 2, 1]

        # coef_init[l][z][q] -> float
        coef_init = initgen_pw(orb_matrix_init, nzeta, diagosis=False)

        nthreads = 2
        options = {'ftol': 0, 'gtol': 1e-6, 'maxiter': 2000,
                   'disp': False, 'maxcor': 20}

        #======================================
        #               level-1
        #======================================
        ibands = range(4)
        iconfs = [0, 1, 2]
        # coef in optimization: coef[TYPE][l][z][q] -> float
        coef_lvl1_init = [[[coef_init[0][0]],
                           [coef_init[1][0]]]]
        coef_lvl1 = orbgen.opt(coef_lvl1_init, None, iconfs, ibands,
                               options, nthreads)
        coef_tot = coef_lvl1

        #======================================
        #               level-2
        #======================================
        ibands = range(8)
        iconfs = [0, 1, 2]
        coef_lvl2_init = [[[coef_init[0][1]],
                           [coef_init[1][1]],
                           [coef_init[2][0]]]]
        coef_lvl2 = orbgen.opt(coef_lvl2_init, coef_lvl1, iconfs, ibands,
                               options, nthreads)
        coef_tot = merge(coef_tot, coef_lvl2, 2)

        return # supress the plot 

        dr = 0.01
        rcut = 7.0
        r = np.linspace(0, rcut, int(rcut/dr)+1)
        chi = build_reduced(coef_tot[0], rcut, r, True)

        plot_chi(chi, r)
        plt.show()


if __name__ == '__main__':
    unittest.main()


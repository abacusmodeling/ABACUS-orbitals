from SIAB.spillage.radial import _nbes, jl_reduce, jl_raw_norm,\
        coeff_normalized2raw, coeff_reduced2raw
from SIAB.spillage.listmanip import flatten, nest, nestpat
from SIAB.spillage.jlzeros import JLZEROS
from SIAB.spillage.index import index_map, perm_zeta_m, _nao
from SIAB.spillage.linalg_helper import mrdiv, rfrob
from SIAB.spillage.basistrans import jy2ao
from SIAB.spillage.datparse import read_orb_mat, _assert_consistency, \
        read_wfc_lcao_txt, read_abacus_csr

import glob
import numpy as np
from scipy.optimize import minimize, basinhopping
from copy import deepcopy


def _overlap_spillage(natom, lmax, rcut, nbes,
                      jy_jy, mo_jy, mo_mo, wk,
                      coef, ibands, coef_frozen=None):
    '''
    Standard spillage function (overlap spillage).

    Note
    ----
    This function is not supposed to be used in the optimization.
    As a special case of the generalized spillage (op = I), it serves
    as a cross-check for the implementation of the generalized spillage.

    '''
    spill = (wk @ mo_mo[:,ibands]).real.sum()

    mo_jy = mo_jy[:,ibands,:]
    _jy2ao = jy2ao(coef, natom, lmax, nbes, rcut)
    V = mo_jy @ _jy2ao
    W = _jy2ao.T @ jy_jy @ _jy2ao

    if coef_frozen is not None:
        jy2frozen = jy2ao(coef_frozen, natom, lmax, nbes, rcut)
        X = mo_jy @ jy2frozen
        S = jy2frozen.T @ jy_jy @ jy2frozen
        X_dual = mrdiv(X, S)

        V -= X_dual @ jy2frozen.T @ jy_jy @ _jy2ao
        spill -= wk @ rfrob(X_dual, X)

    spill -= wk @ rfrob(mrdiv(V, W), V)

    return spill / len(ibands)


def initgen_pw(nzeta, ecut, lmax, rcut, nbes_raw, mo_jy, wk,
               reduced=False, diagosis=False):
    '''
    Generates an initial guess for the spherical Bessel coefficients from
    the single-atom overlap data (plane-wave reference state).

    Parameters
    ----------
        nzeta : list of int
            Target number of zeta for each l.
        ecut : float
            Kinetic energy cutoff for the target coefficients.
            Note that the deduced number of spherical Bessel components
            must not be larger than nbes.
        lmax : int
            Maximum l of the single-atom data
        rcut : float
            Cutoff radius.
        nbes_raw : int
            Number of raw spherical Bessel components for each l of the
            single-atom data.
        mo_jy : ndarray, shape (nk, nbands, nao*nbes)
            Overlap between the single-atom reference states and raw-
            truncated spherical waves (jy).
        wk : ndarray, shape (nk,)
            k-point weights.
        reduced : bool
            If true, the initial guess is generated in terms of the
            reduced spherical Bessel basis; otherwise in the normalized
            truncated spherical Bessel basis.

    '''
    # maximum l of the generated coefficients
    lmax_gen = len(nzeta) - 1
    assert lmax_gen <= lmax

    # number of spherical Bessel components for each l
    # of the generated coefficients
    nbes_gen = [_nbes(l, rcut, ecut) for l in range(lmax_gen + 1)]
    assert all(n > 0 and n <= nbes_raw for n in nbes_gen)

    # transform <mo|jy(raw)> to <mo|jy(reduced)> or <mo|jy(normalized)>
    if reduced:
        coef = [[jl_reduce(l, nbes_raw, rcut).T.tolist()
                 for l in range(lmax + 1)]]
    else: # normalized
        uvec = lambda v, k, n: [v if i == k else 0 for i in range(n)]
        coef = [[[uvec(1. / jl_raw_norm(l, q, rcut), q, nbes_raw)
                  for q in range(nbes_raw)]
                 for l in range(lmax + 1)]]

    nk, nbands, _ = mo_jy.shape
    nbes_now = nbes_raw - 1 if reduced else nbes_raw
    Y = (mo_jy @ jy2ao(coef, [1], [lmax], nbes_raw, rcut)) \
            .reshape(nk, nbands, -1, nbes_now)

    coef = []
    for l in range(lmax_gen + 1):
        Yl = Y[:, :, l*l:(l+1)*(l+1), :] \
                .reshape(nk, -1, nbes_now)[:,:,:nbes_gen[l]]

        YdaggerY = ((Yl.transpose((0, 2, 1)).conj() @ Yl)
                    * wk.reshape(-1, 1, 1)).sum(0).real

        val, vec = np.linalg.eigh(YdaggerY)

        # eigenvectors corresponding to the largest nzeta eigenvalues
        coef.append(vec[:,-nzeta[l]:][:,::-1])
        # NOTE coef is column-wise at this stage; to be transposed later

        if diagosis:
            print( "ORBGEN: <jy|mo><mo|jy> eigval diagnosis:")
            print(f"        l = {l}: {val[-nzeta[l]:][::-1]}")

    return [np.linalg.qr(coef_l)[0].T.tolist() for coef_l in coef]


def initgen(nzeta, ecut, rcut, nbes, mo_jy, wk, diagosis=False):
    '''
    Generates an initial guess of the spherical Bessel coefficients from
    single-atom overlap data.

    Parameters
    ----------
        nzeta : list of int
            Target number of zeta for each l.
        ecut : float
            Kinetic energy cutoff used to deduce the size of target
            coefficients. Note that ecut should not yield numbers
            larger than the counterparts in nbes.
        rcut : float
            Cutoff radius.
        nbes : list of int or a 2-tuple of int 
            Number of spherical Bessel components for each l of the
            single-atom data (mo_jy).
            If a list, nbes[l] is the number corresponding to l.
            If a 2-tuple, it is taken as (lmax, nbes_each) and the same
            number of spherical Bessel components is assumed for all l.
        mo_jy : ndarray, shape (nk, nbands, njy)
            Overlap between single-atom reference states and spherical
            waves (jy). The order of jy must be lexicographic in terms
            of (l, m, q) where l & m are the angular momentum indices
            and q is the radial index.
        wk : ndarray, shape (nk,)
            Weights of k points.
        diagosis : bool
            If true, print the eigenvalues of <jy|mo><mo|jy> for each l.

    Returns
    -------
        A nested list. coef[l][zeta][q] -> float.

    Note
    ----
    This function does not discriminate various types of spherical waves
    (raw/normalized/reduced). The generated coefficients are in the same
    as that of the input data (mo_jy). Note, however, that the coefficients
    are always "normalized" in the sense that norm(coef[l][zeta]) = 1 for
    any l and zeta, which means they yield normalized orbitals in the case
    of normalized/reduced spherical waves, but not for raw spherical waves.

    '''
    if isinstance(nbes, tuple):
        lmax, nbes_each = nbes
        nbes = [nbes_each] * (lmax + 1)
    else:
        lmax = len(nbes) - 1

    # maximum l of the generated coefficients
    lmax_gen = len(nzeta) - 1
    assert lmax_gen <= lmax

    # number of spherical Bessel components for each l
    # of the generated coefficients
    # TODO & FIXME we need to know whether it's reduced or normalized
    # to determine the number of Bessel components
    nbes_gen = [_nbes(l, rcut, ecut) -1 for l in range(lmax_gen + 1)]
    assert all(nbes_gen[l] > 0 and nbes_gen[l] <= nbes[l]
               for l in range(lmax_gen+1))

    nk, nbands, njy = mo_jy.shape

    # number of spherical waves per l
    njy_l = [(2*l+1) * nbes[l] for l in range(lmax+1)]
    assert(sum(njy_l) == njy)

    index_l_start = [0] + np.cumsum(njy_l).tolist()
    Y = [mo_jy[:,:,range(index_l_start[l], index_l_start[l+1])]
         .reshape(nk, -1, nbes[l]) # reshape to (nk, nbands*(2*l+1), nbes[l])
         [:,:,:nbes_gen[l]]
         for l in range(lmax+1)]

    coef = []
    for l in range(lmax_gen + 1):
        if nzeta[l] == 0:
            coef.append([])
            continue

        YdaggerY = ((Y[l].swapaxes(-2, -1).conj() @ Y[l])
                    * np.array(wk).reshape(-1, 1, 1)).sum(0).real

        val, vec = np.linalg.eigh(YdaggerY)

        # eigenvectors corresponding to the largest nzeta eigenvalues
        coef.append(vec[:,-nzeta[l]:][:,::-1].T.tolist())

        if diagosis:
            print( "ORBGEN: <jy|mo><mo|jy> eigval diagnosis:")
            print(f"        l = {l}: {val[-nzeta[l]:][::-1]}")

    return coef


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

        self.spill_frozen = None
        self.mo_Pfrozen_jy = None
        self.mo_Qfrozen_dao = []
        self.dao_jy = []


    def config_add_pw(self, config, weight=(0.0, 1.0)):
        '''
        Parses the plane-wave data and appends it to the configuration list.

        Parameters
        ----------
            config : str or 2-tuple of str
                If a str is given, it is taken as the path to the plane-wave
                job directory.
                If 2-tuple of str is given, it is understood as the paths to
                the data files (orb_matrix.x.dat).
            weight : tuple of float
                Weights for the overlap and operator data respectively.

        '''
        if isinstance(config, str):
            ov = read_orb_mat(config + '/orb_matrix.0.dat')
            op = read_orb_mat(config + '/orb_matrix.1.dat')
        else:
            ov = read_orb_mat(config[0])
            op = read_orb_mat(config[1])

        wov, wop = weight

        # The overlap and operator data must be consistent except
        # for their matrix data (mo_mo, mo_jy and jy_jy).
        _assert_consistency(ov, op)

        ntype, natom, lmax, nbes, rcut = \
            [ov[key] for key in ['ntype', 'natom', 'lmax', 'nbes', 'rcut']]

        # The output of PW calculation is based on raw truncated spherical
        # Bessel basis, while the optimization will be performed in either
        # reduced or normalized basis. Here we transform mo_jy & jy_jy to
        # the basis of the optimization.
        if self.reduced:
            coef = [[jl_reduce(l, nbes, rcut).T.tolist()
                     for l in range(lmax[itype]+1)]
                    for itype in range(ntype)]
        else:
            uvec = lambda v, k, n: [v if i == k else 0 for i in range(n)]
            coef = [[[uvec(1. / jl_raw_norm(l, q, rcut), q, nbes)
                      for q in range(nbes)]
                     for l in range(lmax[itype]+1)]
                    for itype in range(ntype)]

        # basis transformation matrix
        C = jy2ao(coef, natom, lmax, nbes, rcut)
        #print(f"coef.shape = {C.shape}")
        #print(f"natom = {natom}")
        #print(f"lmax = {lmax}")
        #print(f"nbes = {nbes}")
        #print(f"rcut = {rcut}")


        # packed data for a configuration
        dat = {'ntype': ntype,
               'natom': natom,
               'lmax': lmax,
               'nbes': nbes-1 if self.reduced else nbes,
               'rcut': rcut,
               'nbands': ov['nbands'],
               'nk': ov['nk'],
               'wk': ov['wk'],
               'mo_mo': np.array([ov['mo_mo'],
                                  wov*ov['mo_mo'] + wop*op['mo_mo']]),
               'mo_jy': np.array([ov['mo_jy'] @ C,
                                  (wov*ov['mo_jy'] + wop*op['mo_jy']) @ C]),
               'jy_jy': np.array([C.T @ ov['jy_jy'] @ C,
                                  C.T @ (wov*ov['jy_jy'] + wop*op['jy_jy'])
                                  @ C]),
               }

        #print(f"mo_jy.shape = {dat['mo_jy'].shape}")
        #print(f"jy_jy.shape = {dat['jy_jy'].shape}")
        #print(f"mo_mo.shape = {dat['mo_mo'].shape}")
        #print('')

        self.config.append(dat)

        # NOTE currently a dataset merely contains one atom type
        # and one rcut. This may change in the future.
        if self.rcut is None:
            self.rcut = dat['rcut']
        else:
            assert self.rcut == ov['rcut']


    def config_add_jy(self, config, nbes, rcut, weight=(0.0, 1.0)):
        '''


        '''
        from struio import read_stru

        file_stru = glob.glob(config + '/STRU*')
        assert len(file_stru) == 1

        stru = read_stru(file_stru[0])
        ntype = len(stru['species'])
        natom = [spec['natom'] for spec in stru['species']]
        lmax = [len(nbes_t) - 1 for nbes_t in nbes]

        file_SR = glob.glob(config + '/OUT*/data-SR-sparse_SPIN0.csr')
        file_TR = glob.glob(config + '/OUT*/data-TR-sparse_SPIN0.csr')
        assert len(file_SR) == 1 and len(file_TR) == 1

        S = sum(read_abacus_csr(file_SR[0])[0]).toarray() # S(k=0)
        T = sum(read_abacus_csr(file_TR[0])[0]).toarray() # T(k=0)

        file_wfc = glob.glob(config + '/OUT*/WFC_NAO*')
        nk = len(file_wfc)

        # currently only supports Gamma point (with possibly nspin=2)
        assert nk == 1 or nk == 2 

        if nk == 2:
            wk = [0.5, 0.5]
            wfc = np.array([read_wfc_lcao_txt(file_wfc[0])[0],
                            read_wfc_lcao_txt(file_wfc[1])[0]])
            S = np.array([S, S])
            T = np.array([T, T])
        else:
            wk = [1.0]
            wfc = np.expand_dims(read_wfc_lcao_txt(file_wfc[0])[0], axis=0)
            S = np.expand_dims(S, axis=0)
            T = np.expand_dims(T, axis=0)

        # wfc has shape (nk, nbands, nao)
        nbands = wfc.shape[-2]

        # simply set mo_mo_ov = 1 instead of calculating it
        mo_mo_op = np.real(np.sum((wfc.conj() @ T) * wfc, 2))
        mo_jy_ov = wfc.conj() @ S
        mo_jy_op = wfc.conj() @ T

        wov, wop = weight
        mo_mo_ov = np.ones((nk, nbands))

        mo_mo = np.array([mo_mo_ov, wov*mo_mo_ov + wop*mo_mo_op])
        mo_jy = np.array([mo_jy_ov, wov*mo_jy_ov + wop*mo_jy_op])
        jy_jy = np.array([S, wov*S + wop*T])

        # NOTE: The LCAO basis in ABACUS follows a lexicographic order of
        # (itype, iatom, l, q, 2*|m|-(m>0))
        # which has to be transformed to a lexicographic order of
        # (itype, iatom, l, 2*|m|-(m>0), q)

        # permutation
        p = perm_zeta_m(index_map(natom, lmax, nbes)[1])
        mo_jy = mo_jy[:,:,:,p]
        jy_jy = jy_jy[:,:,:,p][:,:,p,:]

        print(f"mo_jy.shape = {mo_jy.shape}")
        print(f"jy_jy.shape = {jy_jy.shape}")
        print(f"mo_mo.shape = {mo_mo.shape}")

        # packed data for a configuration
        dat = {'ntype': ntype,
               'natom': natom,
               'lmax': lmax,
               'nbes': nbes[0],
               'rcut': rcut,
               'nbands': nbands,
               'nk': nk,
               'wk': wk,
               'mo_mo': mo_mo,
               'mo_jy': mo_jy,
               'jy_jy': jy_jy,
               }

        self.config.append(dat)

        # NOTE currently a dataset merely contains one atom type
        # and one rcut. This may change in the future.
        if self.rcut is None:
            self.rcut = dat['rcut']
        else:
            assert self.rcut == dat['rcut']


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
        jy2frozen = [jy2ao(coef_frozen, dat['natom'], dat['lmax'],
                           dat['nbes'], dat['rcut'])
                     for dat in self.config]

        frozen_frozen = [jy2froz.T @ dat['jy_jy'] @ jy2froz
                         for dat, jy2froz in zip(self.config, jy2frozen)]

        mo_frozen = [dat['mo_jy'] @ jy2froz
                     for dat, jy2froz in zip(self.config, jy2frozen)]

        # <mo|frozen_dual> only; no need to compute <mo|op|frozen_dual>
        mo_frozen_dual = [mrdiv(mo_froz[0], froz_froz[0])
                          for mo_froz, froz_froz
                          in zip(mo_frozen, frozen_frozen)]

        # for each config, indexed as [0/1][k][mo][jy]
        self.mo_Pfrozen_jy = [mo_froz_dual @ jy2froz.T @ dat['jy_jy']
                              for mo_froz_dual, dat, jy2froz
                              in zip(mo_frozen_dual, self.config, jy2frozen)]

        tmp = [rfrob(mo_froz_dual @ froz_froz[1], mo_froz_dual, True)
               - 2.0 * rfrob(mo_froz_dual, mo_froz[1], True)
               for mo_froz_dual, mo_froz, froz_froz
               in zip(mo_frozen_dual, mo_frozen, frozen_frozen)]

        # weighted sum over k
        self.spill_frozen = [dat['wk'] @ spill_froz
                             for dat, spill_froz
                             in zip(self.config, tmp)]


    def _tab_deriv(self, coef):
        '''
        Tabulates for each configuration the derivatives of

                                <ao|jy>
                                <ao|op|jy>

                            <mo|Q_frozen   |ao>
                            <mo|Q_frozen op|ao>

        with respect to each Bessel coefficient of |ao>, where Q_frozen is
        the projection operator onto the complement of the frozen subspace:

                        Q_frozen = 1 - |frozen_dual><frozen|

        (Q_frozen = 1 if there is no frozen orbitals)


        Note
        ----
        The only useful information of coef is its nesting pattern, which
        determines what derivatives to compute.

        '''
        # jy -> (d/dcoef)ao transformation matrices
        jy2dao_all = [[jy2ao(nest(ci.tolist(), nestpat(coef)), dat['natom'],
                             dat['lmax'], dat['nbes'], dat['rcut'])
                       for ci in np.eye(len(flatten(coef)))]
                      for dat in self.config]

        # derivatives of <ao|jy>, indexed as [ov/op][icoef][k][ao][jy]
        # for each config
        print('nestpat: ', nestpat(jy2dao_all))
        print('shape: ', jy2dao_all[0][0].shape)
        print('dao_jy...')
        self.dao_jy = [np.array([jy2dao_i.T @ dat['jy_jy']
                                 for jy2dao_i in jy2dao])
                       .transpose(1,0,2,3,4)
                       for dat, jy2dao in zip(self.config, jy2dao_all)]
        print('...done')

        # derivatives of <mo|ao> and <mo|op|ao>
        self.mo_Qfrozen_dao = [np.array([dat['mo_jy'] @ jy2dao_i
                                         for jy2dao_i in jy2dao])
                               for dat, jy2dao in zip(self.config, jy2dao_all)
                               ]
        # at this stage, the index for each config follows
        # [icoef][ov/op][k][mo][ao]
        # where 0->overlap; 1->operator

        if self.spill_frozen is not None:
            # subtract from the previous results <mo|P_frozen|ao>
            # and <mo|P_frozen op|ao>
            self.mo_Qfrozen_dao = [mo_Qfroz_dao -
                                   np.array([mo_Pfroz_jy @ jy2dao_i
                                             for jy2dao_i in jy2dao])
                                   for mo_Qfroz_dao, mo_Pfroz_jy, jy2dao
                                   in zip(self.mo_Qfrozen_dao,
                                          self.mo_Pfrozen_jy,
                                          jy2dao_all)
                                   ]

        # transpose to [0/1][icoef][k][mo][ao]
        self.mo_Qfrozen_dao = [dV.transpose(1,0,2,3,4)
                               for dV in self.mo_Qfrozen_dao]



    def _generalize_spillage(self, iconf, coef, ibands, with_grad=False):
        '''
        Generalized spillage function and its gradient.

        '''
        dat = self.config[iconf]

        spill = (dat['wk'] @ dat['mo_mo'][1][:,ibands]).real.sum()

        # jy->ao basis transformation matrix
        _jy2ao = jy2ao(coef, dat['natom'], dat['lmax'],
                       dat['nbes'], dat['rcut'])

        # <mo|Q_frozen|ao> and <mo|Q_frozen op|ao>
        V = dat['mo_jy'][:,:,ibands,:] @ _jy2ao
        if self.spill_frozen is not None:
            V -= self.mo_Pfrozen_jy[iconf][:,:,ibands,:] @ _jy2ao
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

            # (d/dcoef)<mo|Q_frozen|ao> and (d/dcoef)<mo|Q_frozen op|ao>
            dV = self.mo_Qfrozen_dao[iconf][:,:,:,ibands,:]

            grad = (rfrob(dW[1], VdaggerV)
                    - 2.0 * rfrob(V_dual, dV[1])
                    + 2.0 * rfrob(dV[0] - V_dual @ dW[0],
                                   mrdiv(V_dual @ W[1] - V[1], W[0]))
                    ) @ dat['wk']

            grad /= len(ibands)
            grad = nest(grad.tolist(), nestpat(coef))

        return (spill, grad) if with_grad else spill


    def opt(self, coef_init, coef_frozen, iconfs, ibands, options, nthreads=1):
        '''
        Spillage minimization w.r.t. (normalized or reduced) spherical
        Bessel coefficients.

        Parameters
        ----------
            coef_init : nested list
                Initial guess for the coefficients.
            coef_frozen : nested list
                Coefficients for the frozen orbitals.
            iconfs : list of int or 'all'
                List of configuration indices to be included in the
                optimization. If 'all', all configurations are included.
            ibands : range/tuple or list of range/tuple
                Band indices to be included in the spillage calculation.
                If a range or tuple is given, the same indices are used
                for all configurations.
                If a list of range/tuple is given, each range/tuple will
                be applied to the configuration specified by iconfs
                respectively.
            options : dict
                Options for the optimization.
            nthreads : int
                Number of threads for config-level parallellization.

        '''
        from multiprocessing.pool import ThreadPool
        pool = ThreadPool(nthreads)

        if coef_frozen is not None:
            self._tab_frozen(coef_frozen)

        print('tab deriv...')
        import time
        start = time.time()
        self._tab_deriv(coef_init)
        print('time elapsed = ', time.time()-start)
        exit()

        iconfs = range(len(self.config)) if iconfs == 'all' else iconfs
        nconf = len(iconfs)

        ibands = [ibands] * nconf if not isinstance(ibands, list) else ibands
        assert len(ibands) == nconf, f"len(ibands) = {len(ibands)} != {nconf}"

        pat = nestpat(coef_init)
        def f(c): # function to be minimized
            s = lambda i: self._generalize_spillage(iconfs[i],
                                                    nest(c.tolist(), pat),
                                                    ibands[i],
                                                    with_grad=True)
            spills, grads = zip(*pool.map(s, range(nconf)))
            return (sum(spills) / nconf,
                    sum(np.array(flatten(g)) for g in grads) / nconf)

        c0 = np.array(flatten(coef_init))

        # Restricts the coefficients to [-1, 1] for better numerical stability
        # FIXME Is this necessary?
        bounds = [(-1.0, 1.0) for _ in c0]
        #bounds = None

        res = minimize(f, c0, jac=True, method='L-BFGS-B',
                       bounds=bounds, options=options)

        #minimizer_kwargs = {"method": "L-BFGS-B", "jac": True,
        #                    "bounds": bounds}
        #res = basinhopping(f, c0, minimizer_kwargs=minimizer_kwargs,
        #                   niter=20, disp=True)

        pool.close()

        coef_opt = nest(res.x.tolist(), pat)
        return [[np.linalg.qr(np.array(coef_tl).T)[0].T.tolist()
                 if coef_tl else []
                 for coef_tl in coef_t] for coef_t in coef_opt]


############################################################
#                           Test
############################################################
import unittest

from SIAB.spillage.radial import build_reduced, build_raw
from SIAB.spillage.plot import plot_chi

import matplotlib.pyplot as plt


class _TestSpillage(unittest.TestCase):

    def setUp(self):
        self.orbgen_rdc = Spillage(True)
        self.orbgen_nrm = Spillage(False)

        #datadir_pw = './testfiles/Si/pw/'
        #config_pw = ['Si-dimer-1.8', 'Si-dimer-2.8', 'Si-dimer-3.8',
        #             'Si-trimer-1.7', 'Si-trimer-2.7']
        datadir_pw = '../../testfiles/pw_10au/'
        config_pw = ['Si-dimer-1.80', 'Si-dimer-2.80', 'Si-dimer-3.80',
                     'Si-trimer-1.70', 'Si-trimer-2.70',
                     ]

        self.config_pw = [datadir_pw + conf for conf in config_pw]
        self.config0_pw = datadir_pw + 'Si-monomer'

        datadir_jy = '../../testfiles/jy_10au/'
        config_jy = ['Si-dimer-1.80', 'Si-dimer-2.80', 'Si-dimer-3.80',
                     'Si-trimer-1.70', 'Si-trimer-2.70',
                     ]
        self.config_jy = [datadir_jy + conf for conf in config_jy]
        self.config0_jy = datadir_jy + 'Si-monomer'


    def randcoef(self, nzeta, nbes):
        '''
        Generates some random coefficients for unit tests.

        Parameters
        ----------
            nzeta : nested list of int
                nzeta[itype][l] gives the number of zeta.
            nbes : int
                Number of spherical Bessel basis functions for each zeta.

        Returns
        -------
            A nested list.
            coef[itype][l][zeta][q] -> float

        '''
        return [[np.random.randn(nzeta_tl, nbes).tolist()
                 for l, nzeta_tl in enumerate(nzeta_t)]
                for it, nzeta_t in enumerate(nzeta)]


    def est_initgen_pw(self):
        '''
        Checks intial guess generation from the plane-wave reference state.

        '''
        reduced = True # False: normalized; True: reduced

        ov = read_orb_mat('./testfiles/Si/pw/Si-monomer/orb_matrix.0.dat')
        nzeta = [2, 2, 1]
        rcut = ov['rcut']
        ecut = ov['ecutjlq']

        coef = initgen_pw(nzeta, ecut, len(nzeta)-1, rcut, ov['nbes'],
                          ov['mo_jy'], ov['wk'], reduced)

        self.assertEqual(len(coef), len(nzeta))
        self.assertEqual([len(coef[l]) for l in range(len(nzeta))], nzeta)

        return # suppress the plot

        dr = 0.01
        r = np.linspace(0, rcut, int(rcut/dr)+1)

        if reduced:
            chi = build_reduced(coef, rcut, r, True)
        else:
            chi = build_raw(coeff_normalized2raw(coef, rcut), rcut, r, True)

        plot_chi(chi, r)
        plt.show()


    def est_initgen(self):
        '''
        Checks intial guess generation from spherical-wave reference state.

        '''
        file_SR = './testfiles/Si/jy/Si-monomer/data-SR-sparse_SPIN0.csr'
        file_wfc = './testfiles/Si/jy/Si-monomer/WFC_NAO_K1.txt'

        S = sum(read_abacus_csr(file_SR)[0]).toarray() # S(k=0)
        S = np.expand_dims(S, axis=0) # (nk, nao, nao)

        nk = 1
        wk = np.array([1.0])
        wfc = np.expand_dims(read_wfc_lcao_txt(file_wfc)[0], axis=0)

        mo_jy = wfc.conj() @ S

        nzeta = [3, 3, 2]
        nbes = [16,15,15]
        p = perm_zeta_m(index_map([1], [2], [nbes])[1])
        mo_jy = mo_jy[:,:,p]

        coef = initgen(nzeta, 60.0, 7.0, [16,15,15], mo_jy, wk, True)

        self.assertEqual(len(coef), len(nzeta))
        self.assertEqual([len(coef[l]) for l in range(len(nzeta))], nzeta)

        return # suppress the plot

        dr = 0.01
        r = np.linspace(0, rcut, int(rcut/dr)+1)

        if reduced:
            chi = build_reduced(coef, rcut, r, True)
        else:
            chi = build_raw(coeff_normalized2raw(coef, rcut), rcut, r, True)

        plot_chi(chi, r)
        plt.show()


    def est_config_add_pw(self):
        '''
        Checks if config_add_pw loads & transforms data correctly.

        '''
        for conf in self.config_pw:
            for orbgen in [self.orbgen_nrm, self.orbgen_rdc]:
                orbgen.config_add_pw(conf)

        for iconf, config in enumerate(self.config_pw):
            for orbgen in [self.orbgen_nrm, self.orbgen_rdc]:

                dat = orbgen.config[iconf]
                njy = _nao(dat['natom'], dat['lmax']) * dat['nbes']

                self.assertEqual(dat['mo_mo'].shape,
                                 (2, dat['nk'], dat['nbands']))
                self.assertEqual(dat['mo_jy'].shape,
                                 (2, dat['nk'], dat['nbands'], njy))
                self.assertEqual(dat['jy_jy'].shape,
                                 (2, dat['nk'], njy, njy))

                # config_add_pw not only load data, but also performs
                # normalization/reduction. Here we check them by looking
                # at the overlap matrix on the first atom. Either way,
                # the overlap block should be close to the identity matrix.
                nao_0 = njy // dat['natom'][0]
                S = dat['jy_jy'][0, 0][:nao_0, :nao_0]
                self.assertTrue(np.allclose(S, np.eye(nao_0), atol=1e-6))

            self.assertEqual(self.orbgen_nrm.config[iconf]['nbes'],
                             self.orbgen_rdc.config[iconf]['nbes'] + 1)


    def est_config_add_jy(self):

        reduced = False
        orbgen = self.orbgen_rdc if reduced else self.orbgen_nrm

        orbgen.config_add_jy(self.config_jy[0], [[30, 30, 29]], 7.0)


    def est_tab_frozen(self):
        '''
        Checks if data tabulated by tab_frozen have the correct shape.

        '''
        for conf in self.config_pw:
            for orbgen in [self.orbgen_nrm, self.orbgen_rdc]:
                orbgen.config_add_pw(conf)

        nbes_min = min(dat['nbes'] for dat in self.orbgen_rdc.config)
        coef_frozen = self.randcoef([[2, 1, 0]], nbes_min)

        for orbgen in [self.orbgen_nrm, self.orbgen_rdc]:
            orbgen._tab_frozen(coef_frozen)
            self.assertEqual(len(orbgen.mo_Pfrozen_jy), len(self.config_pw))
            self.assertEqual(len(orbgen.spill_frozen), len(self.config_pw))

            for iconf, config in enumerate(self.config_pw):
                dat = orbgen.config[iconf]
                njy = _nao(dat['natom'], dat['lmax']) * dat['nbes']
                self.assertEqual(orbgen.spill_frozen[iconf].shape,
                                 (dat['nbands'],))
                self.assertEqual(orbgen.mo_Pfrozen_jy[iconf].shape,
                                 (2, dat['nk'], dat['nbands'], njy))


    def est_tab_deriv(self):
        '''
        Checks if data tabulated by tab_deriv have the correct shape.

        '''
        for conf in self.config_pw:
            for orbgen in [self.orbgen_nrm, self.orbgen_rdc]:
                orbgen.config_add_pw(conf)

        nbes_min = min(dat['nbes'] for dat in self.orbgen_rdc.config)
        coef = self.randcoef([[2, 1, 0]], nbes_min)

        # total number of coefficients
        ncoef = len(flatten(coef))

        # number of spherical Bessel basis related to coef
        njy_ao = [sum(len(coef_tl) * (2*l+1)
                      for l, coef_tl in enumerate(coef_t))
                    for coef_t in coef]

        for orbgen in [self.orbgen_nrm, self.orbgen_rdc]:
            orbgen._tab_deriv(coef)

            self.assertEqual(len(orbgen.dao_jy), len(self.config_pw))
            self.assertEqual(len(orbgen.mo_Qfrozen_dao), len(self.config_pw))

            for iconf, config in enumerate(self.config_pw):
                dat = orbgen.config[iconf]
                n_dao = np.dot(njy_ao, dat['natom'])
                njy = _nao(dat['natom'], dat['lmax']) * dat['nbes']
                self.assertEqual(orbgen.dao_jy[iconf].shape,
                                 (2, ncoef, dat['nk'], n_dao, njy))


    def est_overlap_spillage(self):
        '''
        Verifies that generalized spillage with op = I
        recovers the overlap spillage.

        '''
        for conf in self.config_pw:
            for orbgen in [self.orbgen_nrm, self.orbgen_rdc]:
                orbgen.config_add_pw((conf + '/orb_matrix.0.dat',
                                      conf + '/orb_matrix.0.dat'))

        ibands = range(5)
        nbes_min = min(dat['nbes'] for dat in self.orbgen_rdc.config)

        coef = self.randcoef([[2, 2, 1]], nbes_min)
        coef_frozen_list = [
                None,
                self.randcoef([[1, 1]], nbes_min),
                self.randcoef([[2, 1, 0]], nbes_min),
                self.randcoef([[0, 1, 1]], nbes_min),
                ]

        for orbgen in [self.orbgen_nrm, self.orbgen_rdc]:
            for coef_frozen in coef_frozen_list:
                orbgen._tab_frozen(coef_frozen)
                for iconf, config in enumerate(self.config_pw):
                    dat = orbgen.config[iconf]
                    spill_ref = _overlap_spillage(dat['natom'], dat['lmax'],
                                                  dat['rcut'], dat['nbes'],
                                                  dat['jy_jy'][0],
                                                  dat['mo_jy'][0],
                                                  dat['mo_mo'][0],
                                                  dat['wk'],
                                                  coef, ibands, coef_frozen)
                    spill = orbgen._generalize_spillage(iconf, coef, ibands)
                    self.assertAlmostEqual(spill, spill_ref, places=10)


    def est_finite_difference(self):
        '''
        Checks the gradient of the generalized spillage
        with finite difference.

        '''
        for conf in self.config_pw:
            for orbgen in [self.orbgen_nrm, self.orbgen_rdc]:
                orbgen.config_add_pw(conf)

        ibands = range(6)
        nbes_min = min(dat['nbes'] for dat in self.orbgen_rdc.config)

        coef = self.randcoef([[2, 1, 1]], nbes_min)
        coef_frozen_list = [None, self.randcoef([[1, 1]], nbes_min)]

        for orbgen in [self.orbgen_nrm, self.orbgen_rdc]:
            for coef_frozen in coef_frozen_list:
                orbgen._tab_frozen(coef_frozen)
                orbgen._tab_deriv(coef)

                for iconf, _ in enumerate(orbgen.config):
                    dspill = orbgen._generalize_spillage(iconf, coef,
                                                         ibands, True)[1]
                    dspill = np.array(flatten(dspill))

                    pat = nestpat(coef)
                    sz = len(flatten(coef))

                    dspill_fd = np.zeros(sz)
                    dc = 1e-6
                    for i in range(sz):
                        coef_p = flatten(deepcopy(coef))
                        coef_p[i] += dc
                        coef_p = nest(coef_p, pat)
                        spill_p = orbgen._generalize_spillage(iconf, coef_p,
                                                              ibands)

                        coef_m = flatten(deepcopy(coef))
                        coef_m[i] -= dc
                        coef_m = nest(coef_m, pat)
                        spill_m = orbgen._generalize_spillage(iconf, coef_m,
                                                              ibands)

                        dspill_fd[i] = (spill_p - spill_m) / (2 * dc)

                    self.assertTrue(np.allclose(dspill, dspill_fd, atol=1e-7))


    def est_opt_pw(self):
        from listmanip import merge

        reduced = True
        orbgen = self.orbgen_rdc if reduced else self.orbgen_nrm

        for conf in self.config_pw:
            orbgen.config_add_pw(conf)

        nthreads = 2
        options = {'ftol': 0, 'gtol': 1e-6, 'maxiter': 2000,
                   'disp': True, 'maxcor': 20}

        # initial guess
        ov = read_orb_mat(self.config0_pw + '/orb_matrix.0.dat')
        nzeta = [3, 3, 2]
        rcut = ov['rcut']
        ecut = ov['ecutjlq']

        coef_init = initgen_pw(nzeta, ecut, len(nzeta)-1, rcut, ov['nbes'],
                               ov['mo_jy'], ov['wk'], reduced)


        nbes = [_nbes(l, rcut, ecut) for l in range(len(nzeta))]
        if reduced:
            nbes = [n - 1 for n in nbes]

        coef_init = [np.eye(nz, nbes[l]).tolist()
                     for l, nz in enumerate(nzeta)]

        ibands = range(4)
        iconfs = [0, 1, 2]
        # coef_lvl1_init: [t][l][z][q]
        coef_lvl1_init = [[[coef_init[0][0]],
                           [coef_init[1][0]]]]
        coef_lvl1 = orbgen.opt(coef_lvl1_init, None, iconfs, ibands,
                               options, nthreads)
        coef_tot = coef_lvl1

        ibands = range(4)
        iconfs = [0, 1, 2]
        coef_lvl2_init = [[[coef_init[0][1]],
                           [coef_init[1][1]],
                           [coef_init[2][0]]]]
        coef_lvl2 = orbgen.opt(coef_lvl2_init, coef_lvl1, iconfs, ibands,
                               options, nthreads)
        coef_tot = merge(coef_tot, coef_lvl2, 2)

        ibands = range(8)
        iconfs = [3, 4]
        coef_lvl3_init = [[[coef_init[0][2]],
                           [coef_init[1][2]],
                           [coef_init[2][1]]]]
        coef_lvl3 = orbgen.opt(coef_lvl3_init, coef_tot, iconfs, ibands,
                               options, nthreads)
        coef_tot = merge(coef_tot, coef_lvl3, 2)

        #return # supress the plot 

        dr = 0.01
        r = np.linspace(0, rcut, int(rcut/dr)+1)

        if reduced:
            chi = build_reduced(coef_tot[0], rcut, r, True)
        else:
            coeff_raw = coeff_normalized2raw(coef_tot, rcut)
            chi = build_raw(coeff_raw[0], rcut, r, 0.0, True)

        plot_chi(chi, r)
        plt.show()


    def test_opt_jy(self):
        from listmanip import merge

        reduced = True 
        orbgen = self.orbgen_rdc if reduced else self.orbgen_nrm

        nbes = [30, 30, 29]
        rcut = 10.0
        ecut = 100

        for conf in self.config_jy:
            orbgen.config_add_jy(conf, [nbes], 10.0)

        nthreads = 2
        options = {'ftol': 0, 'gtol': 1e-6, 'maxiter': 2000,
                   'disp': True, 'maxcor': 20}

        # initial guess
        coef_init = [np.eye(3,30).tolist(),
                     np.eye(3,30).tolist(),
                     np.eye(2,29).tolist()]

        file_SR = glob.glob(self.config0_jy
                            + '/OUT*/data-SR-sparse_SPIN0.csr')[0]
        file_wfc = glob.glob(self.config0_jy
                             + '/OUT*/WFC_NAO*')[0]

        print(file_SR)
        print(file_wfc)

        S = sum(read_abacus_csr(file_SR)[0]).toarray() # S(k=0)
        wfc = np.expand_dims(read_wfc_lcao_txt(file_wfc)[0], axis=0)

        mo_jy = wfc.conj() @ S
        p = perm_zeta_m(index_map([1], [2], [nbes])[1])
        mo_jy = mo_jy[:,:,p]

        nzeta = [3, 3, 2]
        wk = [1.0]
        #coef_init = initgen(nzeta, ecut, rcut, nbes, mo_jy, wk, True)

        ibands = range(8)
        iconfs = [0, 1, 2]
        # coef_lvl1_init: [t][l][z][q]
        coef_lvl1_init = [[[coef_init[0][0]],
                           [coef_init[1][0]]]]
        coef_lvl1 = orbgen.opt(coef_lvl1_init, None, iconfs, ibands,
                               options, nthreads)
        coef_tot = coef_lvl1

        ibands = range(8)
        iconfs = [0, 1, 2]
        coef_lvl2_init = [[[coef_init[0][1]],
                           [coef_init[1][1]],
                           [coef_init[2][0]]]]
        coef_lvl2 = orbgen.opt(coef_lvl2_init, coef_lvl1, iconfs, ibands,
                               options, nthreads)
        coef_tot = merge(coef_tot, coef_lvl2, 2)

        #ibands = range(12)
        #iconfs = [3, 4]
        #coef_lvl3_init = [[[coef_init[0][2]],
        #                   [coef_init[1][2]],
        #                   [coef_init[2][1]]]]
        #coef_lvl3 = orbgen.opt(coef_lvl3_init, coef_tot, iconfs, ibands,
        #                       options, nthreads)
        #coef_tot = merge(coef_tot, coef_lvl3, 2)

        #return # supress the plot 

        dr = 0.01
        r = np.linspace(0, rcut, int(rcut/dr)+1)

        if reduced:
            chi = build_reduced(coef_tot[0], rcut, r, True)
        else:
            coeff_raw = coeff_normalized2raw(coef_tot, rcut)
            chi = build_raw(coeff_raw[0], rcut, r, 0.0, True)

        plot_chi(chi, r)
        plt.show()

if __name__ == '__main__':
    unittest.main()


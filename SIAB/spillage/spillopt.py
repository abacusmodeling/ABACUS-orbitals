from datparse import read_orb_mat
from indexmap import _index_map
from radial import jl_reduce

import numpy as np
from scipy.linalg import block_diag

import matplotlib.pyplot as plt

class SpillOpt:
    '''
    Orbital generation by minimizing the spillage.

    Attributes
    ----------
        config : list
            A list of 2-tuples like (ovlp_dat, op_dat). Each pair corresponds
            to a geometric configuration, where ovlp_dat and op_dat are data
            read from orb_matrix_rcutXderiv0.dat and orb_matrix_rcutXderiv1.dat.
            See read_orb_mat in datparse.py for details.
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
            the MOs and dual frozen orbitals.

    '''

    def __init__(self):
        self.reset()


    def reset(self):
        '''
        Clears all configurations and related data.

        '''
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
        # Checks if the cutoff radii are consistent.
        assert ovlp_dat['rcut'] == op_dat['rcut']
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
        Tabulates the frozen-orbital-related matrix elements, including

        <frozen|frozen>     <frozen|op|frozen>
        <mo|frozen>         <mo|op|frozen>
        <mo|frozen's dual>    <mo|op|frozen's dual>

        '''
        self.coef_frozen = coef_frozen

        self.frozen_frozen = [(
            self._ao_ao(coef_frozen, coef_frozen, ovlp_dat['jy_jy'],
                        ovlp_dat['mu2comp'], ovlp_dat['rcut']),
            self._ao_ao(coef_frozen, coef_frozen, op_dat['jy_jy'], 
                        op_dat['mu2comp'], op_dat['rcut'])
            ) for ovlp_dat, op_dat in self.config]

        self.mo_frozen = [(
            self._mo_ao(coef_frozen, ovlp_dat['mo_jy'], ovlp_dat['mu2comp'],
                        ovlp_dat['rcut']),
            self._mo_ao(coef_frozen, op_dat['mo_jy'], op_dat['mu2comp'],
                        op_dat['rcut'])
            ) for ovlp_dat, op_dat in self.config]

        self.mo_frozen_dual = [(
            self._make_dual(X[0], S[0]),
            self._make_dual(X[1], S[0])
            ) for X, S in zip(self.mo_frozen, self.frozen_frozen)]

    
    def _make_dual(self, X, S):
        '''
        Given two 3-d arrays X and S, returns a 3-d array Xdual such that

            Xdual[i] = X[i] * inv(S[i])

        '''
        assert len(X.shape) == 3 and len(S.shape) == 3
        return np.array([np.linalg.solve(S[i].T, X[i].T).T for i in range(X.shape[0])])


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


    def _gen_q2zeta(self, coef, mu2comp, nbes, rcut):
        '''
        Basis transformation matrix from the truncated spherical Bessel
        function to the pseudo-atomic orbital.

        Given an index map "mu2comp" (see indexmap.py), this generator generates
        for each mu the transformation matrix from the truncated spherical Bessel
        function to the pseudo-atomic orbital, which is a linear combination of
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
                yield np.zeros((nbes, 0))
            else:
                C = np.zeros((nbes-1, nzeta))
                C[:len(coef[itype][l][0])] = np.array(coef[itype][l]).T
                yield jl_reduce(l, nbes, rcut) @ C


    def _ao_ao(self, coef_bra, coef_ket, jy_jy, jy_mu2comp, rcut):
        '''
        Given matrix elements evaluated between jY, builds the matrix
        elements between pseudo-atomic orbitals specified by the given
        orthonormal end-smoothed mixed spherical Bessel coefficients.

        Parameters
        ----------
            coef_bra, coef_ket : nested list
                The coefficients for the orthonormal end-smoothed mixed
                spherical Bessel basis. coef[itype][l][zeta] gives a list of
                float (jY coefficients) that specifies a pseudo-atomic orbital.
                If coef_ket is None, it is assumed to be the same as coef_bra.
            jy_jy : np.ndarray
                The original matrix in jY basis as read from an
                orb_matrix_rcutXderivY.dat file.
                Shape: (nk, nao, nao, nbes, nbes)
            jy_mu2comp : dict
                Index map mu -> (itype, iatom, l, zeta, m).
                NOTE: zeta is supposed to be 0 for all mu.
            rcut : float
                Cutoff radius.

        '''
        nk, nao, nbes = jy_jy.shape[0], jy_jy.shape[1], jy_jy.shape[-1]

        # basis transformation matrix from the truncated spherical Bessel
        # function to the pseudo-atomic orbital
        M_bra = block_diag(*self._gen_q2zeta(coef_bra, jy_mu2comp, nbes, rcut))
        M_ket = block_diag(*self._gen_q2zeta(coef_ket, jy_mu2comp, nbes, rcut)) \
                if coef_ket is not None else M_bra

        # '@' would broadcast
        return M_bra.T.conj() \
                @ jy_jy.transpose((0,1,3,2,4)).reshape((nk, nao*nbes, nao*nbes)) \
                @ M_ket

    def _mo_ao(self, coef, mo_jy, jy_mu2comp, rcut, ibands=None):
        '''
        Given matrix elements evaluated between MO and jY, builds the matrix
        elements between MO and pseudo-atomic orbitals specified by the given
        orthonormal end-smoothed mixed spherical Bessel coefficients.

        Parameters
        ----------
            coef : nested list
                The coefficients for the orthonormal end-smoothed mixed
                spherical Bessel basis. coef[itype][l][zeta] gives a list of
                float (jy coefficients) that specifies a pseudo-atomic orbital.
            mo_jy : np.ndarray
                The overlap matrix of the truncated spherical Bessel function
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

        if ibands is None:
            ibands = range(nbands)

        # basis transformation matrix from the truncated spherical Bessel
        # function to the pseudo-atomic orbital
        M = block_diag(*self._gen_q2zeta(coef, jy_mu2comp, nbes, rcut))

        # '@' would broadcast
        return mo_jy.reshape((nk, nbands, nao*nbes))[:,ibands,:] @ M


    def _spillage_0(self, coef, ibands):
        '''
        Standard spillage function.

        '''
        s0 = 0.0

        for i, (dat, _) in enumerate(self.config):
            # weight of k-points, reshaped to enable broadcasting
            wk = dat['wk'].reshape(dat['nk'], 1, 1)

            W = self._ao_ao(coef, None, dat['jy_jy'], dat['mu2comp'], dat['rcut'])
            Y = self._mo_ao(coef, dat['mo_jy'], dat['mu2comp'], dat['rcut'], ibands)

            if self.coef_frozen is None:
                Ydual = self._make_dual(Y, W)
                s0 += 1.0 - (wk * Ydual * Y.conj()).sum().real / len(ibands)
            else:
                S = self.frozen_frozen[i]
                X = self.mo_frozen[i][:, ibands, :]
                Xdual = self.mo_frozen_dual[i][:, ibands, :]

                Z = self._ao_ao(coef_frozen, coef, dat['jy_jy'], dat['mu2comp'], dat['rcut'])

                V = Y - Xdual @ Z # '@' would broadcast
                Vdual = self._make_dual(V, W)

                s0 += 1.0 - (wk * Xdual * X.conj()).sum().real / len(ibands) \
                        - (wk * Vdual * V.conj()).sum().real / len(ibands)

        return s0


    def opt(self, coef0):
        '''
        '''
        pass











############################################################
#                           Test
############################################################
import unittest

class _TestSpillOpt(unittest.TestCase):

    def test_update_transform_table(self):
        orbgen = SpillOpt()

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
        orbgen = SpillOpt()

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


    def test_tab_frozen(self):
        orbgen = SpillOpt()

        folder = '/home/zuxin/abacus-community/abacus_orbital_generation/tmp/Si-dimer-2.0/'

        mat = read_orb_mat(folder + 'orb_matrix_rcut6deriv0.dat')
        dmat = read_orb_mat(folder + 'orb_matrix_rcut6deriv1.dat')

        orbgen.add_config(mat, dmat)
        orbgen._tab_frozen([[np.eye(2, mat['nbes']-1).tolist() for l in range(3)]])


    def test_gen_q2zeta(self):
        orbgen = SpillOpt()

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

        for mu, mat in enumerate(orbgen._gen_q2zeta(coef, mu2comp, nbes, rcut)):
            itype, iatom, l, zeta, m = mu2comp[mu]
            self.assertEqual(mat.shape, (nbes, nzeta[itype][l]))
            self.assertTrue( np.allclose(mat, \
                    jl_reduce(l, nbes, rcut) @ np.array(coef[itype][l]).T))


    def test_ao_ao(self):
        orbgen = SpillOpt()
        mat = read_orb_mat('./testfiles/orb_matrix_rcut6deriv0.dat')
        #mat = read_orb_mat('./testfiles/orb_matrix_rcut7deriv1.dat')

        # 2s2p1d
        coef = [[np.eye(2, mat['nbes']-1).tolist(), \
                np.eye(2, mat['nbes']-1).tolist(), \
                np.eye(1, mat['nbes']-1).tolist()]]

        S = orbgen._ao_ao(coef, coef, mat['jy_jy'], mat['mu2comp'], mat['rcut'])

        # overlap matrix should be hermitian
        for ik in range(mat['nk']):
            self.assertLess(np.linalg.norm(S[ik]-S[ik].T.conj(), np.inf), 1e-12)

        #plt.imshow(S[0])
        #plt.show()


    def test_mo_ao(self):
        orbgen = SpillOpt()
        mat = read_orb_mat('./testfiles/orb_matrix_rcut6deriv0.dat')

        # 2s2p1d
        coef = [[np.eye(2, mat['nbes']-1).tolist(), \
                np.eye(2, mat['nbes']-1).tolist(), \
                np.eye(1, mat['nbes']-1).tolist()]]

        # mo-basis overlap matrix
        X = orbgen._mo_ao(coef, mat['mo_jy'], mat['mu2comp'], mat['rcut'], range(1,4))

        #fig, ax = plt.subplots(1, 2)
        #im = ax[0].imshow(np.real(np.log(np.abs(X[0]))))
        #fig.colorbar(im, ax=ax[0], location='bottom')

        #im = ax[1].imshow(np.imag(np.log(np.abs(X[0]))))
        #fig.colorbar(im, ax=ax[1], location='bottom')

        #plt.show()

    def test_spillage_0(self):
        orbgen = SpillOpt()

        folder = '/home/zuxin/abacus-community/abacus_orbital_generation/tmp/Si-dimer-2.0/'

        mat = read_orb_mat(folder + 'orb_matrix_rcut6deriv0.dat')
        dmat = read_orb_mat(folder + 'orb_matrix_rcut6deriv1.dat')

        orbgen.add_config(mat, dmat)

        coef = [[np.eye(2, mat['nbes']-1).tolist(), \
                np.eye(2, mat['nbes']-1).tolist(), \
                np.eye(1, mat['nbes']-1).tolist()]]

        ibands = range(5)
        print(orbgen._spillage_0(coef, ibands))



if __name__ == '__main__':
    unittest.main()


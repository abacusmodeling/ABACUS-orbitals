'''


Notes
-----

Several assumptions are made by this script.


'''
from datparse import read_orb_mat
from indexmap import _index_map
from radial import jl_reduce
from scipy.linalg import block_diag
import itertools

class SpillOpt:
    '''
    '''

    def __init__(self):
        '''
        '''
        self.config = []
        self.coef_base = []

        # transformation matrix from the truncated spherical Bessel function
        # to the orthonormal end-smoothed mixed spherical Bessel function.
        self.T = {} # rcut -> [Tl]

    def add_config(self, orb_dat, dorb_dat):
        '''
        '''
        self.config.append((orb_dat, dorb_dat))

        # updates the the table of jl transformation matrix
        self._update_transform_table(orb_dat['rcut'], orb_dat['nbes'], orb_dat['lmax'])
        self._update_transform_table(dorb_dat['rcut'], dorb_dat['nbes'], dorb_dat['lmax'])


    def set_coef_base(self, coef_base):
        '''
        '''
        self.coef_base = coef_base
        

    def opt(self, coef0):
        '''
        '''
        pass


    def _update_transform_table(self, rcut, nbes, lmax):
        '''
        Updates the table of jl transformation matrix.

        The table stores transformation matrices from the truncated spherical
        Bessel function to the orthonormal end-smoothed mixed spherical Bessel
        basis. Given rcut and lmax, the transformation is guaranteed to be
        consistent with respect to different nbes, i.e., the transformation
        matrix for nbes = N is a submatrix of the transformation matrix for
        nbes = M, if N < M.

        The table, self.T, is a dictionary with rcut as the key and a list of
        transformation matrices as the value. The list is indexed by l, which
        is the order of the spherical Bessel function.

        '''
        if rcut not in self.T:
            self.T[rcut] = [jl_reduce(l, nbes, rcut) for l in range(lmax+1)]
        else:
            # If the key already exists, check if the tabulated matrix size
            # is large enough. If not, re-tabulate the the whole list;
            # If yes, append to the existing list.
            _nbes = self.T[rcut][0].shape[0] # tabulated matrix size
            _lmax = len(self.T[rcut])-1 # max tabulated l
            if _nbes < nbes:
                self.T[rcut] = [jl_reduce(l, nbes, rcut) \
                        for l in range(max(lmax, _lmax)+1)]
            else:
                self.T[rcut] += [jl_reduce(l, _nbes, rcut) \
                        for l in range(_lmax+1, lmax+1)]


    def _spillage(self, coef):
        '''
        '''
        pass


    def _gen_q2zeta(self, coef, mu2comp, nbes, rcut):
        '''
        Given an index map "mu2comp" where the number of zeta is always 1,
        this function returns a list of matrices, each of which corresponds
        to the basis transformation matrix from the truncated spherical
        Bessel function to the orthonormal end-smoothed mixed spherical
        Bessel basis.

        '''
        c = [[np.array(coef_l).T for coef_l in coef_t] for coef_t in coef]

        for mu in mu2comp:
            itype, iatom, l, _, m = jy_mu2comp[mu]
            yield jl_reduce(l, nbes, rcut) @ c[itype][l]



    def _basis_overlap(self, coef, jy_jy, jy_mu2comp, rcut):
        '''
        Builds the basis overlap matrix from the given orthonormal
        end-smoothed mixed spherical Bessel coefficients and jy_jy
        overlap matrix.

        Note
        ----

        '''
        nk, nao, nbes = jy_jy.shape[[0, 1, -1]]

        q2zeta = self._gen_q2zeta(coef, jy_mu2comp, nbes, rcut) # generator
        M = block_diag(*q2zeta)

        return np.array([M.T @ jy_jy[ik].transpose((0,2,1,3)).reshape((nao*nbes, nao*nbes)) \
                @ M for ik in range(nk)])











############################################################
#                           Test
############################################################
import unittest

class _TestSpillOpt(unittest.TestCase):

    def test_update_transform_table(self):
        orbgen = SpillOpt()

        rcut = 5.0
        nbes = 30
        lmax = 2

        orbgen._update_transform_table(rcut, nbes, lmax)
        self.assertEqual(len(orbgen.T[rcut]), lmax+1)
        
        # a smaller lmax & nbes would not alter the table
        orbgen._update_transform_table(rcut, nbes, 1)
        self.assertEqual(len(orbgen.T[rcut]), lmax+1)

        orbgen._update_transform_table(rcut, 10, lmax)
        self.assertEqual(orbgen.T[rcut][0].shape[0], nbes)

        orbgen._update_transform_table(rcut, 10, 1)
        self.assertEqual(len(orbgen.T[rcut]), lmax+1)
        self.assertEqual(orbgen.T[rcut][0].shape[0], nbes)

        # a larger lmax/nbes would extend the table
        lmax = 4
        orbgen._update_transform_table(rcut, nbes, lmax)
        self.assertEqual(len(orbgen.T[rcut]), lmax+1)


        nbes = 40
        orbgen._update_transform_table(rcut, nbes, 2)
        self.assertEqual(len(orbgen.T[rcut]), lmax+1)
        self.assertEqual(orbgen.T[rcut][0].shape[0], nbes)

        nbes = 50
        lmax = 7
        orbgen._update_transform_table(rcut, nbes, lmax)
        self.assertEqual(len(orbgen.T[rcut]), lmax+1)
        self.assertEqual(orbgen.T[rcut][0].shape[0], nbes)

    
    def test_add_config(self):
        orbgen = SpillOpt()

        folder = '/home/zuxin/abacus-community/abacus_orbital_generation/tmp/Si-dimer-2.0/'

        mat = read_orb_mat(folder + 'orb_matrix_rcut6deriv0.dat')
        dmat = read_orb_mat(folder + 'orb_matrix_rcut6deriv1.dat')

        orbgen.add_config(mat, dmat)

        self.assertEqual( len(orbgen.config), 1 )
        self.assertEqual( len(orbgen.T[mat['rcut']]), mat['lmax']+1 )

        mat = read_orb_mat(folder + 'orb_matrix_rcut7deriv0.dat')
        dmat = read_orb_mat(folder + 'orb_matrix_rcut7deriv1.dat')

        orbgen.add_config(mat, dmat)

        self.assertEqual( len(orbgen.config), 2 )
        self.assertEqual( len(orbgen.T[mat['rcut']]), mat['lmax']+1 )



if __name__ == '__main__':
    unittest.main()


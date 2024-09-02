from SIAB.spillage.index import index_map

import numpy as np
from scipy.linalg import block_diag

def jy2ao(coef, natom, nbes):
    '''
    Basis transformation matrix from a spherical wave basis to a pseudo-
    atomic orbital basis.

    Assuming a spherical wave basis ([raw/normalized/reduced spherical
    Bessel radial function] x [spherical harmonics]) arranged in the
    lexicographic order of (itype, iatom, l, mm, q) where mm=2*abs(m)-(m>0)
    and q is the index for radial functions, this function constructs the
    transformation matrix from the spherical wave basis to the pseudo-atomic
    orbital basis specified by coef and arranged in the lexicographic order
    of (itype, iatom, l, mm, zeta). The transformation matrix is block-
    diagonal, with each block corresponding to a specific q -> zeta.

    Parameters
    ----------
        coef : nested list
            The coefficients of pseudo-atomic orbital basis orbitals
            in terms of the spherical wave basis. coef[itype][l][zeta]
            gives a list of spherical wave coefficients that specifies
            an orbital.
            Note that len(coef[itype][l][zeta]) must not be larger than
            nbes[itype][l]; coef[itype][l][zeta] will be padded with zeros
            if len(coef[itype][l][zeta]) < nbes[itype][l].
        natom : list of int
            Number of atoms for each atom type.
        nbes : list of int / list of list of int
            Number of spherical wave radial functions.
            If a list, nbes[l] specifies the number for angular momentum l
            and is assumed to be the same for all atomic types.
            If a nested list, nbes[itype][l] specifies the number for angular
            momentum l of atomic type `itype`.

    '''
    # if a plain list is given, convert to a list of list of int
    # such that nbes[itype][l] gives the corresponding number.
    if isinstance(nbes[0], int):
        nbes = [nbes] * len(natom)
    
    assert len(natom) == len(coef) == len(nbes) # should all equal ntype
    assert all(len(nbes_t) >= len(coef_t)
               for nbes_t, coef_t in zip(nbes, coef))

    def _gen_q2zeta(coef, lin2comp, nbes):
        for comp in lin2comp:
            itype, _, l, _ = comp
            if l >= len(coef[itype]) or len(coef[itype][l]) == 0:
                # The generator should yield a zero matrix with the
                # appropriate size when no coefficient is provided.
                yield np.zeros((nbes[itype][l], 0))
            else:
                # zero-padding coef[itype][l] to the specified size
                C = np.zeros((nbes[itype][l], len(coef[itype][l])))
                C[:len(coef[itype][l][0])] = np.array(coef[itype][l]).T
                yield C

    lmax = [len(coef_t) - 1 for coef_t in coef]
    lin2comp = index_map(natom, lmax=lmax)[1]
    return block_diag(*_gen_q2zeta(coef, lin2comp, nbes))


############################################################
#                           Test
############################################################
import unittest

class _TestBasisTrans(unittest.TestCase):

    def test_jy2ao_nbes1(self):
        '''
        Test the case where nbes is a list of int.

        '''
        nbes = [7, 7, 6]

        nzeta = [[3, 1, 4], [0, 5], [9]] # nzeta[itype][l]
        lmax = [len(nzt) - 1 for nzt in nzeta]
        coef = [[np.random.randn(nzeta_tl, nbes[l]).tolist()
                 for l, nzeta_tl in enumerate(nzeta_t)]
                for nzeta_t in nzeta]

        natom = [1, 2, 3]
        M = jy2ao(coef, natom, nbes)

        irow = 0
        icol = 0
        for (itype, iatom, l, m) in index_map(natom, lmax=lmax)[1]:
            nz = nzeta[itype][l]
            self.assertTrue(np.allclose(
                M[irow:irow+nbes[l], icol:icol+nz],
                np.array(coef[itype][l]).T
            ))
            irow += nbes[l]
            icol += nz


    def test_jy2ao_nbes2(self):
        '''
        Test the case where nbes is a list of list of int.

        '''
        nbes = [[10, 9, 8], [7, 6], [10]]

        nzeta = [[3, 1, 4], [0, 5], [9]] # nzeta[itype][l]
        lmax = [len(nzt) - 1 for nzt in nzeta]
        coef = [[np.random.randn(nzeta_tl, nbes[it][l]).tolist()
                 for l, nzeta_tl in enumerate(nzeta_t)]
                for it, nzeta_t in enumerate(nzeta)]

        natom = [1, 2, 3]
        M = jy2ao(coef, natom, nbes)

        irow = 0
        icol = 0
        for (itype, iatom, l, m) in index_map(natom, lmax=lmax)[1]:
            nz = nzeta[itype][l]
            self.assertTrue(np.allclose(
                M[irow:irow+nbes[itype][l], icol:icol+nz],
                np.array(coef[itype][l]).T
            ))
            irow += nbes[itype][l]
            icol += nz


if __name__ == '__main__':
    unittest.main()


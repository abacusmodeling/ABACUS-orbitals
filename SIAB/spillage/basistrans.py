from SIAB.spillage.index import index_map, _nao

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
            in terms of the spherical wave basis.
            coef[itype][l][zeta][q] -> float
        natom : list of int
            Number of atoms for each atom type.
        nbes : list of list of int
            Number of spherical wave radial functions for a given type
            and angular momentum l.
            nbes[itype][l] -> int

    Note
    ----
    len(coef) must agree with that of natom and nbes, which is the number
    of atom types.
    len(coef[itype]), however, can be less than that of nbes[itype] (i.e.,
    omitting a few largest l's).
    There is no restriction on the length of coef[itype][l], so the
    resulting pseudo-atomic orbital basis may have linear-dependence,
    if len(coef[itype][l]) > nbes[itype][l] occurs.
    len(coef[itype][l][zeta]) can be equal or less than nbes[itype][l].
    If it is less, the remaining elements are assumed to be zero.    

    '''
    # some sanity checks
    # 1. the length of natom, nbes & coef should all agree (ntype)
    assert len(natom) == len(nbes) == len(coef)

    # 2. len(coef[itype]) (number of l) should not exceed len(nbes[itype]).
    assert all(len(coef_t) <= len(nbes_t)
               for nbes_t, coef_t in zip(nbes, coef))

    # 3. len(coef[itype][l][zeta]) should not exceed nbes[itype][l].
    assert all(all(all(len(coef_tlz) <= nbes_tl for coef_tlz in coef_tl)
                   for nbes_tl, coef_tl in zip(nbes_t, coef_t))
               for nbes_t, coef_t in zip(nbes, coef))

    def _gen_q2zeta(coef, lin2comp, nbes):
        for comp in lin2comp:
            itype, _, l, _ = comp
            if l >= len(coef[itype]) or len(coef[itype][l]) == 0:
                # The generator should yield a zero matrix with the
                # appropriate size when no coefficient is provided.
                yield np.zeros((nbes[itype][l], 0))
            else:
                # zero-padding the coefficient to nbes[itype][l]
                C = np.zeros((nbes[itype][l], len(coef[itype][l])))
                C[:len(coef[itype][l][0])] = np.array(coef[itype][l]).T
                yield C

    lmax = [len(nbes_t) - 1 for nbes_t in nbes]
    lin2comp = index_map(natom, lmax=lmax)[1]
    return block_diag(*_gen_q2zeta(coef, lin2comp, nbes))


############################################################
#                           Test
############################################################
import unittest

class _TestBasisTrans(unittest.TestCase):

    def test_jy2ao_nbes(self):
        nbes = [[11, 10, 9, 8], [7, 6], [5], [4, 3], [10]]

        nzeta = [[3, 0, 4], [0, 5], [], [2, 0], [9]] # nzeta[itype][l]
        coef = [[np.random.randn(nzeta_tl, nbes[it][l]).tolist()
                 for l, nzeta_tl in enumerate(nzeta_t)]
                for it, nzeta_t in enumerate(nzeta)]

        natom = [1, 2, 3, 4, 5]
        M = jy2ao(coef, natom, nbes)

        nrow = _nao(natom, nbes)
        ncol = _nao(natom, nzeta)
        self.assertTrue(M.shape == (nrow, ncol))

        irow = 0
        icol = 0
        lmax = [len(nbes_t) - 1 for nbes_t in nbes]
        for (itype, iatom, l, m) in index_map(natom, lmax=lmax)[1]:
            if l < len(nzeta[itype]):
                nz = nzeta[itype][l]
                self.assertTrue(np.allclose(
                    M[irow:irow+nbes[itype][l], icol:icol+nz],
                    np.array(coef[itype][l]).T
                ))
                icol += nz
            irow += nbes[itype][l]


if __name__ == '__main__':
    unittest.main()


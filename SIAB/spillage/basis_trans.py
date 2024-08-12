from SIAB.spillage.index import index_map

import numpy as np

def jy2ao(coef, lin2comp, nbes, rcut):
    '''
    Basis transformation matrix from a spherical wave basis to a pseudo-
    atomic orbital basis.

    Assuming a spherical wave basis ([raw/normalized/reduced spherical
    Bessel radial function] x [spherical harmonics]) arranged in the
    lexicographic order of (itype, iatom, l, mm, q) where mm=2*abs(m)-(m>0)
    and q is the index for radial functions, this function constructs the
    transformation matrix from the spherical wave basis to the pseudo-atomic
    orbital basis arranged in the lexicographic order of (itype, iatom, l,
    mm, zeta). The entire transformation matrix is block-diagonal, with
    each block corresponding to a specific q -> zeta.

    Parameters
    ----------
        coef : nested list
            The coefficients of pseudo-atomic orbital basis orbitals
            in terms of the spherical wave basis. coef[itype][l][zeta]
            gives a list of spherical wave coefficients that specifies
            an orbital.
            Note that the length of this coefficient list must not be
            larger than its corresponding nbes; it will be padded with
            zeros if its length is smaller.
        lin2comp : list
            linear-to-composite index map without the radial index q.

                    mu -> (itype, iatom, l, m).

        nbes : list of int or int
            nbes[l] specifies the number of radial functions for angular
            momemtum l. If an integer, the same number is used for all l.
        rcut : float
            Cutoff radius.

    '''
    from scipy.linalg import block_diag
    lmax = max(comp[2] for comp in lin2comp)
    nbes = [nbes] * (lmax + 1) if isinstance(nbes, int) else nbes

    def _gen_q2zeta(coef, lin2comp, nbes, rcut):
        for comp in lin2comp:
            itype, _, l, _ = comp
            if l >= len(coef[itype]) or len(coef[itype][l]) == 0:
                # The generator should yield a zero matrix with the
                # appropriate size when no coefficient is provided.
                yield np.zeros((nbes[l], 0))
            else:
                # zero-padding to the appropriate size
                C = np.zeros((nbes[l], len(coef[itype][l])))
                C[:len(coef[itype][l][0])] = np.array(coef[itype][l]).T
                yield C

    return block_diag(*_gen_q2zeta(coef, lin2comp, nbes, rcut))


############################################################
#                           Test
############################################################
import unittest

class _TestBasisTrans(unittest.TestCase):

    def coefgen(self, nzeta, nbes):
        '''
        Generates some random pseudo-atomic orbital coefficients.

        Parameters
        ----------
            nzeta : nested list of int
                nzeta[itype][l] gives the number of zeta.
            nbes : list of int or int
                nbes[l] specifies the number of radial functions for angular
                momemtum l. If an integer, the same number is used for all l.

        Returns
        -------
            A nested list of float.
            coef[itype][l][zeta] gives a list of spherical wave coefficients
            that specifies an orbital.

        '''
        nbes = [nbes] * (max(len(nzt) for nzt in nzeta) + 1) \
                if isinstance(nbes, int) else nbes
        return [[np.random.randn(nzeta_tl, nbes[l]).tolist()
                 for l, nzeta_tl in enumerate(nzeta_t)]
                for it, nzeta_t in enumerate(nzeta)]


    def test_jy2ao(self):

        # generate some random coefficients
        nbes = [5, 6, 7]
        nzeta = [[2, 1, 3], [2, 2], [3]] # nzeta[itype][l]
        coef = self.coefgen(nzeta, nbes)

        natom = [1, 2, 3]
        lmax = [2, 1, 0]
        lin2comp = index_map(natom, lmax)[1]
        rcut = 6.0
        M = jy2ao(coef, lin2comp, nbes, rcut)

        irow = 0
        icol = 0
        for (itype, iatom, l, m) in lin2comp:
            nz = nzeta[itype][l]
            self.assertTrue(np.allclose(
                M[irow:irow+nbes[l], icol:icol+nz],
                np.array(coef[itype][l]).T
            ))
            irow += nbes[l]
            icol += nz


if __name__ == '__main__':
    unittest.main()


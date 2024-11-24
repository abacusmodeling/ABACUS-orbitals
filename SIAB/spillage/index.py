def _nao(natom, nzeta=None, lmax=None):
    '''
    Total number of orbitals.

    Parameters
    ----------
        natom : list of int
            Number of atoms for each type.
        nzeta : list of list of int
            nzeta[i][l] specifies the number of zeta orbitals of the
            angular momentum l of type i.
            If not None, len(nzeta) must equal len(natom) and lmax must
            be None (which will be deduced from nzeta).
            If None, nzeta is assumed to be 1 for all atom types and l;
            In this case lmax must be specified and len(lmax) must equal
            len(natom).
        lmax : list of int
            lmax[i] specifies the maximum angular momentum of type i.
            This argument is only used if nzeta is None.

    '''
    assert (nzeta is None) != (lmax is None)

    if nzeta is None:
        return sum(sum(2*l+1 for l in range(lmax[itype]+1)) * nat
                   for itype, nat in enumerate(natom))
    else:
        return sum(sum((2*l+1) * nztl for l, nztl in enumerate(nzeta[itype]))
                   * nat for itype, nat in enumerate(natom))


def _lin2comp(natom, nzeta=None, lmax=None):
    '''
    Linearized-to-composite index map.

    An orbital is indexed by its atomic species (type), atomic index within
    species, angular momentum numbers l & m, and possibly a zeta number.
    Suppose there are a total of N orbitals, each orbital can also be
    assigned a unique index mu in [0, N-1].

    This function returns an index map from linearized indices to composite
    indices (itype, iatom, l[, zeta], m). The composite indices are ordered
    lexicographically in term of (itype, iatom, l(, zeta), mm) where
    mm = 2*abs(m)-(m>0) (i.e. m is ordered as 0, 1, -1, 2, -2, ..., l, -l),
    in accordance with the ABACUS convention.

    Parameters
    ----------
        natom : list of int
            Number of atoms for each type.
        nzeta : list of list of int
            nzeta[i][l] specifies the number of zeta orbitals of the
            angular momentum l of type i.
            If not None, len(nzeta) must equal len(natom) and lmax must
            be None (which will be deduced from nzeta).
            If None, nzeta is assumed to be 1 for all atom types and l;
            composite indices will not include zeta. In this case lmax
            must be specified and len(lmax) must equal len(natom).
        lmax : list of int
            lmax[i] specifies the maximum angular momentum of type i.
            This argument is only used if nzeta is None.

    Returns
    -------
        lin2comp : list
            lin2comp[i] gives the composite index of the orbital with
            linearized index i. If nzeta is None, composite indices
            have the form (itype, iatom, l, m); otherwise they have the
            form (itype, iatom, l, zeta, m).


    '''
    assert (nzeta is None) != (lmax is None)

    if nzeta is None:
        assert len(natom) == len(lmax)
        return [(itype, iatom, l,
                 -mm // 2 if mm % 2 == 0 else (mm + 1) // 2)
                for itype, nat in enumerate(natom)
                for iatom in range(nat)
                for l in range(lmax[itype]+1)
                for mm in range(0, 2*l+1)
                ]
    else:
        return [(itype, iatom, l, zeta,
                 -mm // 2 if mm % 2 == 0 else (mm + 1) // 2)
                for itype, nat in enumerate(natom)
                for iatom in range(nat)
                for l, nztl in enumerate(nzeta[itype])
                for zeta in range(nztl)
                for mm in range(0, 2*l+1)
                ]


def perm_zeta_m(lin2comp):
    '''
    Given a list of composite indices (itype, iatom, l, zeta, m) following
    the ABACUS order, this function returns a permutation `p` such that
    lin2comp[p] becomes a list with the relative lexicographic order of
    zeta & m reversed.

    '''
    # preserve the original intra-m order (0, 1, -1, 2, -2, ..., l, -l),
    comp = [(it, ia, l, 2*abs(m)-(m>0), q) for it, ia, l, q, m in lin2comp]
    return sorted(range(len(comp)), key=lambda i: comp[i])

def _coef_flatten(nzeta, nbes):
    '''
    Generate the index mapping in two direction, 
    [itype, l, zeta, q] <-> mu
    
    Parameters
    ----------
    nzeta : list[list[int]]
        the number of zeta orbitals of the angular momentum l of type i
    nbes : list[list[int]]
        the number of spherical wave radial functions for a given type
        and angular momentum l of type i
    
    Returns
    -------
    backward : list[tuple[int, int, int, int]]
        the index mapping from [itype, l, zeta, q] to mu
    '''
    index_, range_ = [], []
    i = 0
    for it, (nzeta_t, nbes_t) in enumerate(zip(nzeta, nbes)):
        for l, nz in enumerate(nzeta_t):
            index_.append((it, l))
            range_.append(range(i, i+nz*nbes_t[l]))
            i += nz*nbes_t[l]
    
    return index_, range_

############################################################
#                           Test
############################################################
import unittest

class TestIndex(unittest.TestCase):

    def test_nao(self):
        natom = [7]
        nzeta = [[2,3,4]]
        self.assertEqual(_nao(natom, nzeta=nzeta), 7*(2*1 + 3*3 + 4*5))

        natom = [2, 1, 3]
        lmax = [3, 0, 4]
        self.assertEqual(_nao(natom, lmax=lmax), 2*16 + 1*1 + 3*25)

        natom = [7, 8, 9]
        nzeta = [[2,3,1,1], [4], [1, 2, 2, 1, 3]]
        self.assertEqual(_nao(natom, nzeta=nzeta),
                         7*(2*1 + 3*3 + 1*5 + 1*7) +
                         8*(4*1) +
                         9*(1*1 + 2*3 + 2*5 + 1*7 + 3*9))


    def test_lin2comp(self):
        natom = [2, 1, 3]
        lmax = [1, 2, 4]
        nzeta = [[2,3], [1,0,1], [1, 2, 2, 1, 3]]
        lin2comp = _lin2comp(natom, nzeta=nzeta)

        # check the total number of orbitals
        nao = _nao(natom, nzeta=nzeta)
        self.assertEqual(len(lin2comp), nao)

        # check the first and the last
        self.assertEqual(lin2comp[0], (0, 0, 0, 0, 0))
        self.assertEqual(lin2comp[nao-1],
                         (len(natom)-1, natom[-1]-1, lmax[-1],
                          nzeta[-1][-1]-1, -lmax[-1]))

        # repeat the above checks for nzeta = None
        lin2comp = _lin2comp(natom, lmax=lmax)

        nao = _nao(natom, lmax=lmax)
        self.assertEqual(len(lin2comp), nao)

        self.assertEqual(lin2comp[0], (0, 0, 0, 0))
        self.assertEqual(lin2comp[nao-1],
                         (len(natom)-1, natom[-1]-1, lmax[-1], -lmax[-1]))


    def test_perm_zeta_m(self):
        natom = [2, 1, 3]
        lmax = [1, 2, 4]
        nzeta = [[2,3], [1,0,1], [1, 2, 2, 1, 3]]
        lin2comp = _lin2comp(natom, nzeta=nzeta)

        p = perm_zeta_m(lin2comp)
        comp = [lin2comp[i] for i in p]

        # verify that comp does following a lexicographic order of
        # itype-iatom-l-mm-zeta where mm = 2*abs(m)-(m>0)
        comp2 = [(it, ia, l, 2*abs(m)-(m>0), z) for it, ia, l, z, m in comp]
        self.assertEqual( comp2, sorted(comp2) )

    def test_coef_flatten(self):
        # first generate a coef indexed by [it][l][z][q] -> float
        nbes = [[17, 16, 16]]
        nzeta = [[2, 2, 1]]
        index_, range_ = _coef_flatten(nzeta, nbes)
        self.assertEqual(range_[0].start, 0)
        self.assertEqual(range_[0].stop, 2*17)
        self.assertEqual(range_[1].start, 2*17)
        self.assertEqual(range_[1].stop, 2*17+2*16)
        self.assertEqual(range_[2].start, 2*17+2*16)
        self.assertEqual(range_[2].stop, 2*17+2*16+1*16)
        self.assertEqual(len(range_), 3)
        self.assertEqual(len(index_), 3)
        self.assertEqual(index_[0], (0, 0))
        self.assertEqual(index_[1], (0, 1))
        self.assertEqual(index_[2], (0, 2))

if __name__ == '__main__':
    unittest.main()


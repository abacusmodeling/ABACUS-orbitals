def index_map(natom, lmax, nzeta=None):
    '''
    Bijective map between composite and linearized indices.

    An orbital is labeled by its atomic species, atomic index within
    species, angular momentum numbers l & m, and possibly a zeta number.
    Suppose there are a total of N orbitals, each orbital can also be
    assigned a unique index mu in [0, N-1].

    This function returns a bijective map (dict + list) between composite
    indices (itype, iatom, l[, zeta], m) and linearized indices following
    the ABACUS convention, in which the composite indices are ordered
    lexicographically in term of itype-iatom-l[-zeta]-mm where
    mm = 2*abs(m)-(m>0) (i.e. m is ordered as 0, 1, -1, 2, -2, ..., l, -l)

    Parameters
    ----------
        natom : list of int
            Number of atoms for each type.
        lmax : list of int
            lmax[i] specifies the maximum angular momentum of type i.
            len(lmax) must be equal to len(natom).
        nzeta : list of list of int
            nzeta[i][l] specifies the number of zeta orbitals of the
            angular momentum l of type i.
            If None, nzeta is assumed to be 1 in all cases and the
            composite index will not include zeta.
            If not None, len(nzeta) must be equal to len(natom), and
            len(nzeta[i]) must be equal to lmax[i]+1.

    Returns
    -------
        comp2lin : dict
            A dict with values being the linearized indices. The keys
            are (itype, iatom, l, m) if nzeta is None, or (itype, iatom,
            l, zeta, m) if nzeta is not None.
        lin2comp : list
            lin2comp[i] gives the composite index of the orbital with
            linearized index i.

    '''
    assert len(natom) == len(lmax)

    if nzeta is None:
        lin2comp = [(itype, iatom, l,
                     -mm // 2 if mm % 2 == 0 else (mm + 1) // 2)
                    for itype, nat in enumerate(natom)
                    for iatom in range(nat)
                    for l in range(lmax[itype]+1)
                    for mm in range(0, 2*l+1)
                    ]
    else:
        assert lmax == [len(nzt)-1 for nzt in nzeta]
        lin2comp = [(itype, iatom, l, zeta,
                     -mm // 2 if mm % 2 == 0 else (mm + 1) // 2)
                    for itype, nat in enumerate(natom)
                    for iatom in range(nat)
                    for l in range(lmax[itype]+1)
                    for zeta in range(nzeta[itype][l])
                    for mm in range(0, 2*l+1)
                    ]

    comp2lin = {comp: i for i, comp in enumerate(lin2comp)}

    return comp2lin, lin2comp


def perm_zeta_m(lin2comp):
    '''
    Given a list of composite indices (itype, iatom, l, zeta, m) following
    the ABACUS order, this function returns a permutation `p` such that
    lin2comp[p] becomes a list with the relative order of zeta & m reversed.

    '''
    # preserve the original intra-m order (0, 1, -1, 2, -2, ..., l, -l),
    comp = [(it, ia, l, 2*abs(m)-(m>0), q) for it, ia, l, q, m in lin2comp]
    return sorted(range(len(comp)), key=lambda i: comp[i])


############################################################
#                           Test
############################################################
import unittest

class _TestIndexMap(unittest.TestCase):

    def test_index_map(self):
        natom = [2, 1, 3]
        lmax = [1, 2, 4]
        nzeta = [[2,3], [1,0,1], [1, 2, 2, 1, 3]]
        comp2lin, lin2comp = index_map(natom, lmax, nzeta)

        # check the total number of orbitals
        nao = sum(sum((2*l+1) * nzeta[itype][l]
                      for l in range(lmax[itype]+1))
                  * nat for itype, nat in enumerate(natom))
        self.assertEqual( len(lin2comp), nao )

        # check the first and the last
        self.assertEqual(lin2comp[0], (0, 0, 0, 0, 0))
        self.assertEqual(lin2comp[nao-1],
                         (len(natom)-1, natom[-1]-1, lmax[-1],
                          nzeta[-1][-1]-1, -lmax[-1]))

        # check bijectivity
        for mu in range(nao):
            self.assertEqual(comp2lin[lin2comp[mu]], mu)


        # repeat the above checks for nzeta = None
        comp2lin, lin2comp = index_map(natom, lmax)

        nao = sum(sum((2*l+1) for l in range(lmax[itype]+1))
                  * nat for itype, nat in enumerate(natom))
        self.assertEqual( len(lin2comp), nao )

        self.assertEqual(lin2comp[0], (0, 0, 0, 0))
        self.assertEqual(lin2comp[nao-1],
                         (len(natom)-1, natom[-1]-1, lmax[-1], -lmax[-1]))

        for mu in range(nao):
            self.assertEqual(comp2lin[lin2comp[mu]], mu)


    def test_perm_zeta_m(self):
        natom = [2, 1, 3]
        lmax = [1, 2, 4]
        nzeta = [[2,3], [1,0,1], [1, 2, 2, 1, 3]]
        _, lin2comp = index_map(natom, lmax, nzeta)

        p = perm_zeta_m(lin2comp)
        comp = [lin2comp[i] for i in p]

        # verify that comp does following a lexicographic order of
        # itype-iatom-l-mm-zeta where mm = 2*abs(m)-(m>0)
        comp2 = [(it, ia, l, 2*abs(m)-(m>0), z) for it, ia, l, z, m in comp]
        self.assertEqual( comp2, sorted(comp2) )


if __name__ == '__main__':
    unittest.main()


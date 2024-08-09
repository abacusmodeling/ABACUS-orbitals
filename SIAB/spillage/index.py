def abacus_map(ntype, natom, lmax, nzeta=None):
    '''
    Bijective map between composite indices (itype, iatom, l, zeta, m)
    and linearized indices following ABACUS convention.

    An orbital is labeled by its atomic species, atomic index, angular
    momentum quantum numbers l & m, and a zeta number. Suppose there are
    a total of N orbitals, each orbital can also be assigned a unique index
    mu \in [0, N-1].

    This function returns a pair of bijective maps between composite indices
    (itype, iatom, l, zeta, m) and linearized indices following the ABACUS
    convention. The composite indices are mainly arranged lexicographically
    in term of itype, iatom, l, zeta, and m, except that m is arranged as

                    0, 1, -1, 2, -2, ..., l, -l

    Parameters
    ----------
        ntype : int
            Number of atomic species (types)
        natom : list of int
            Number of atoms for each type.
        lmax : list of int
            lmax[i] specifies the maximum angular momentum of type i.
        nzeta : list of list of int
            nzeta[i][l] specifies the number of zeta orbitals of the
            angular momentum l of type i.
            If None, nzeta is assumed to be 1 in all cases.

    Returns
    -------
        comp2lin : dict
            A dict with keys being composite indices (itype, iatom, l, zeta, m)
            and values being the linearized indices.
        lin2comp : list
            lin2comp[i] gives the composite index of the orbital of linearized
            index i.

    '''
    if nzeta is None:
        nzeta = [[1]*(lmax[itype]+1) for itype in range(ntype)]

    assert len(natom) == ntype
    assert len(lmax) == ntype
    assert lmax == [len(nzeta[itype])-1 for itype in range(ntype)]

    lin2comp = [(itype, iatom, l, zeta, -mm // 2 if mm % 2 == 0 else (mm + 1) // 2)
                for itype in range(ntype)
                for iatom in range(natom[itype])
                for l in range(lmax[itype]+1)
                for zeta in range(nzeta[itype][l])
                for mm in range(0, 2*l+1)
                ]

    comp2lin = {comp: i for i, comp in enumerate(lin2comp)}

    return comp2lin, lin2comp


def perm_zeta_m(lin2comp):
    '''
    Given a list of abacus-ordered composite indices (itype, iatom, l, zeta, m),
    this function returns a permutation `p` such that lin2comp[p] becomes a list
    with the relative order of zeta and m reversed.

    '''
    # preserve the original intra-m order (0, 1, -1, 2, -2, ..., l, -l),
    comp = [(it, ia, l, 2*abs(m)-(m>0), q) for it, ia, l, q, m in lin2comp]
    return sorted(range(len(comp)), key=lambda i: comp[i])


############################################################
#                           Test
############################################################
import unittest

class _TestIndexMap(unittest.TestCase):

    def test_abacus_map(self):
        ntype = 3
        natom = [2, 1, 3]
        lmax = [1, 2, 4]
        nzeta = [[2,3], [1,0,1], [1, 2, 2, 1, 3]]
        comp2lin, lin2comp = abacus_map(ntype, natom, lmax, nzeta)

        # check the total number of orbitals
        nao = sum(sum( (2*l+1) * nzeta[itype][l] for l in range(lmax[itype]+1) ) \
                * natom[itype] for itype in range(ntype))
        self.assertEqual( len(lin2comp), nao )

        # check the first and the last
        self.assertEqual( lin2comp[0], (0, 0, 0, 0, 0) )
        self.assertEqual( lin2comp[nao-1], \
                (ntype-1, natom[-1]-1, lmax[-1], nzeta[-1][-1]-1, -lmax[-1]) )

        # check bijectivity
        for mu in range(nao):
            self.assertEqual( comp2lin[lin2comp[mu]], mu )


    def test_perm_zeta_m(self):
        ntype = 3
        natom = [2, 1, 3]
        lmax = [1, 2, 4]
        nzeta = [[2,3], [1,0,1], [1, 2, 2, 1, 3]]
        _, lin2comp = abacus_map(ntype, natom, lmax, nzeta)

        p = perm_zeta_m(lin2comp)
        comp = [lin2comp[i] for i in p]

        # verify that comp does following a lexicographic order of
        # itype-iatom-l-mm-zeta where mm = 2*abs(m)-(m>0)
        comp2 = [(it, ia, l, 2*abs(m)-(m>0), z) for it, ia, l, z, m in comp]
        self.assertEqual( comp2, sorted(comp2) )


if __name__ == '__main__':
    unittest.main()


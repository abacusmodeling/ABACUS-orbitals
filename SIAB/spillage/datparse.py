import re
import numpy as np
import itertools
from scipy.sparse import csr_matrix

from SIAB.spillage.jlzeros import JLZEROS
from scipy.special import spherical_jn
from SIAB.spillage.index import index_map


def read_orb_mat(fpath):
    '''
    Reads an "orb_matrix" data file.

    In spillage-based orbital generation, ABACUS will generate some
    "orb_matrix" data files which contain some system parameters as
    well as various overlaps. This function parses such a file and
    returns a dictionary containing its content.

    Parameters
    ----------
        fpath : str
            The file path.

    Returns
    -------
        A dictionary containing the following key-value pairs:

        ntype : int
            Number of atom types.
        natom : list of int
            Number of atoms for each atom type.
        ecutwfc : float
            Energy cutoff for wave functions.
        ecutjlq : float
            Energy cutoff for spherical Bessel wave numbers and "kmesh"
            (used in Simpson-based spherical Bessel transforms).
            In the present code, ecutjlq == ecutwfc.
        rcut : float
            Cutoff radius for spherical Bessel functions.
        lmax : list of int
            Maximum angular momentum of each type.
        nbands : int
            Number of bands.
        nbes : int
            Number of spherical Bessel wave numbers.
        nk : int
            Number of k-points.
            Should be 1 or 2 (nspin=2)
        kpt : np.ndarray
            k-points.
        wk : np.ndarray
            k-point weights.
        mo_jy : np.ndarray
            Overlap between MOs and jYs.
            Shape: (nk, nbands, nao*nbes)
        jy_jy : np.ndarray
            Overlap between jYs.
            Shape: (nk, nao*nbes, nao*nbes)
            Note: the original jy_jy data assumed a shape of
            (nk, nao, nao, nbes, nbes), which is permuted and
            reshaped for convenience.
        mo_mo : np.ndarray
            Overlap between MOs.
            Shape: (nk, nbands)
        comp2lin, lin2comp : dict
            Bijective index map between the composite and the
            lineaerized index.
            comp2lin: (itype, iatom, l, zeta, m) -> mu
            lin2comp: mu -> (itype, iatom, l, zeta, m)
            NOTE: zeta is always 0 in the present code.

    Notes
    -----
    "orb_matrix" files might contain overlaps between orbital gradients
    instead of orbitals themselves. Such files have exactly the same
    structure and can be parsed by this function as well. However,
    there's no way to distinguish between the two types of files by
    their format; user should distinguish them by their file names.
    (Although MO-MO overlaps in theory can be used to distinguish them,
    it's not a good practice.)

    '''
    with open(fpath, 'r') as f:
        data = f.read()
        data = data.replace('\n', ' ').split()

    ntype = int(data[data.index('ntype') - 1])
    natom = [int(data[i-1]) \
            for i, label in enumerate(data[:data.index('ecutwfc')]) \
            if label == 'na']

    # ecutwfc of pw calculation
    ecutwfc = float(data[data.index('ecutwfc') - 1])

    # ecut for wave numbers & "kmesh"
    # (used in Simpson-based spherical Bessel transforms)
    # in the present code, ecutjlq = ecutwfc
    ecutjlq = float(data[data.index('ecutwfc_jlq') - 1])

    # cutoff radius of spherical Bessel functions
    rcut = float(data[data.index('rcut_Jlq') - 1])

    lmax = int(data[data.index('lmax') - 1])
    nk = int(data[data.index('nks') - 1])
    nbands = int(data[data.index('nbands') - 1])
    nbes = int(data[data.index('ne') - 1])

    # NOTE In PW calculations, lmax is always the same for all element types,
    # which is the lmax read above. (Will it be different in the future?)
    lmax = [lmax] * ntype

    wk_start = data.index('<WEIGHT_OF_KPOINTS>') + 1
    wk_end = data.index('</WEIGHT_OF_KPOINTS>')
    kinfo = np.array(data[wk_start:wk_end], dtype=float).reshape(nk, 4)
    kpt = kinfo[:, 0:3]
    wk = kinfo[:, 3]

    ####################################################################
    #   bijective map between the composite and linearized index
    ####################################################################
    comp2lin, lin2comp = index_map(natom, lmax)
    nao = len(comp2lin)

    ####################################################################
    #                           MO-jY overlap
    ####################################################################
    mo_jy_start= data.index('<OVERLAP_Q>') + 1
    mo_jy_end = data.index('</OVERLAP_Q>')
    mo_jy = np.array(data[mo_jy_start:mo_jy_end], dtype=float) \
            .view(dtype=complex) \
            .reshape((nk, nbands, nao*nbes)) \
            .conj() # abacus outputs <jy|mo>, so a conjugate is needed


    ####################################################################
    #                           jY-jY overlap
    ####################################################################
    jy_jy_start= data.index('<OVERLAP_Sq>') + 1
    jy_jy_end = data.index('</OVERLAP_Sq>')
    jy_jy = np.array(data[jy_jy_start:jy_jy_end], dtype=float) \
            .view(dtype=complex) \
            .reshape((nk, nao, nao, nbes, nbes))

    # overlap between jY should be real
    assert np.linalg.norm(np.imag(jy_jy.reshape(-1)), np.inf) < 1e-12
    jy_jy = np.real(jy_jy)

    # permute jy_jy from (nk, nao, nao, nbes, nbes) to
    # (nk, nao, nbes, nao, nbes) for convenience later.
    jy_jy = jy_jy \
            .transpose((0, 1, 3, 2, 4)) \
            .reshape((nk, nao*nbes, nao*nbes))

    ####################################################################
    #                           MO-MO overlap
    ####################################################################
    # should be all 1
    mo_mo_start= data.index('<OVERLAP_V>') + 1
    mo_mo_end = data.index('</OVERLAP_V>')
    mo_mo = np.array(data[mo_mo_start:mo_mo_end], dtype=float)

    assert len(mo_mo) == nbands * nk
    mo_mo = mo_mo.reshape((nk, nbands))

    return {'ntype': ntype, 'natom': natom, 'ecutwfc': ecutwfc,
            'ecutjlq': ecutjlq, 'rcut': rcut, 'lmax': lmax, 'nk': nk,
            'nbands': nbands, 'nbes': nbes, 'kpt': kpt, 'wk': wk,
            'jy_jy': jy_jy, 'mo_jy': mo_jy, 'mo_mo': mo_mo,
            'comp2lin': comp2lin, 'lin2comp': lin2comp}


def _assert_consistency(dat1, dat2):
    '''
    Check if two dat files corresponds to the same system.

    '''
    assert dat1['lin2comp'] == dat2['lin2comp'] and \
            dat1['rcut'] == dat2['rcut'] and \
            np.all(dat1['wk'] == dat2['wk']) and \
            np.all(dat1['kpt'] == dat2['kpt'])


def read_wfc_lcao_txt(fname):
    '''
    Read a wave function coefficient file in text format.

    The coefficient file of a multiple-k calculation generated by ABACUS
    with "out_wfc_lcao 1" looks as follows:

    <<<<<<< starts here (taken from test/integrate/212_NO_wfc_out, precision may vary)
    1 (index of k points)
    0 0 0
    4 (number of bands)
    10 (number of orbitals)
    1 (band)
    -7.4829e-01 (Ry)
    2.5000e-01 (Occupations)
    -5.3725e-01 0.0000e+00 -3.5797e-02 0.0000e+00 -1.5305e-02 0.0000e+00 -1.8994e-17 0.0000e+00 -1.0173e-17 0.0000e+00 
    -5.3725e-01 0.0000e+00 -3.5797e-02 0.0000e+00 1.5305e-02 0.0000e+00 2.5960e-17 0.0000e+00 -1.1086e-17 0.0000e+00 
    2 (band)
    4.3889e-01 (Ry)
    0.0000e+00 (Occupations)
    (...more coefficients...)
    >>>>>>> ends here

    For multiple-k calculations, each file corresponds to the coefficients
    of a k point, with filenames showing their internal k index. In the case
    of spin-2 calculations, all spin-down k points are indexed behind spin-up.
    For example, suppose there are 8 (spinless) k points, then there will be
    16 files, with the first 8 corresponding to spin-up. The k point in
    crystal coordinate is also given at the beginning of each file.

    Coefficients are complex in multi-k calculations; the real and imaginary
    parts are simply given one by one, e.g., the first coefficient above is
    (-5.3725e-01, 0.0000e+00).

    For gamma-only calculations, the output files differ in two ways. First,
    there is no k point coordinate at the beginning of the file. Second, the
    coefficients are real. An example of such a file is as follows:

    <<<<<<< starts here
    4 (number of bands)
    10 (number of orbitals)
    1 (band)
    -7.5250e-01 (Ry)
    2.0000e+00 (Occupations)
    -5.3725e-01 -3.6314e-02 -1.5455e-02 4.0973e-17 -3.6212e-18 
    -5.3725e-01 -3.6314e-02 1.5455e-02 7.2589e-18 1.4364e-18 
    2 (band)
    4.3622e-01 (Ry)
    0.0000e+00 (Occupations)
    (...more coefficients...)
    >>>>>>> ends here

    Notes
    -----
    This function applies to both gamma-only and multiple-k calculations.

    '''
    with open(fname, 'r') as f:
        data = f.read()
        data = data.replace('\n', ' ').split()

    is_gamma = False if 'k' in data else True
    nfloat_each_band = int(data[data.index('orbitals)') - 3]) \
                        * (1 if is_gamma else 2)

    # use the string "(band)" as delimiters
    delim = [i for i, x in enumerate(data) if x == '(band)']

    e = np.array([float(data[i+1]) for i in delim])
    occ = np.array([float(data[i+3]) for i in delim])
    wfc = np.array([[float(c) for c in data[i+5:i+5+nfloat_each_band]]
                    for i in delim])

    if not is_gamma:
        wfc = wfc.view(complex)

    return wfc, e, occ


def read_abacus_csr(fname):
    '''
    Read a CSR data file generated by ABACUS.

    When specifying "out_mat_hs2 1" (or out_mat_t) in INPUT, ABACUS will
    output H(R) & S(R) (or T(R)) in CSR format. This function reads such
    data file and returns the corresponding matrices, as well as their
    corresponding R vectors in crystal coordinate.

    '''

    with open(fname, 'r') as f:
        data = f.readlines()

    # make sure the file only contains the data for one step (geometry)
    assert(data[0].startswith('STEP: 0') \
            and not any(x.startswith('STEP') for x in data[1:]))

    sz = int(data[1].split()[-1])

    # some R may have no element at all, that's why we have to parse
    # sequentially and keep track of a line number instead of simply
    # dividing the rest data by blocks of 4.
    i = 3
    R = []
    mat = []
    while i < len(data):
        nnz = int(data[i].split()[-1])
        if nnz == 0:
            i += 1
            continue
        else:
            R.append(tuple(map(int, data[i].split()[:3])))
            val = list(map(float, data[i+1].split()))
            indices = list(map(float, data[i+2].split()))
            indptr = list(map(float, data[i+3].split()))
            mat.append(csr_matrix((val, indices, indptr), shape=(sz, sz)))
            i += 4

    return mat, R


############################################################
#                           Test
############################################################
import unittest

class _TestDatParse(unittest.TestCase):

    def test_read_orb_mat(self):
        fpath = './testfiles/Si/pw/Si-dimer-1.8/orb_matrix.0.dat'
        dat = read_orb_mat(fpath)

        nbes0 = int(np.sqrt(dat['ecutjlq']) * dat['rcut'] / np.pi)

        self.assertEqual(dat['ntype'], 1)
        self.assertEqual(dat['natom'], [2])
        self.assertEqual(dat['ecutwfc'], 40.0)
        self.assertEqual(dat['ecutjlq'], 40.0)
        self.assertEqual(dat['rcut'], 7.0)
        self.assertEqual(dat['lmax'], [2])
        self.assertEqual(dat['nbands'], 8)
        self.assertEqual(dat['nbes'], nbes0)
        self.assertEqual(dat['nk'], 1)
        self.assertTrue(np.all( dat['kpt'] == np.array([[0., 0., 0.]]) ))
        self.assertTrue(np.all( dat['wk'] == np.array([1.0]) ))

        nao = dat['natom'][0] * (dat['lmax'][0] + 1)**2

        self.assertEqual(dat['mo_jy'].shape,
                         (dat['nk'], dat['nbands'], nao*dat['nbes']))
        self.assertEqual(dat['jy_jy'].shape,
                         (dat['nk'], nao*dat['nbes'], nao*dat['nbes']))
        self.assertEqual(dat['mo_mo'].shape,
                         (dat['nk'], dat['nbands']))


        fpath = './testfiles/Si/pw/Si-trimer-1.7/orb_matrix.1.dat'
        dat = read_orb_mat(fpath)

        nbes0 = int(np.sqrt(dat['ecutjlq']) * dat['rcut'] / np.pi)

        self.assertEqual(dat['ntype'], 1)
        self.assertEqual(dat['natom'], [3])
        self.assertEqual(dat['ecutwfc'], 40.0)
        self.assertEqual(dat['ecutjlq'], 40.0)
        self.assertEqual(dat['rcut'], 7.0)
        self.assertEqual(dat['lmax'], [2])
        self.assertEqual(dat['nbands'], 12)
        self.assertEqual(dat['nbes'], nbes0)
        self.assertEqual(dat['nk'], 2)
        self.assertTrue(np.all( dat['kpt'] == np.array([[0., 0., 0.],
                                                        [0., 0., 0.]]) ))
        self.assertTrue(np.all( dat['wk'] == np.array([0.5, 0.5]) ))

        nao = dat['natom'][0] * (dat['lmax'][0] + 1)**2

        self.assertEqual(dat['mo_jy'].shape,
                         (dat['nk'], dat['nbands'], nao*dat['nbes']))
        self.assertEqual(dat['jy_jy'].shape,
                         (dat['nk'], nao*dat['nbes'], nao*dat['nbes']))
        self.assertEqual(dat['mo_mo'].shape,
                         (dat['nk'], dat['nbands']))


    def test_read_abacus_csr(self):
        # the test data is generated by ABACUS with integration test
        # 201_NO_15_f_pseudopots (a single Cerium atom)
        SR, R = read_abacus_csr('./testfiles/data-SR-sparse_SPIN0.csr')

        # S(0) should be identity
        i0 = R.index((0, 0, 0))
        sz = SR[i0].shape[0]
        self.assertTrue(np.allclose(SR[i0].toarray(), np.eye(sz)))


    def test_read_wfc_lcao_txt_gamma(self):
        wfc, e, occ = read_wfc_lcao_txt('./testfiles/WFC_NAO_GAMMA1.txt')
        self.assertEqual(wfc.shape, (4, 10))
        self.assertAlmostEqual(wfc[0,0], -0.53725)


    def test_read_wfc_lcao_txt_multik(self):
        wfc, e, occ = read_wfc_lcao_txt('./testfiles/WFC_NAO_K4.txt')
        self.assertEqual(wfc.shape, (4, 10))
        self.assertAlmostEqual(wfc[-1,-1], -7.0718e-02+6.4397e-04j)


if __name__ == '__main__':
    unittest.main()


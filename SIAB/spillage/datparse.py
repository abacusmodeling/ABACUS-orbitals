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

    In spillage-based orbital generation with plane-wave reference states,
    ABACUS will generate some "orb_matrix" data files which contain some
    system parameters as well as various overlaps. This function parses
    such a file into a dictionary containing the necessary data.

    Parameters
    ----------
        fpath : str
            Path of an "orb_matrix" data file.

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
        kpt : np.ndarray
            k-points.
        wk : np.ndarray
            k-point weights.
        mo_jy : np.ndarray
            Overlap between MOs and jYs.
            Shape: (nk, nbands, nao*nbes)
            NOTE: jYs are arranged in the lexicographic order of
            (itype, iatom, l, mm, q) where mm = 2*abs(m)-(m>0).
        jy_jy : np.ndarray
            Overlap between jYs.
            Shape: (nk, nao*nbes, nao*nbes)
            Note: the original jy_jy data assumed a shape of
            (nk, nao, nao, nbes, nbes), which is permuted and
            reshaped for convenience.
            NOTE: jYs are arranged in the lexicographic order of
            (itype, iatom, l, mm, q) where mm = 2*abs(m)-(m>0).
        mo_mo : np.ndarray
            Overlap between MOs.
            Shape: (nk, nbands)
        comp2lin, lin2comp : dict and list
            Bijective map between the composite and linearized index of
            atomic orbitals (not including zeta/spherical waves indices!).
            comp2lin: (itype, iatom, l, m) -> mu
            lin2comp[mu] -> (itype, iatom, l, m)

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
    #   bijective map between the composite and linearized indices
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

    if np.linalg.norm(np.imag(jy_jy.reshape(-1)), np.inf) < 1e-12:
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

    (taken from test/integrate/212_NO_wfc_out, precision may vary)
    <<<<<<< starts here
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

    Returns
    -------
        wfc : 2D array of shape (nao, nbands)
            Wave function coefficients in LCAO basis. The datatype is
            complex for multi-k calculations and float for gamma-only.
        e : array
            Band energies.
        occ : array
            Occupations.

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

    return wfc.T, e, occ


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


def read_abacus_tri(fname):
    '''
    Read a triangular matrix file generated by ABACUS.

    When specifying "out_mat_hs 1" (or out_mat_tk 1) in INPUT, ABACUS will
    output H(k) & S(k) (or T(k)) to text files. The content of such files
    merely contains the upper-triangle of the matrix.

    '''
    with open(fname, 'r') as f:
        data = f.read()
        data = re.sub('\(|\)|,|\n', ' ', data).split()

    # the first element of the file is the size of the matrix
    sz = int(data[0])
    assert len(data) == sz*(sz+1)//2 + 1 or len(data) == sz*(sz+1) + 1
    dtype = complex if len(data) == sz*(sz+1) + 1 else float

    M = np.zeros((sz, sz), dtype=dtype)
    idx_u = np.triu_indices(sz)
    M[idx_u] = np.array([float(x) for x in data[1:]]).view(dtype)

    # symmetrize the matrix
    idx_l = np.tril_indices(sz)
    M[idx_l] = M.T[idx_l].conj()

    # make diagonal elements real (fix floating point error)
    M[np.diag_indices(sz)] = np.real(M[np.diag_indices(sz)])

    return M


def read_abacus_kpoints(fname):
    '''
    Read a "kpoints" file generated by ABACUS.

    '''
    with open(fname, 'r') as f:
        data = f.read().replace('\n', ' ').split()

    nk = int(data[3])
    wk = [float(data[i]) for i in range(16, 16+(nk-1)*5+1, 5)]
    k = [tuple(map(float, data[i:i+3])) for i in range(13, 13+(nk-1)*5+1, 5)]

    return k, wk



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
        self.assertEqual(wfc.shape, (10, 4))
        self.assertAlmostEqual(wfc[0,0], -0.53725)


    def test_read_wfc_lcao_txt_multik(self):
        wfc, e, occ = read_wfc_lcao_txt('./testfiles/WFC_NAO_K4.txt')
        self.assertEqual(wfc.shape, (10, 4))
        self.assertAlmostEqual(wfc[-1,-1], -7.0718e-02+6.4397e-04j)


    def test_read_abacus_tri_gamma(self):
        Tk = read_abacus_tri('./testfiles/data-0-T')
        self.assertEqual(Tk.shape, (34, 34))
        self.assertTrue(np.all(Tk == Tk.T))
        self.assertEqual(Tk[0,0], 0.243234968741)
        self.assertEqual(Tk[0,1], 0.137088191847)
        self.assertEqual(Tk[-1,-1], 2.20537992217)


    def test_read_abacus_tri_multik(self):
        k = np.array([0.4, 0.4, 0.4]) # direct coordinates
        Tk = read_abacus_tri('./testfiles/data-2-T')

        TR, R_all = read_abacus_csr('./testfiles/data-TR-sparse_SPIN0.csr')
        Tk_ = np.zeros_like(Tk, dtype=complex)
        for i, R in enumerate(R_all):
            Tk_ += TR[i] * np.exp(2j * np.pi * np.dot(k, R))

        self.assertTrue(np.allclose(Tk, Tk_))


    def test_read_abacus_kpoints(self):
        k, wk = read_abacus_kpoints('./testfiles/kpoints')

        self.assertEqual(len(k), 10)
        self.assertEqual(len(wk), 10)

        self.assertTrue(np.allclose(k[0], [0.0, 0.0, 0.0]))
        self.assertEqual(wk[0], 0.008)

        self.assertTrue(np.allclose(k[-1], [0.4, -0.4, 0.2]))
        self.assertEqual(wk[-1], 0.192)



if __name__ == '__main__':
    unittest.main()


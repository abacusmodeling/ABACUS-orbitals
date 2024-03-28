import re
import numpy as np
import itertools

from jlzeros import JLZEROS
from scipy.special import spherical_jn
from indexmap import _index_map


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
            Shape: (nk, nbands, nao, nbes)
        jy_jy : np.ndarray
            Overlap between jYs.
            Shape: (nk, nao, nao, nbes, nbes)
        mo_mo : np.ndarray
            Overlap between MOs.
            Shape: (nk, nbands)
        comp2mu, mu2comp : dict
            Bijective index map (itype, iatom, l, zeta, m) <-> mu.
            comp2mu: (itype, iatom, l, zeta, m) -> mu
            mu2comp: mu -> (itype, iatom, l, zeta, m)
            zeta is always 0 in the present code.

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
    #       bijective index map (itype, iatom, l, zeta, m) <-> mu
    ####################################################################
    comp2mu, mu2comp = _index_map(ntype, natom, lmax)
    nao = len(comp2mu)

    ####################################################################
    #                           MO-jY overlap
    ####################################################################
    mo_jy_start= data.index('<OVERLAP_Q>') + 1
    mo_jy_end = data.index('</OVERLAP_Q>')
    mo_jy = np.array(data[mo_jy_start:mo_jy_end], dtype=float) \
            .view(dtype=complex) \
            .reshape((nk, nbands, nao, nbes))

    ####################################################################
    #                           Phase Adjustment
    ####################################################################
    # NOTE In theory this step should not exist at all!
    # but currently mo_jy computed by ABACUS does carry some non-zero phase
    # of unknown origin.

    #for ik in range(nk):
    #    for ib in range(nbands):
    #        idx = np.argmax(np.abs(mo_jy[ik, ib]))
    #        mo_jy[ik, ib] *= np.exp(-1j * np.angle(mo_jy[ik, ib].reshape(-1)[idx]))
            #print('ib = {}   max mo_jy imag = {} ' \
            #        .format(ib, np.linalg.norm(np.imag(mo_jy[ik, ib]), np.inf)))


    #for mu in range(nao):
    #    tmp = mo_jy[:, mu, :].reshape(-1)
    #    idx = np.argmax(np.abs(tmp))
    #    mo_jy[:, mu, :] *= np.exp(-1j * np.angle(tmp[idx]))
    #    print('mu = {}   max mo_jy imag = {} ' \
    #            .format(mu, np.linalg.norm(np.imag(mo_jy[:, mu, :]), np.inf)))
    
    #for ib in range(nbands):
    #    for mu in range(nao):
    #        idx = np.argmax(np.abs(mo_jy[ib, mu]))
    #        mo_jy[ib, mu] *= np.exp(-1j * np.angle(mo_jy[ib, mu, idx]))
    #        print('ib = {}   mu = {}   max mo_jy imag = {}   norm = {}' \
    #            .format(ib, mu, np.linalg.norm(np.imag(mo_jy[ib, mu]), np.inf), \
    #            np.linalg.norm(mo_jy[ib,mu]) ))


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

    # NOTE permute jy_jy such that it has a shape of (nk, nao, nbes, nao, nbes)
    # which is more convenient for later use.
    jy_jy = jy_jy.transpose((0, 1, 3, 2, 4))

    ####################################################################
    #                           MO-MO overlap
    ####################################################################
    # should be all 1
    mo_mo_start= data.index('<OVERLAP_V>') + 1
    mo_mo_end = data.index('</OVERLAP_V>')
    mo_mo = np.array(data[mo_mo_start:mo_mo_end], dtype=float)

    assert len(mo_mo) == nbands * nk
    mo_mo = mo_mo.reshape((nk, nbands))


    return {'ntype': ntype, 'natom': natom, 'ecutwfc': ecutwfc, \
            'ecutjlq': ecutjlq, 'rcut': rcut, 'lmax': lmax, 'nk': nk, \
            'nbands': nbands, 'nbes': nbes, 'kpt': kpt, 'wk': wk, \
            'jy_jy': jy_jy, 'mo_jy': mo_jy, 'mo_mo': mo_mo, \
            'comp2mu': comp2mu, 'mu2comp': mu2comp}


def are_consistent(dat1, dat2):
    '''
    Check if two dat files corresponds to the same system.

    '''
    return dat1['mu2comp'] == dat2['mu2comp'] and \
            dat1['rcut'] == dat2['rcut'] and \
            np.all(dat1['wk'] == dat2['wk']) and \
            np.all(dat1['kpt'] == dat2['kpt'])


############################################################
#                           Test
############################################################
import unittest

class _TestDatParse(unittest.TestCase):

    def test_read_orb_mat(self):
        fpath = './testfiles/orb_matrix_rcut6deriv0.dat'
        dat = read_orb_mat(fpath)

        self.assertEqual(dat['ntype'], 1)
        self.assertEqual(dat['natom'], [3])
        self.assertEqual(dat['ecutwfc'], 10.0)
        self.assertEqual(dat['ecutjlq'], 10.0)
        self.assertEqual(dat['rcut'], 6.0)
        self.assertEqual(dat['lmax'], [2])
        self.assertEqual(dat['nbands'], 10)
        self.assertEqual(dat['nbes'], \
                int(np.sqrt(dat['ecutjlq']) * dat['rcut'] / np.pi))
        self.assertEqual(dat['nk'], 1)
        self.assertTrue(np.all( dat['kpt'] == np.array([[0., 0., 0.]]) ))
        self.assertTrue(np.all( dat['wk'] == np.array([1.0]) ))

        nao = dat['natom'][0] * (dat['lmax'][0] + 1)**2

        self.assertEqual(dat['mo_jy'].shape, \
                (dat['nk'], dat['nbands'], nao, dat['nbes']))
        #self.assertEqual(dat['jy_jy'].shape, \
        #        (dat['nk'], nao, nao, dat['nbes'], dat['nbes']))
        self.assertEqual(dat['jy_jy'].shape, \
                (dat['nk'], nao, dat['nbes'], nao, dat['nbes']))
        self.assertEqual(dat['mo_mo'].shape, \
                (dat['nk'], dat['nbands']))


        fpath = './testfiles/orb_matrix_rcut7deriv1.dat'
        dat = read_orb_mat(fpath)

        self.assertEqual(dat['ntype'], 1)
        self.assertEqual(dat['natom'], [2])
        self.assertEqual(dat['ecutwfc'], 10.0)
        self.assertEqual(dat['ecutjlq'], 10.0)
        self.assertEqual(dat['rcut'], 7.0)
        self.assertEqual(dat['lmax'], [2])
        self.assertEqual(dat['nbands'], 8)
        self.assertEqual(dat['nbes'], \
                int(np.sqrt(dat['ecutjlq']) * dat['rcut'] / np.pi))
        self.assertEqual(dat['nk'], 2)
        self.assertTrue(np.all( dat['kpt'] == np.array([[0., 0., 0.], [0., 0., 0.]]) ))
        self.assertTrue(np.all( dat['wk'] == np.array([0.5, 0.5]) ))

        nao = dat['natom'][0] * (dat['lmax'][0] + 1)**2

        self.assertEqual(dat['mo_jy'].shape, \
                (dat['nk'], dat['nbands'], nao, dat['nbes']))
#        self.assertEqual(dat['jy_jy'].shape, \
#                (dat['nk'], nao, nao, dat['nbes'], dat['nbes']))
        self.assertEqual(dat['jy_jy'].shape, \
                (dat['nk'], nao, dat['nbes'], nao, dat['nbes']))
        self.assertEqual(dat['mo_mo'].shape, \
                (dat['nk'], dat['nbands']))


if __name__ == '__main__':
    unittest.main()


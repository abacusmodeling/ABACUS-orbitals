import re
import numpy as np
import itertools

from jlzeros import JLZEROS
from scipy.special import spherical_jn
from indexmap import _index_map


def read(job_dir):
    '''
    Reads the overlap matrix from the job directory.

    '''
    pass

def _read_orb_mat(fpath):
    '''
    Reads an orb_matrix_xxx.dat file and retrieves
    the MO-jY overlap as well as the jY-jY overlap.
    (MO = molecular orbital, jY = [spherical Bessel]x[spherical harmonics])

    The MOs have 2 indices: band & k (composited with spin).
    The jYs have 5 indices: atom type (which affects the cutoff radius),
    angular & magnetic quantum numbers, atomic position and spherical Bessel
    wave number.

    '''
    with open(fpath, 'r') as f:
        data = f.read()
        data = data.replace('\n', ' ').split()

    ntype = int(data[data.index('ntype') - 1])
    natom = [int(data[i-1]) for i, label in enumerate(data[:data.index('ecutwfc')]) \
            if label == 'na']

    # ecutwfc of pw calculation
    ecutwfc = float(data[data.index('ecutwfc') - 1])

    # ecut for wave numbers & "kmesh" (for simpson-based spherical Bessel transforms)
    # in the present code, ecutjlq = ecutwfc
    ecutjlq = float(data[data.index('ecutwfc_jlq') - 1])

    # cutoff radius of spherical Bessel functions
    rcut = float(data[data.index('rcut_Jlq') - 1])

    lmax = int(data[data.index('lmax') - 1])
    nk = int(data[data.index('nks') - 1])
    nbands = int(data[data.index('nbands') - 1])
    nbes = int(data[data.index('ne') - 1])

    print('ntype = {}'.format(ntype))
    print('natom = {}'.format(natom))
    print('ecutwfc = {}'.format(ecutwfc))
    print('ecutjlq = {}'.format(ecutjlq))
    print('rcut = {}'.format(rcut))
    print('lmax = {}'.format(lmax))
    print('nk = {}'.format(nk))
    print('nbands = {}'.format(nbands))
    print('nbes = {}'.format(nbes))

    wk_start= data.index('<WEIGHT_OF_KPOINTS>') + 1
    wk_end = data.index('</WEIGHT_OF_KPOINTS>')
    kinfo= np.array(data[wk_start:wk_end], dtype=float).reshape(nk, 4)
    kpt = kinfo[:, 0:3]
    wk = kinfo[:, 3]
    print('kpts = ', kpt)
    print('wk = ', wk)

    # NOTE In PW calculations, lmax is always the same for all element types,
    # which is the lmax read above. (Will it be different in the future?)

    ####################################################################
    #       bijective index map (itype, iatom, l, zeta, m) <-> mu
    ####################################################################
    comp2mu, mu2comp = _index_map(ntype, natom, [lmax])
    #print(comp2mu)
    #print(mu2comp)

    # number of distinct (itype, iatom, l, m) indices
    # "talm" stands for "type-atom-l-m"
    ntalm = len(comp2mu.items())
    print('ntalm = ', ntalm)

    ####################################################################
    #                           MO-jY overlap
    ####################################################################
    mo_jy_start= data.index('<OVERLAP_Q>') + 1
    mo_jy_end = data.index('</OVERLAP_Q>')
    mo_jy = np.array(data[mo_jy_start:mo_jy_end], dtype=float).view(dtype=complex) \
            .reshape((nk, nbands, ntalm, nbes))
    print('mo_jy.shape = ', mo_jy.shape)

    ####################################################################
    #                           Phase Adjustment
    ####################################################################
    # NOTE In theory this step should not exist at all!
    # but currently mo_jy computed by ABACUS does carry some non-zero phase
    # of unknown origin.

    for ik in range(nk):
        for ib in range(nbands):
            idx = np.argmax(np.abs(mo_jy[ik, ib]))
            mo_jy[ik, ib] *= np.exp(-1j * np.angle(mo_jy[ik, ib].reshape(-1)[idx]))
            print('ib = {}   max mo_jy imag = {} ' \
                    .format(ib, np.linalg.norm(np.imag(mo_jy[ik, ib]), np.inf)))


    #for mu in range(ntalm):
    #    tmp = mo_jy[:, mu, :].reshape(-1)
    #    idx = np.argmax(np.abs(tmp))
    #    mo_jy[:, mu, :] *= np.exp(-1j * np.angle(tmp[idx]))
    #    print('mu = {}   max mo_jy imag = {} ' \
    #            .format(mu, np.linalg.norm(np.imag(mo_jy[:, mu, :]), np.inf)))
    
    #for ib in range(nbands):
    #    for mu in range(ntalm):
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
    jy_jy = np.array(data[jy_jy_start:jy_jy_end], dtype=float).view(dtype=complex)

    # overlap between jY should be real
    assert np.linalg.norm(np.imag(jy_jy), np.inf) < 1e-14

    jy_jy = np.real(jy_jy).reshape((nk, ntalm, ntalm, nbes, nbes))
    print('jy_jy.shape = ', jy_jy.shape)

    mu = 0
    #print(jy_jy[mu, mu, :, :])

    _, _, l, _, _ = mu2comp[mu]

    rcut = 6.0
    print('l = ', l )
    print('abacus = ', np.diag(jy_jy[1, mu, mu, :, :]))
    print('abacus (factor fixed) = ', np.diag(jy_jy[1, mu, mu, :, :]) * (np.pi/10)**2)
    print('exact = ', 0.5 * rcut**3 * spherical_jn(l+1, JLZEROS[l][:nbes])**2 )
    print('exact (deriv) = ', 0.5 * rcut * (JLZEROS[l][:nbes] * spherical_jn(l+1, JLZEROS[l][:nbes]))**2 )

    ####################################################################
    #                           MO-MO overlap
    ####################################################################
    # should be all 1
    mo_mo_start= data.index('<OVERLAP_V>') + 1
    mo_mo_end = data.index('</OVERLAP_V>')
    mo_mo = np.array(data[mo_mo_start:mo_mo_end], dtype=float)
    assert len(mo_mo) == nbands * nk

    mo_mo = mo_mo.reshape((nk, nbands))
    print('mo_mo: ', mo_mo)

    return {'ntype': ntype, 'natom': natom, 'ecutwfc': ecutwfc, \
            'ecutjlq': ecutjlq, 'rcut': rcut, 'lmax': lmax, 'nk': nk, \
            'nbands': nbands, 'nbes': nbes, 'kpt': kpt, 'wk': wk, \
            'jy_jy': jy_jy, 'mo_jy': mo_jy, 'mo_mo': mo_mo, \
            'comp2mu': comp2mu, 'mu2comp': mu2comp}


    #============FIXME===================


############################################################
#                           Test
############################################################
import os
import unittest

class _TestDataRead(unittest.TestCase):

    def test_read_orb_mat(self):
        fpath = '/home/zuxin/abacus-community/abacus_orbital_generation/tmp/Si-dimer-2.0/orb_matrix_rcut6deriv0.dat'
        _read_orb_mat(fpath)



if __name__ == '__main__':
    unittest.main()


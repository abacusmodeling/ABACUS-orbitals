import re
import numpy as np
import itertools

def _parse_overlap_q():
    pass


def _parse_overlap_sq():
    pass

def _index_map(ntype, natom, lmax):
    mu = 0
    index_map = {}
    for itype in range(ntype):
        for iatom in range(natom[itype]):
            for l in range(lmax+1):
                '''
                In ABACUS, magnetic quantum numbers are often looped from 0 to 2l
                (instead of -l to l), and, in terms of the m of real spherical
                harmonics (as given by module_base/ylm.cpp, in a sign convention
                consistent with Homeier1996), follow:

                              0, 1, -1, 2, -2, 3, -3, ..., l, -l
                
                Here we "demangle" this index so that the returned index_map can be
                indexed by ordinary magnetic quantum number m \in [-l, l] like

                            mu_index = index_map[(itype, iatom, l, m)]

                '''
                for mm in range(0, 2*l+1):
                    m = -mm // 2 if mm % 2 == 0 else (mm + 1) // 2
                    index_map[(itype, iatom, l, m)] = mu
                    mu += 1

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

    mo_jy_start= data.index('<OVERLAP_Q>') + 1
    mo_jy_end = data.index('</OVERLAP_Q>')
    mo_jy = np.array(data[mo_jy_start:mo_jy_end], dtype=float).view(dtype=complex)
    print('mo_jy: ', mo_jy[0:5])

    print('angle: ', np.angle(mo_jy[0:10]))
    print('ratio: ', np.real(mo_jy[0:10]) / np.imag(mo_jy[:10]))


    # reshape TBD



    jy_jy_start= data.index('<OVERLAP_Sq>') + 1
    jy_jy_end = data.index('</OVERLAP_Sq>')
    jy_jy = np.array(data[jy_jy_start:jy_jy_end], dtype=float).view(dtype=complex)

    # overlap between jY should be real
    assert np.linalg.norm(np.imag(jy_jy), np.inf) < 1e-14
    jy_jy = np.real(jy_jy)

    # build index map (itype, iatom, l, m) -> mu
    index_map = _index_map(ntype, natom, lmax)

    # rearrange
    ovl_jy = {}
    for itype_pair in itertools.product(range(ntype), repeat=2):
        ovl_jy[itype_pair] = jy_jy[i[0]*nbes:(i[0]+1)*nbes, i[1]*nbes:(i[1]+1)*nbes]




    mo_mo_start= data.index('<OVERLAP_V>') + 1
    mo_mo_end = data.index('</OVERLAP_V>')
    mo_mo = np.array(data[mo_mo_start:mo_mo_end], dtype=float)
    assert len(mo_mo) == nbands

    print('mo_mo: ', mo_mo)










############################################################
#                           Test
############################################################
import os
import unittest

class _TestDataRead(unittest.TestCase):

    def test_read(self):
        fpath = '/home/zuxin/abacus-community/abacus_orbital_generation/tmp/Si-dimer-1.22/orb_matrix_rcut6deriv0.dat'
        _read_orb_mat(fpath)


if __name__ == '__main__':
    unittest.main()


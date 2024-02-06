import re
import numpy as np

def _parse_overlap_q():
    pass


def _parse_overlap_sq():
    pass

def read(job_dir):
    '''
    Reads the overlap matrix from the job directory.

    '''
    pass

def _read_orb_mat(fpath):
    '''
    Reads an orb_matrix.x.dat file (x = 0 or 1) and retrieves
    the MO-jY overlap as well as the jY-jY overlap.
    (MO = molecular orbital, jY = [spherical Bessel]x[spherical harmonics])

    The MOs have (up to) 3 indices: band, k and spin;
    The jYs have 5 indices: atom type (which affects the cutoff radius),
    angular & magnetic quantum numbers, atomic position and spherical Bessel
    wave number.

    '''
    with open(fpath, 'r') as f:
        data = f.read()
        data = data.replace('\n', ' ').split(' ')

    ntype = int(data[data.index('ntype') - 1])
    assert ntype == 1, 'ntype in orbital generation must be 1'

    natom = int(data[data.index('na') - 1])

    ecutwfc = float(data[data.index('ecutwfc') - 1]) # ecutwfc of pw calculation
    ecutjlq = float(data[data.index('ecutwfc_jlq') - 1]) # ecut for wave numbers & kmesh determination
    rcut = float(data[data.index('rcut_Jlq') - 1]) # cutoff radius of spherical Bessel functions
                
    lmax = int(data[data.index('lmax') - 1])
    nk = int(data[data.index('nks') - 1])
    nbands = int(data[data.index('nbands') - 1])
    nbes = int(data[data.index('ne') - 1])


    print('ntype = {}'.format(ntype))
    print('natom = {}'.format(natom))
    print('nbes = {}'.format(nbes))




############################################################
#                           Test
############################################################
import os
import unittest

class _TestDataRead(unittest.TestCase):

    def test_read(self):
        fpath = '/home/zuxin/abacus-community/ABACUS-orbitals/tmp/14_Si_100Ry/OUT.Si-STRU1-7-1.8/orb_matrix.0.dat'
        _read_orb_mat(fpath)


if __name__ == '__main__':
    unittest.main()


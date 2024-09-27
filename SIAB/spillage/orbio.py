from SIAB.spillage.radial import _nbes, build_raw, build_reduced

import numpy as np
import re
from itertools import accumulate

def _write_header(f, elem, ecut, rcut, nzeta, nr, dr):
    '''
    Writes an ABACUS orbital file header to a file object.
    
    A typical header looks like
    
    <<<<<<< starts here (taken from C_gga_8au_100Ry_2s2p1d.orb)
    --------------------------------------------------------------------------
    Element                     C
    Energy Cutoff(Ry)          100
    Radius Cutoff(a.u.)         8
    Lmax                        2
    Number of Sorbital-->       2
    Number of Porbital-->       2
    Number of Dorbital-->       1
    --------------------------------------------------------------------------
    SUMMARY  END
    
    Mesh                        801
    dr                          0.01
    >>>>>>> ends here
    
    Parameters
    ----------
        f : file object
            Must be opened in advance.
        elem : str
            Element symbol.
        ecut : int or float
            Energy cutoff.
        rcut : int or float
            Cutoff radius.
        nzeta : list of int
            Number of orbitals for each angular momentum.
        lmax : int
            Maximum angular momentum.
        nr : int
            Number of radial grid points.
        dr : float
            Grid spacing.

    '''
    lmax = len(nzeta)-1
    spec_symbol = 'SPDFGHIKLMNOQRTUVWXYZ'

    f.write('-' * 75 + '\n')
    f.write(f'Element                     {elem}\n')
    f.write(f'Energy Cutoff(Ry)           {ecut}\n')
    f.write(f'Radius Cutoff(a.u.)         {rcut}\n')
    f.write(f'Lmax                        {lmax}\n')

    for l in range(lmax+1):
        f.write(f"Number of {spec_symbol[l]}orbital-->       {nzeta[l]}\n")

    f.write('-' * 75 + '\n')
    f.write('SUMMARY  END\n\n')
    f.write(f'Mesh                        {nr}\n')
    f.write(f'dr                          {dr}\n')


def _write_chi(f, l, zeta, chi):
    '''
    Writes a numerical radial function to a file object.
    
    Parameters
    ----------
        f : file object
            Must be opened in advance.
        l : int
            Angular momentum.
        zeta : int
            Zeta number.
        chi : array of float
            A radial function on a grid.

    '''
    f.write(f"{'Type':>20}{'L':>16}{'N':>16}\n")
    f.write(f"{'0':>20}{l:>16}{zeta:>16}\n")

    for ir, chi_of_r in enumerate(chi):
        f.write(f'{chi_of_r: 23.14e}')
        if ir % 4 == 3 and ir != len(chi)-1:
            f.write('\n')
    f.write('\n')


def write_nao(fpath, elem, ecut, rcut, nr, dr, chi):
    '''
    Generates a numerical atomic orbital file of the ABACUS format.
    
    Parameters
    ----------
        fpath : str
            Path to the orbital file.
        elem : str
            Element symbol.
        ecut : float
            Energy cutoff.
        rcut : float
            Cutoff radius.
        nr : int
            Number of radial grid points.
        dr : float
            Grid spacing.
        chi : list of list of array of float
            A nested list of numerical radial functions,
            chi[l][zeta][ir] -> float

    '''
    lmax = len(chi)-1
    nzeta = [len(chi[l]) for l in range(lmax+1)]
    
    with open(fpath, 'w') as f:
        _write_header(f, elem, ecut, rcut, nzeta, nr, dr)
        for l in range(lmax+1):
            for zeta in range(nzeta[l]):
                _write_chi(f, l, zeta, chi[l][zeta])


def read_nao(fpath):
    '''
    Reads a numerical atomic orbital file of the ABACUS format.
    
    Parameters
    ----------
        fpath : str
            Path to the orbital file.
    
    Returns
    -------
        A dictionary containing the following key-value pairs:

        'elem' : str
            Element symbol.
        'ecut' : float
            Energy cutoff.
        'rcut' : float
            Cutoff radius of the orbital.
        'nr' : int
            Number of radial grid points.
        'dr' : float
            Grid spacing.
        'chi' : list of list of array of float
            A nested list of numerical radial functions,
            chi[l][zeta][ir] -> float.

    '''
    with open(fpath, 'r') as f:
        data = list(filter(None, re.split('\t| |\n', f.read())))

    elem = data[data.index('Element')+1]
    ecut = float(data[data.index('Cutoff(Ry)')+1])
    rcut = float(data[data.index('Cutoff(a.u.)')+1])
    lmax = int(data[data.index('Lmax')+1])

    spec_symbol = 'SPDFGHIKLMNOQRTUVWXYZ'
    nzeta = [int(data[data.index(spec_symbol[l] + 'orbital-->') + 1])
             for l in range(lmax+1)]

    nr = int(data[data.index('Mesh')+1])
    dr = float(data[data.index('dr')+1])

    delim = [i for i, x in enumerate(data) if x == 'Type'] + [len(data)]
    nzeta_cumu = [0] + list(accumulate(nzeta))
    iorb = lambda l, zeta : nzeta_cumu[l] + zeta
    chi = [[np.array(data[delim[iorb(l,zeta)]+6:delim[iorb(l,zeta)+1]],
                     np.float64)
            for zeta in range(nzeta[l]) ] for l in range(lmax+1)]

    return {'elem': elem, 'ecut': ecut, 'rcut': rcut, 'nr': nr, 'dr': dr,
            'chi': chi}


def _extract(keyword, text):
    '''
    Extracts VALUE from the pattern KEYWORD=" VALUE ".

    '''
    result = re.search(keyword + '=" *([^= ]*) *"', text)
    return result.group(1) if result else None


def read_param(fpath):
    '''
    Reads an orbital parameter file of the SIAB/PTG format.
    
    Parameters
    ----------
        fpath : str
            Path to the orbital parameter file.
    
    Returns
    -------
        A dictionary containing the following key-value pairs:

        'coeff' : list of list of list of float
            A nested list of spherical Bessel coefficients,
            coeff[l][zeta][iq] -> float.
        'rcut' : float
            Cutoff radius of the orbital.
        'sigma' : float
            Smoothing width.
        'elem' : str
            Element symbol.

    '''
    with open(fpath, 'r') as f:
        data = f.read()

    # convert '\n' to ' ' for regex matching (.)
    data = data.replace('\n', ' ')

    # extract the Coefficient block
    result = re.search('<Coefficient(.*)</Coefficient>', data)
    if result is None:
        raise ValueError('Coefficient block not found.')
    data = result.group(1)

    # extract the parameters in header
    rcut = float(_extract('rcut', data))
    sigma = float(_extract('sigma', data))
    elem = _extract('element', data)

    # split the data into a list of strings
    data = list(filter(None, re.split('\t| ', data)))
    delim = [i for i, x in enumerate(data) if x == 'Type'] + [len(data)]
    ll = [int(data[delim[i]+4]) for i in range(len(delim)-1)]
    lmax = max(ll)
    nzeta = [ll.count(l) for l in range(lmax+1)]
    
    nzeta_cumu = [0] + list(accumulate(nzeta))
    iorb = lambda l, zeta : nzeta_cumu[l] + zeta
    coeff = [[list(map(float,
                       data[delim[iorb(l,zeta)]+6:delim[iorb(l,zeta)+1]]))
              for zeta in range(nzeta[l])] for l in range(lmax+1)]

    return {'coeff': coeff, 'rcut': rcut, 'sigma': sigma, 'elem': elem}


def write_param(fpath, coeff, rcut, sigma, elem):
    '''
    Writes orbital parameters to a file of the SIAB/PTG format.
    
    Parameters
    ----------
        fpath : str
            Path to the orbital parameter file.
        coeff : list of list of list of float
            Spherical Bessel coefficients as coeff[l][zeta][iq].
        rcut : float
            Cutoff radius of the orbital.
        sigma : float
            Smoothing width.
        elem : str
            Element symbol.

    '''
    with open(fpath, 'w') as f:
        lmax = len(coeff)-1
        nzeta = [len(coeff[l]) for l in range(lmax+1)]
        n = sum(nzeta)

        f.write(f'<Coefficient rcut="{rcut}" sigma="{sigma}" ' \
                f'element="{elem}">\n')
        f.write(f'     {n} Total number of radial orbitals.\n')

        for l in range(lmax+1):
            for zeta in range(nzeta[l]):
                f.write('    Type   L   Zeta-Orbital\n')
                f.write(f'      {elem}   {l}       {zeta}\n')

                for i in range(len(coeff[l][zeta])):
                    f.write(f'{coeff[l][zeta][i]: 21.14f}\n')

        f.write('</Coefficient>\n')


def jygen(fname, rcut, dr, lmax, ecut, elem, reduced=True):
    '''
    Generates a normalized/reduced spherical wave (jy) basis.

    '''
    if reduced:
        nbes = [_nbes(l, rcut, ecut) - 1 for l in range(lmax + 1)]
        suffix = 'reduced'
    else:
        nbes = [_nbes(l, rcut, ecut) for l in range(lmax + 1)]
        suffix = 'normalized'

    coef = [np.eye(nbes[l]).tolist() for l in range(lmax+1)]

    r = np.linspace(0.0, rcut, int(rcut / dr) + 1)
    if reduced:
        chi = build_reduced(coef, rcut, r, True)
    else:
        chi = build_raw(coef, rcut, r, 0.0, True)
            
    spec_symbol = 'spdfghiklmnoqrtuvwxyz'

    if fname is None:
        fname = f'jy_{suffix}_{rcut}au_{ecut}Ry_' \
                + ''.join([f"{nbes[l]}{spec_symbol[l]}"
                           for l in range(lmax+1)]) \
                + '.orb'
    write_nao(fname, elem, ecut, rcut, len(r), dr, chi)


############################################################
#                           Test
############################################################
import os
import unittest

class _TestOrbIO(unittest.TestCase):

    def test_read_param(self):
        import os
        here = os.path.dirname(os.path.abspath(__file__))
        jobdir = os.path.join(here, 'testfiles')

        #param = read_param('./testfiles/ORBITAL_RESULTS.txt')
        param = read_param(os.path.join(jobdir, 'ORBITAL_RESULTS.txt'))

        self.assertEqual(param['elem'], 'In')
        self.assertEqual(param['rcut'], 10.0)
        self.assertEqual(param['sigma'], 0.1)

        coeff = param['coeff']
        lmax = len(coeff)-1
        nzeta = [len(coeff[l]) for l in range(lmax+1)]
        nq = [len(coeff[l][zeta])
              for l in range(lmax+1)
              for zeta in range(nzeta[l])]

        self.assertEqual(lmax, 3)
        self.assertEqual(nzeta, [3, 3, 3, 2])
        self.assertEqual(nq, [31] * 11)
        self.assertEqual(coeff[0][0][0], 0.09775364133148)
        self.assertEqual(coeff[0][0][30], 0.00025319062647)
        self.assertEqual(coeff[1][2][0], -0.31113835728419)
        self.assertEqual(coeff[3][1][30], -0.39682668632947)


    def test_write_param(self):
        import os
        here = os.path.dirname(os.path.abspath(__file__))
        jobdir = os.path.join(here, 'testfiles')

        #param = read_param('./testfiles/ORBITAL_RESULTS.txt')
        param = read_param(os.path.join(jobdir, 'ORBITAL_RESULTS.txt'))
        #tmpfile = './testfiles/ORBITAL_RESULTS.txt.tmp'
        tmpfile = os.path.join(jobdir, 'ORBITAL_RESULTS.txt.tmp')
        write_param(tmpfile, **param)
        param2 = read_param(tmpfile)
        os.remove(tmpfile)
        self.assertDictEqual(param, param2)
    
    
    def test_read_nao(self):
        import os
        here = os.path.dirname(os.path.abspath(__file__))
        jobdir = os.path.join(here, 'testfiles')

        #nao = read_nao('./testfiles/In_gga_10au_100Ry_3s3p3d2f.orb')
        nao = read_nao(os.path.join(jobdir, 'In_gga_10au_100Ry_3s3p3d2f.orb'))

        self.assertEqual(nao['elem'], 'In')
        self.assertEqual(nao['rcut'], 10.0)
        self.assertEqual(nao['ecut'], 100.0)
        self.assertEqual(nao['dr'], 0.01)
        self.assertEqual(nao['nr'], 1001)

        chi = nao['chi']
        lmax = len(chi)-1
        nzeta = [len(chi[l]) for l in range(lmax+1)]

        self.assertEqual(lmax, 3)
        self.assertEqual(nzeta, [3, 3, 3, 2])
        self.assertAlmostEqual(chi[0][0][0], 3.63938508691915e-03)
        self.assertAlmostEqual(chi[0][0][1000], 0.000000000000e+00)
        self.assertAlmostEqual(chi[0][2][0], 2.96686754251501e+00)
        self.assertAlmostEqual(chi[2][2][1], 6.64420946004809e-05)
        self.assertAlmostEqual(chi[3][1][4], -1.08612755666000e-04)

    
    def test_write_nao(self):
        import os
        here = os.path.dirname(os.path.abspath(__file__))
        jobdir = os.path.join(here, 'testfiles')

        #nao = read_nao('./testfiles/In_gga_10au_100Ry_3s3p3d2f.orb')
        nao = read_nao(os.path.join(jobdir, 'In_gga_10au_100Ry_3s3p3d2f.orb'))
        #tmpfile = './testfiles/In_gga_10au_100Ry_3s3p3d2f.orb.tmp'
        tmpfile = os.path.join(jobdir, 'In_gga_10au_100Ry_3s3p3d2f.orb.tmp')

        write_nao(tmpfile, **nao)
        nao2 = read_nao(tmpfile)
        os.remove(tmpfile)

        # numpy array comparisons are element-wise;
        # convert to list to compare them as a whole
        nao['chi'] = [[chi_lq.tolist() for chi_lq in chi_l ]
                      for chi_l in nao['chi']]
        nao2['chi'] = [[chi_lq.tolist() for chi_lq in chi_l ]
                       for chi_l in nao2['chi']]
        self.assertDictEqual(nao, nao2)


    def test_jygen(self):
        import os
        here = os.path.dirname(os.path.abspath(__file__))
        jobdir = os.path.join(here, 'testfiles')

        #tmpfile = './testfiles/tmp.orb'
        tmpfile = os.path.join(jobdir, 'tmp.orb')
        
        jygen(tmpfile, 7, 0.01, 2, 60, 'Si', False)

        nao = read_nao(tmpfile)
        self.assertEqual(nao['elem'], 'Si')
        self.assertEqual(nao['rcut'], 7.0)
        self.assertEqual(nao['ecut'], 60.0)
        self.assertEqual(nao['dr'], 0.01)
        self.assertEqual(nao['nr'], 701)

        os.remove(tmpfile)

        #jygen(None, 10, 0.01, 2, 100, 'Si', True)

if __name__ == '__main__':
    unittest.main()


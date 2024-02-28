'''
This script contains functions to generate ABACUS STRU files from a "STRU" dict, and vice versa.

A "STRU" dict contains the following key-value pairs:

'lat' : dict
    Parameters to determine the lattice vectors. The associated value is a dict with the following key-value pairs:
    'const' : float
        Lattice constant in Bohr.
    'vec' : list of list of float, optional
        Lattice vectors in unit of stru['lat']['const']. Should not be present if 'latname' is specified in INPUT.
    'param' : list of float, optional
        Extra lattice parameters used when latname is specified in INPUT. See reference for details.
'coord_type' : str
    Coordinate type in ATOMIC_POSITIONS, could be 'Direct', 'Cartesian', 'Cartesian_angstrom' and several more.
    See reference for details.
'species' : list of dict
    A list of atomic species, each of which is a dict containing the following key-value pairs:
    'symbol' : str
        Element symbol.
    'mass' : float
        Mass of the atomic species.
    'pp_file' : str
        Name of the pseudopotential file.
    'pp_type' : str, optional
        Type of the pseudopotential file. See reference for details.
    'orb_file' : str, optional
        Name of the numerical atomic orbital file. Should not be present in PW calculations.
    'natom' : int
        Number of atoms of this species.
    'mag_each' : float
        Magnetic moment of each atom.
    'atoms' : list of dict
        A list of atoms of this species, each of which is a dict containing the following key-value pairs:
        'coord' : list of float
            Coordinates of the atom. The type is determined by stru['coord_type'].
        'm' : list of int, optional
            Multiplication factor of the coordinates in MD or relaxation. Each int must be 0 or 1.
        'v' : list of float, optional
            Initial velocity of the atom in MD or relaxation.
        'mag' : float|tuple, optional
            Magnetic moment of the atom. A float value implies collinear magnetism.
            If tuple, it could be either ('Cartesian', [mx, my, mz]) or ('Spherical', [mr, mpolar, mazimuth]).

Reference
---------
https://abacus.deepmodeling.com/en/latest/advanced/input_files/stru.html

'''
def write_stru(job_dir, stru, fname='STRU'):
    '''
    Generates a ABACUS STRU file from a STRU dict.

    Parameters
    ----------
    job_dir: str
        Directory in which the STRU file is generated.
    stru : dict
        Parameters to generate the STRU file. It contains the following key-value pairs:
    fname : str
        Name of the STRU file.

    '''
    with open(job_dir + '/' + fname, 'w') as f:

        #============ ATOMIC_SPECIES ============
        f.write('ATOMIC_SPECIES\n')
        width = { key + '_w' : max([len(str(s[key])) for s in stru['species'] ]) for key in ['symbol', 'mass', 'pp_file'] }
        for s in stru['species']:
            f.write('{symbol:<{symbol_w}}  {mass:>{mass_w}}  {pp_file:>{pp_file_w}}'.format(**s, **width))
            if 'pp_type' in s:
                f.write('  {}'.format(s['pp_type']))
            f.write('\n')

        #============ NUMERICAL_ORBITAL ============
        if 'orb_file' in stru['species'][0]:
            f.write('\nNUMERICAL_ORBITAL\n')
            for s in stru['species']:
                f.write('{}\n'.format(s['orb_file']))
        
        #============ LATTICE_CONSTANT/PARAMETER/VECTORS ============
        f.write('\nLATTICE_CONSTANT\n')
        f.write('{}\n'.format(stru['lat']['const']))

        if 'vec' in stru['lat']:
            f.write('\nLATTICE_VECTORS\n')
            for v in stru['lat']['vec']:
                f.write('{} {} {}\n'.format(v[0], v[1], v[2]))

        if 'param' in stru['lat']:
            f.write('\nLATTICE_PARAMETER\n')
            for param in stru['lat']['param']:
                f.write('{} '.format(param))
            f.write('\n')

        #============ ATOMIC_POSITIONS ============
        f.write('\nATOMIC_POSITIONS\n')
        f.write('{}\n'.format(stru['coord_type']))

        for s in stru['species']:
            f.write('\n{}\n'.format(s['symbol']))
            f.write('{}\n'.format(s['mag_each']))
            f.write('{}\n'.format(s['natom']))

            for atom in s['atom']:
                f.write('{} {} {}'.format(atom['coord'][0], atom['coord'][1], atom['coord'][2]))

                for key in ['m', 'v']: # frozen atom / initial velocity
                    if key in atom:
                        f.write(' {} {} {} {}'.format(key, atom[key][0], atom[key][1], atom[key][2]))

                if 'mag' in atom:
                    if not isinstance(atom['mag'], tuple): # collinear
                        f.write(' mag {}'.format(atom['mag']))
                    else: # non-collinear
                        mag_coord_type, mag = atom['mag']
                        assert mag_coord_type in ['Cartesian', 'Spherical']
                        if mag_coord_type == 'Cartesian':
                            f.write(' mag {} {} {}'.format(mag[0], mag[1], mag[2]))
                        else:
                            f.write(' mag {} angle1 {} angle2 {}'.format(mag[0], mag[1], mag[2]))

                f.write('\n')


def _parse_coordinate_line(line):
    '''
    Parses a coordinate line (which may include extra parameters) in the ATOMIC_POSITIONS block.

    A coordinate line always contains the x, y, z coordinates of an atom, and may also include
        - whether an atom is frozen in MD or relaxation
        - initial velocity of an atom in MD or relaxation
        - magnetic moment of an atom

    See https://abacus.deepmodeling.com/en/latest/advanced/input_files/stru.html#More-Key-Words
    for details.

    '''
    fields = line.split()
    result = { 'coord' : [float(x) for x in fields[0:3]] }

    idx = 3
    while idx < len(fields):
        if fields[idx].isdigit(): # no keyword, 0/1 -> frozen atom
            result['m'] = [int(x) for x in fields[idx:idx+3]]
            idx += 3
        elif fields[idx] == 'm': # frozen atom
            result['m'] = [int(x) for x in fields[idx+1:idx+4]]
            idx += 4
        elif fields[idx] in ['v', 'vel', 'velocity']: # initial velocity
            result['v'] = [float(x) for x in fields[idx+1:idx+4]]
            idx += 4
        elif fields[idx] in ['mag', 'magmom']:
            '''
            here we assume that frozen atom info cannot be placed after a collinear mag info without a keyword
            i.e., the following coordinate line
                0.0 0.0 0.0 mag 1.0 0 0 0
            is not allowed; one must explicitly specify 'm' in this case:
                0.0 0.0 0.0 mag 1.0 m 0 0 0

            '''
            if idx + 2 < len(fields) and fields[idx+2] == 'angle1':
                result['mag'] = ('Spherical', [float(fields[idx+1]), float(fields[idx+3]), float(fields[idx+5])])
                idx += 6
            elif idx + 2 < len(fields) and fields[idx+2][0].isdigit():
                result['mag'] = ('Cartesian', [float(fields[idx+1]), float(fields[idx+2]), float(fields[idx+3])])
                idx += 4
            else: # collinear
                result['mag'] = float(fields[idx+1])
                idx += 2
        else:
            raise ValueError('Error: unknown keyword %s'%fields[idx])

    return result


def _atomic_positions_gen(lines):
    '''
    Iteratively generates info per species from the ATOMIC_POSITIONS block.

    '''
    natom = int(lines[2])
    yield { 'symbol': lines[0], 'mag_each': float(lines[1]), 'natom': natom, \
            'atom': [ _parse_coordinate_line(line) for line in lines[3:3+natom] ] }
    if len(lines) > 3 + natom:
        yield from _atomic_positions_gen(lines[3+natom:])


def read_stru(fpath):
    '''
    Builds a STRU dict from a ABACUS STRU file.

    Returns
    -------
        A dict containing the following keys-value pairs:
        'species' : list of dict
            List of atomic species. Each dict contains 'symbol', 'mass', 'pp_file',
            and optionally 'pp_type'.
        
    '''
    block_title = ['ATOMIC_SPECIES', 'NUMERICAL_ORBITAL', 'LATTICE_CONSTANT', 'LATTICE_PARAMETER', \
            'LATTICE_VECTORS', 'ATOMIC_POSITIONS']

    _trim = lambda line: line.split('#')[0].split('//')[0].strip(' \t\n')
    with open(fpath, 'r') as f:
        lines = [_trim(line).replace('\t', ' ') for line in f.readlines() if len(_trim(line)) > 0]

    # break the content into blocks
    delim = [i for i, line in enumerate(lines) if line in block_title] + [len(lines)]
    blocks = { lines[delim[i]] : lines[delim[i]+1:delim[i+1]] for i in range(len(delim) - 1) }

    stru = {}
    #============ LATTICE_CONSTANT/PARAMETER/VECTORS ============
    stru['lat'] = {'const': float(blocks['LATTICE_CONSTANT'][0])}
    if 'LATTICE_VECTORS' in blocks:
        stru['lat']['vec'] = [[float(x) for x in line.split()] for line in blocks['LATTICE_VECTORS']]
    elif 'LATTICE_PARAMETER' in blocks:
        stru['lat']['param'] = [float(x) for x in blocks['LATTICE_PARAMETERS'].split()]

    #============ ATOMIC_SPECIES ============
    stru['species'] = [ dict(zip(['symbol', 'mass', 'pp_file', 'pp_type'], line.split())) for line in blocks['ATOMIC_SPECIES'] ]
    for s in stru['species']:
        s['mass'] = float(s['mass'])

    #============ NUMERICAL_ORBITAL ============
    if 'NUMERICAL_ORBITAL' in blocks:
        for i, s in enumerate(stru['species']):
            s['orb_file'] = blocks['NUMERICAL_ORBITAL'][i]

    #============ ATOMIC_POSITIONS ============
    stru['coord_type'] = blocks['ATOMIC_POSITIONS'][0]
    index = { s['symbol'] : i for i, s in enumerate(stru['species']) }
    for ap in _atomic_positions_gen(blocks['ATOMIC_POSITIONS'][1:]):
        stru['species'][index[ap['symbol']]].update(ap)

    return stru


############################################################
#                       Test
############################################################
import unittest
import os

class _TestStruIO(unittest.TestCase):

    def test_read_stru(self):
        stru = read_stru('./testfiles/STRU.test')

        self.assertEqual(stru['lat']['const'], 20.0)
        self.assertEqual(stru['lat']['vec'], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

        self.assertEqual(stru['coord_type'], 'Cartesian_angstrom')

        self.assertEqual(stru['species'][0]['symbol'], 'In')
        self.assertEqual(stru['species'][1]['mass'], 12.011)
        self.assertEqual(stru['species'][2]['pp_file'], 'O_ONCV_PBE-1.0.upf')
        self.assertEqual(stru['species'][2]['pp_type'], 'upf100')
        self.assertEqual(stru['species'][2]['orb_file'], 'O_gga_7au_100Ry.orb')

        self.assertEqual(stru['species'][0]['natom'], 2)
        self.assertEqual(stru['species'][1]['natom'], 1)
        self.assertEqual(stru['species'][2]['natom'], 3)

        self.assertEqual(stru['species'][0]['mag_each'], 0.0)
        self.assertEqual(stru['species'][1]['mag_each'], 0.0)
        self.assertEqual(stru['species'][2]['mag_each'], 1.0)

        self.assertEqual(stru['species'][0]['atom'][0]['coord'], [0.0, 0.0, 0.0])
        self.assertEqual(stru['species'][1]['atom'][0]['coord'], [3.0, 4.0, 5.0])
        self.assertEqual(stru['species'][2]['atom'][2]['coord'], [0.7, 0.8, 0.9])

        self.assertEqual(stru['species'][0]['atom'][1]['m'], [1, 1, 1])
        self.assertEqual(stru['species'][1]['atom'][0]['v'], [0.0, 0.0, -1.0])
        self.assertEqual(stru['species'][2]['atom'][2]['m'], [0, 0, 0])

        self.assertEqual(stru['species'][0]['atom'][0]['mag'], ('Spherical', [0.5, 72.0, 36.0]))
        self.assertEqual(stru['species'][0]['atom'][1]['mag'], ('Cartesian', [0.5, 0.5, 0.5]))
        self.assertEqual(stru['species'][2]['atom'][0]['mag'], 0.5)


    def test_write_stru(self):
        jobdir = './testfiles/'

        stru = {
                'lat': {
                    'const': 20.0,
                    'vec': [
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                        ],
                    },
                'coord_type': 'Cartesian_angstrom',
                'species': [
                    {
                        'symbol': 'In',
                        'mass': 114.818,
                        'pp_file': 'In_ONCV_PBE-1.0.upf',
                        'pp_type': 'upf201',
                        'orb_file': 'In_gga_7au_100Ry.orb',
                        'natom': 2,
                        'mag_each': 0.0,
                        'atom': [
                            {'coord': [0.0, 0.0, 0.0], 'v': [0.0, 0.0, -1.0], 'mag': ('Spherical', [0.5, 72.0, 36.0])},
                            {'coord': [0.0, 0.0, 1.0], 'm': [1, 1, 1], 'mag': ('Cartesian', [0.5, 0.5, 0.5])},
                            ]
                        },
                    {
                        'symbol': 'C',
                        'mass': 12.011,
                        'pp_file': 'C.LDA.upf',
                        'orb_file': 'C_gga_7au_100Ry.orb',
                        'natom': 1,
                        'mag_each': 0.0,
                        'atom': [
                            {'coord': [0.0, 0.0, 0.0], 'm': [0, 0, 0], 'v': [0.0, 0.0, -1.0], 'mag': 0.5},
                            ]
                        },
                    {
                        'symbol': 'O',
                        'mass': 15.999,
                        'pp_file': 'O_ONCV_PBE-1.0.upf',
                        'pp_type': 'upf100',
                        'orb_file': 'O_gga_7au_100Ry.orb',
                        'natom': 3,
                        'mag_each': 1.0,
                        'atom': [
                            {'coord': [0.0, 0.0, 0.0], 'mag': 0.5},
                            {'coord': [0.0, 0.0, 0.0], 'mag': ('Cartesian', [1.0, 1.0, 1.0]), 'v': [0.0, 0.0,  1.0], 'm': [0, 0, 0]},
                            {'coord': [0.0, 0.0, 0.0], 'v': [0.0, 0.0, -1.0]},
                            ]
                        },
                    ],
                }

        fstru = 'STRU.tmp'
        write_stru(jobdir, stru, fstru)

        stru2 = read_stru(jobdir + '/' + fstru)
        self.assertEqual(stru, stru2)

        os.remove(jobdir + '/' + fstru)



############################################################
#                       Main
############################################################
if __name__ == '__main__':
    unittest.main()


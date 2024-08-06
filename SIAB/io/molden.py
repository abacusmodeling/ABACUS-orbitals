def read_abacus_lowf(flowf, pat=r'^WFC_NAO_(K\d+|GAMMA(0|1)).txt$'):
    import re, os
    import numpy as np
    if not re.match(pat, os.path.basename(flowf)):
        return None
    
    with open(flowf, 'r') as file:
        lines = file.readlines()

    if lines[0].endswith('(index of k points)\n'):
        # discard the first two lines
        lines = lines[2:]
    # initialize lists
    occ = []
    band = []
    ener = []
    data = []

    # read nbands and nlocal
    i = 0
    nbands = int(lines[i].strip().split()[0])
    i += 1
    nlocal = int(lines[i].strip().split()[0])
    i += 1

    # loop over lines and process
    while i < len(lines):
        line = lines[i].strip()
        if '(band)' in line:
            band.append(int(line.split(' ')[0]))
        elif '(Ry)' in line:
            ener.append(float(line.split(' ')[0]))
        elif '(Occupations)' in line:
            occ.append(float(line.split(' ')[0]))
        else:
            data.extend([float(x) for x in line.split()])
        i += 1

    # check if the data we collected has the correct number of elements
    if "WFC_NAO_K" in flowf: # multi-k case, the coef will be complex
        data = [complex(data[i], data[i+1]) for i in range(0, len(data), 2)]
        data = [d.real for d in data]
    data = np.array(data).reshape(nbands, nlocal)
    if data.shape != (nbands, nlocal):
        print(f"nbands = {nbands}, nlocal = {nlocal}")
        print(f"data.shape = {data.shape}")
        raise ValueError("Data read from file is not consistent with expected size.")

    return nbands, nlocal, occ, band, ener, data

def write_molden_mo(coefs, occ, ener, spin=0, ndigits=3):
    """for writing Molecular Orbitals (MOs) in molden file required format."""
    import numpy as np
    out = "[5D7F]\n[9G]\n[MO]\n"
    def mo(ener, spin, occ, coef, ndigits=3):
        spin = "Alpha" if spin == 1 else "Beta"
        out  = f"Ene={ener:>20.10e}\n"
        out += f"Spin={spin:>6s}\n"
        out += f"Occup={occ:>12.7f}\n"
        for ic, c in enumerate(coef):
            if c >= 10**(-ndigits):
                out += f"{ic+1:5d} {c:>{ndigits+4+3}.{ndigits}e}\n"
        return out
    nbands, nlocal = np.array(coefs).shape
    assert nbands == len(occ) == len(ener)
    assert spin in [0, 1]
    for i in range(nbands):
        out += mo(ener[i], spin, occ[i], coefs[i], ndigits)
    return out

def read_abacus_stru(fstru):
    """this function benefit from the implementation by jinzx10"""
    return read_stru(fstru)

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

def write_molden_cell(const, vec):
    out = "[Cell]\n"
    const *= 0.529177249
    assert len(vec) == 3
    assert all(len(v) == 3 for v in vec)
    for i in range(3):
        out += f"{vec[i][0]:>15.10f}{vec[i][1]:>15.10f}{vec[i][2]:>15.10f}\n"
    return out

def write_molden_atoms(labels, kinds, labels_kinds_map, coords):
    from SIAB.data.interface import PERIODIC_TABLE_TOINDEX as ptable
    assert len(labels) == len(coords)
    natom = len(labels)
    lkm = labels_kinds_map # just for short notation
    out  = "[Atoms] AU\n"
    for i in range(natom):
        elem = kinds[lkm[i]]
        out += f"{elem:<2s}{i+1:>8d}{ptable[elem]:>8d}{coords[i][0]:>15.6f}{coords[i][1]:>15.6f}{coords[i][2]:>15.6f}\n"
    return out

def read_abacus_input(finput):
    import os, re
    assert os.path.basename(finput) == "INPUT"
    with open(finput, 'r') as file:
        lines = file.readlines()
    if lines[0] == "INPUT_PARAMETERS\n":
        lines = lines[1:]
    kvpat = r"^(\S+)\s*(\S+)\s*(^#.*$)?"
    kv = {}
    for line in lines:
        if line.startswith("#"):
            continue
        m = re.match(kvpat, line)
        if m:
            kv[m.group(1)] = m.group(2)
    return kv

def moldengen(folder: str):
    """generate molden file by reading the outdir of ABACUS, for only LCAO calculation!"""
    import os
    import numpy as np
    from SIAB.io.nao2gto import convert_nao_to_gto, GTORadials

    files = os.listdir(folder)
    assert ("STRU" in files) and ("INPUT" in files) and ("KPT" in files)
    cwd = os.path.abspath(os.getcwd())
    os.chdir(folder)
    ####################
    # write the header #
    ####################
    out = "[Molden Format]\n"

    ####################
    # write the cell   #
    ####################
    kv = read_abacus_input("INPUT")
    _temp = kv.get("stru_file", "STRU")
    stru = read_abacus_stru(_temp)
    out += write_molden_cell(stru['lat']['const'], stru['lat']['vec'])
    
    ####################
    # write the atoms  #
    ####################
    # molden requires coordinates in Bohr
    
    kinds = [spec['symbol'] for spec in stru['species']]
    labels_kinds_map = []
    for i, spec in enumerate(stru['species']):
        labels_kinds_map.extend([i]*spec['natom'])
    labels = [kinds[i] for i in labels_kinds_map]
    coords = [atom['coord'] for spec in stru['species'] for atom in spec['atom']]
    if stru['coord_type'] == "Cartesian": # not direct but in Bohr
        coords = coords
    elif stru['coord_type'] == "Direct": # in fractional coordinates
        vec = np.array(stru['lat']['vec'])
        coords = [np.dot(c, vec)*stru['lat']['const'] for c in coords]
    elif stru['coord_type'] == "Cartesian_Angstrom":
        coords = [c*0.529177249 for c in coords]
    else:
        raise NotImplementedError(f"Unknown coordinate type {stru['coord_type']}")
    coords = np.array(coords).reshape(-1, 3).tolist()
    out += write_molden_atoms(labels, kinds, labels_kinds_map, coords)

    ####################
    # write the basis  #
    ####################
    orbital_dir = kv.get("orbital_dir", "./")
    forbs = [os.path.join(orbital_dir, spec['orb_file']) for spec in stru["species"]]
    forbs = [os.path.abspath(forb) for forb in forbs]
    
    total_gto = GTORadials()
    for forb in forbs:
        gto = convert_nao_to_gto(forb)
        total_gto.NumericalRadials.append(gto.NumericalRadials[0])
        total_gto.symbols.append(gto.symbols[0])
    out += total_gto.molden()
    
    ####################
    # write the mo     #
    ####################
    suffix = kv.get("suffix", "ABACUS")
    out_dir = ".".join(["OUT", suffix])
    os.chdir(out_dir)
    mo_files = [f for f in os.listdir() if f.startswith("WFC_NAO_") and f.endswith(".txt")][0]
    print(f"Reading MOs from {mo_files}")
    nbands, nlocal, occ, band, ener, data_np = read_abacus_lowf(mo_files)
    coefs = data_np.tolist()
    out += write_molden_mo(coefs, occ, ener)

    os.chdir(cwd)
    with open("ABACUS.molden", "w") as file:
        file.write(out)
    return out

if __name__ == "__main__":
    moldengen("/root/documents/simulation/abacus/test_H2/")

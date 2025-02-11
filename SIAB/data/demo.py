'''this is just a demo file'''

class AtomSpecies:
    '''Class for atom species'''
    def __init__(self, 
                 elem,
                 mass,
                 fpsp,
                 forb,
                 ecutjy,
                 rcutjy,
                 lmax,
                 orbgen):
        import os
        self.elem = elem
        self.mass = mass
        if not os.path.exists(fpsp):
            raise FileNotFoundError(f'Pseudopotential file {fpsp} not found')
        self.fpsp = fpsp
        self.forb = forb
        self.ecutjy = ecutjy
        self.rcutjy = rcutjy
        self.lmax = lmax
        self.orbgen = orbgen
    
    def jygen(self):
        '''Generate jy orbital file, return the file path'''
        pass

class Cell:
    '''Class for cell'''
    def __init__(self,
                 atomspecies,
                 type_map,
                 cell,
                 coords):
        import numpy as np
        self.atomspecies = atomspecies
        self.type_map = type_map
        self.cell = np.array(cell)
        if self.cell.shape != (3, 3):
            raise ValueError('Cell should be a 3x3 matrix')
        self.coords = np.array(coords)
    
    @staticmethod
    def z2cart(zmat):
        '''convert the z-matrix to cartesian coordinates
        
        the zmat should be organized in the following way:
        zmat = [{},
                {'i': 0, 'bl': 1.2},
                {'i': 1, 'bl': 1.2, 'j': 0, 'angl': 90.0},
                {'i': 2, 'bl': 1.2, 'j': 1, 'angl': 90.0, 'k': 0, 'dihl': 180.0},
                {'i': 3, 'bl': 1.2, 'j': 1, 'angl': 90.0, 'k': 2, 'dihl': 180.0},
                ...]
        
        Parameters
        ----------
        zmat : list
            the z-matrix of the molecule, see the above example
        
        Returns
        -------
        coords : np.ndarray
            the cartesian coordinates of the molecule, in shape of (natom, 3)
        '''
        import numpy as np
        coords = []
        for idx, atom in enumerate(zmat):
            if idx == 0:
                coords.append([0.0, 0.0, 0.0])
            elif idx == 1:
                bl = atom['bl']
                coords.append([bl, 0.0, 0.0])
            elif idx == 2:
                i = atom['i']
                bl = atom['bl']
                angl = np.deg2rad(atom['angl'])
                x = bl * np.cos(angl)
                y = bl * np.sin(angl)
                coords.append([x, y, 0.0])
            else:
                i = atom['i']
                j = atom['j']
                k = atom['k']
                bl = atom['bl']
                angl = np.deg2rad(atom['angl'])
                dihl = np.deg2rad(atom['dihl'])

                v1 = np.array(coords[i])
                v2 = np.array(coords[j])
                v3 = np.array(coords[k])

                v1v2 = v1 - v2
                v2v3 = v2 - v3

                n1 = np.cross(v1v2, v2v3)
                n1 /= np.linalg.norm(n1)

                n2 = np.cross(n1, v2v3)
                n2 /= np.linalg.norm(n2)

                d = bl * np.cos(angl)
                h = bl * np.sin(angl)

                v4 = v3 + d * (v2v3 / np.linalg.norm(v2v3)) + h * (np.cos(dihl) * n1 + np.sin(dihl) * n2)
                coords.append(v4.tolist())

        return np.array(coords)
    
    @staticmethod
    def from_simple_molecule(atom, lat, shape, bl):
        '''
        build an instance of Cell from a molecule
        
        Parameters
        ----------
        atom : AtomSpecies
            the atom species, see the above AtomSpecies class
        lat : float
            the lattice constant, in Angstrom
        shape : str
            the shape of the molecule, one of the following:
            monomer, dimer, trimer, tetrahedron, square, triangular_bipyramid, octahedron, cube
        bl : float
            the bond length, in Angstrom
            
        Returns
        -------
        cell : Cell
            the instance of Cell
        '''
        import numpy as np
        def dimer(bl):
            return np.array([[0.0, 0.0, 0.0], 
                             [0.0, 0.0, bl]])
        def trimer(bl):
            return np.array([[0.0, 0.0, 0.0], 
                             [0.0, 0.0, bl], 
                             [0.0, bl*0.86603, bl*0.5]])
        def tetrahedron(bl):
            return np.array([[0.0, 0.0, 0.0], 
                             [0.0, 0.0, bl], 
                             [0.0, bl*0.86603, bl*0.5], 
                             [bl*0.81649, bl*0.28867, bl*0.5]])
        def square(bl):
            return np.array([[0.0, 0.0, 0.0], 
                             [0.0, 0.0, bl], 
                             [bl, 0.0, 0.0], 
                             [bl, 0.0, bl]])
        def triangular_bipyramid(bl):
            return np.array([[bl/1.73205, 0.0, 0.0], 
                             [-bl/1.73205/2, bl/2, 0.0], 
                             [-bl/1.73205/2, -bl/2, 0.0], 
                             [0.0, 0.0, bl*(2/3)**(1/2)], 
                             [0.0, 0.0, -bl*(2/3)**(1/2)]])
        def octahedron(bl):
            return np.array([[bl/2, bl/2, 0.0], 
                             [-bl/2, -bl/2, 0.0], 
                             [bl/2, -bl/2, 0.0], 
                             [-bl/2, bl/2, 0.0], 
                             [0.0, 0.0, bl/2**(1/2)], 
                             [0.0, 0.0, -bl/2**(1/2)]])
        def cube(bl):
            return np.array([[bl/2, bl/2, bl/2], 
                             [-bl/2, -bl/2, bl/2], 
                             [bl/2, -bl/2, bl/2], 
                             [-bl/2, bl/2, bl/2], 
                             [bl/2, bl/2, -bl/2], 
                             [-bl/2, -bl/2, -bl/2], 
                             [bl/2, -bl/2, -bl/2], 
                             [-bl/2, bl/2, -bl/2]])
        
        builder = {'monomer': dimer, 'dimer': dimer, 'trimer': trimer, 'tetrahedron': tetrahedron,
                   'square': square, 'triangular_bipyramid': triangular_bipyramid, 
                   'octahedron': octahedron, 'cube': cube}        
        coords=builder[shape](bl)
        return Cell(atomspecies=[atom], 
                    type_map=[0]*len(coords), 
                    cell=np.eye(3) * lat, 
                    coords=coords)
    
    def write(self, fn, fmt='abacus'):
        '''
        '''
        if fmt == 'abacus':
            with open(fn, 'w') as f:
                f.write(self._write_core_abacus())
        else:
            raise NotImplementedError(f'Format {fmt} not supported')
        return self
    
    def run(self, dftparam):
        '''
        
        '''
    
    def _write_core_abacus(self, orbital_dir):
        '''
        write the information of cell to abacus input file
        
        Returns
        -------
        out : str
            the content of the abacus input file
        '''
        import numpy as np
        out = ''
        out += 'ATOMIC_SPECIES\n'

        assert all([atom.forb is not None or atom.orbgen for atom in self.atomspecies]), \
            'illegal atom species configuration for `forb` and `orbgen`'
        for a in self.atomspecies:
            out += f'{a.elem} {a.mass} {a.fpsp}\n'
        out += '\n'
        if any([atom.forb for atom in self.atomspecies]):
            out += 'NUMERICAL_ORBITAL\n'
            for a in self.atomspecies:
                if a.forb is None:
                    a.forb = a.jygen(orbital_dir)
                out += f'{a.forb}\n'
            out += '\n'
        out += 'LATTICE_CONSTANT\n1.8897259886\n' # 1 Angstrom in a.u.
        out += 'LATTICE_VECTORS\n'
        for i in range(3):
            out += ' '.join([str(x) for x in self.cell[i]]) + '\n'
        out += 'ATOMIC_POSITIONS\nCartesian_angstrom_center_xyz\n'
        
        idx = np.argsort(self.type_map)
        
    
class CellGenerator:
    '''Class for generating Cell'''
    def __init__(self, 
                 proto,
                 atomspecies,
                 types,
                 type_map,
                 dftparam,
                 compparam):
        self.proto = proto
        self.atomspecies = atomspecies
        self.types = types
        self.type_map = type_map
        self.dftparam = dftparam
        self.compparam = compparam
        # hidden variables
        self.cells = []
    
    def run(self):
        '''run dft calculation, return the jobdir'''
        out = []
        
        return out
    
    def perturb(self, pertkind, pertmags):
        '''configure a perturbation task'''
        self.cells.append([self._stretch(self.proto, pertkind, m) for m in pertmags])
        return self
    
    def _stretch(self):
        
        return self

##############################################
#         input files preparation            #
##############################################
def monomer(element, mass, fpseudo, lattice_constant, nspin, forb):
    """generate monomer structure"""
    shift = lattice_constant/2/1.8897259886
    starting_magnetization = 0.0 if nspin == 1 else 2.0
    result = "ATOMIC_SPECIES\n%s %.6f %s\n"%(element, mass, fpseudo)
    if forb is not None:
        result += "\nNUMERICAL_ORBITAL\n%s\n\n"%forb
    result += "LATTICE_CONSTANT\n%.6f  // add lattice constant(a.u.)\n"%lattice_constant
    result += "LATTICE_VECTORS\n"
    result += "%10.8f %10.8f %10.8f\n"%(1.0, 0.0, 0.0)
    result += "%10.8f %10.8f %10.8f\n"%(0.0, 1.0, 0.0)
    result += "%10.8f %10.8f %10.8f\n"%(0.0, 0.0, 1.0)
    result += "ATOMIC_POSITIONS\nCartesian_angstrom_center_xyz  //Cartesian or Direct coordinate.\n"
    result += "%s      //Element Label\n"%element
    result += "%.2f     //starting magnetism\n"%starting_magnetization
    result += "1       //number of atoms\n"
    result += "%10.8f %10.8f %10.8f 0 0 0\n"%(0.0 + shift, 0.0 + shift, 0.0 + shift)
    return result

def dimer(element, mass, fpseudo, lattice_constant, bond_length, nspin, forb = None):
    """generate dimer structure"""
    shift = lattice_constant/2/1.8897259886
    starting_magnetization = 0.0 if nspin == 1 else 2.0
    result = "ATOMIC_SPECIES\n%s %.6f %s\n"%(element, mass, fpseudo)
    if forb is not None:
        result += "\nNUMERICAL_ORBITAL\n%s\n\n"%forb
    result += "LATTICE_CONSTANT\n%.6f  // add lattice constant(a.u.)\n"%lattice_constant
    result += "LATTICE_VECTORS\n"
    result += "%10.8f %10.8f %10.8f\n"%(1.0, 0.0, 0.0)
    result += "%10.8f %10.8f %10.8f\n"%(0.0, 1.0, 0.0)
    result += "%10.8f %10.8f %10.8f\n"%(0.0, 0.0, 1.0)
    result += "ATOMIC_POSITIONS\nCartesian_angstrom_center_xyz  //Cartesian or Direct coordinate.\n"
    result += "%s      //Element Label\n"%element
    result += "%.2f     //starting magnetism\n"%starting_magnetization
    result += "2       //number of atoms\n"
    result += "%10.8f %10.8f %10.8f 0 0 0\n"%(0.0 + shift, 0.0 + shift, 0.0 + shift)
    result += "%10.8f %10.8f %10.8f 0 0 0\n"%(0.0 + shift, 0.0 + shift, bond_length + shift)
    return result

def trimer(element, mass, fpseudo, lattice_constant, bond_length, nspin, forb = None):
    """generate trimer structure"""
    shift = lattice_constant/2/1.8897259886
    starting_magnetization = 0.0 if nspin == 1 else 2.0
    dis1 = bond_length * 0.86603
    dis2 = bond_length * 0.5
    result = "ATOMIC_SPECIES\n%s %.6f %s\n"%(element, mass, fpseudo)
    if forb is not None:
        result += "\nNUMERICAL_ORBITAL\n%s\n\n"%forb
    result += "LATTICE_CONSTANT\n%.6f  // add lattice constant(a.u.)\n"%lattice_constant
    result += "LATTICE_VECTORS\n"
    result += "%10.8f %10.8f %10.8f\n"%(1.0, 0.0, 0.0)
    result += "%10.8f %10.8f %10.8f\n"%(0.0, 1.0, 0.0)
    result += "%10.8f %10.8f %10.8f\n"%(0.0, 0.0, 1.0)
    result += "ATOMIC_POSITIONS\nCartesian_angstrom_center_xyz  //Cartesian or Direct coordinate.\n"
    result += "%s      //Element Label\n"%element
    result += "%.2f     //starting magnetism\n"%starting_magnetization
    result += "3       //number of atoms\n"
    result += "%10.8f %10.8f %10.8f 0 0 0\n"%(0.0 + shift, 0.0 + shift, 0.0 + shift)
    result += "%10.8f %10.8f %10.8f 0 0 0\n"%(0.0 + shift, 0.0 + shift, bond_length + shift)
    result += "%10.8f %10.8f %10.8f 0 0 0\n"%(0.0 + shift, dis1 + shift, dis2 + shift)
    return result

def tetrahedron(element, mass, fpseudo, lattice_constant, bond_length, nspin, forb = None):
    """generate tetrahedron structure"""
    shift = lattice_constant/2/1.8897259886
    starting_magnetization = 0.0 if nspin == 1 else 2.0
    dis1 = bond_length * 0.86603
    dis2 = bond_length * 0.5
    dis3 = bond_length * 0.81649
    dis4 = bond_length * 0.28867
    result = "ATOMIC_SPECIES\n%s %.6f %s\n"%(element, mass, fpseudo)
    if forb is not None:
        result += "\nNUMERICAL_ORBITAL\n%s\n\n"%forb
    result += "LATTICE_CONSTANT\n%.6f  // add lattice constant(a.u.)\n"%lattice_constant
    result += "LATTICE_VECTORS\n"
    result += "%10.8f %10.8f %10.8f\n"%(1.0, 0.0, 0.0)
    result += "%10.8f %10.8f %10.8f\n"%(0.0, 1.0, 0.0)
    result += "%10.8f %10.8f %10.8f\n"%(0.0, 0.0, 1.0)
    result += "ATOMIC_POSITIONS\nCartesian_angstrom_center_xyz  //Cartesian or Direct coordinate.\n"
    result += "%s      //Element Label\n"%element
    result += "%.2f     //starting magnetism\n"%starting_magnetization
    result += "4       //number of atoms\n"
    result += "%10.8f %10.8f %10.8f 0 0 0\n"%(0.0 + shift, 0.0 + shift, 0.0 + shift)
    result += "%10.8f %10.8f %10.8f 0 0 0\n"%(0.0 + shift, 0.0 + shift, bond_length + shift)
    result += "%10.8f %10.8f %10.8f 0 0 0\n"%(0.0 + shift, dis1 + shift, dis2 + shift)
    result += "%10.8f %10.8f %10.8f 0 0 0\n"%(dis3 + shift, dis4 + shift, dis2 + shift)
    return result

def square(element, mass, fpseudo, lattice_constant, bond_length, nspin, forb = None):
    """generate square structure"""
    shift = lattice_constant/2/1.8897259886
    starting_magnetization = 0.0 if nspin == 1 else 2.0
    result = "ATOMIC_SPECIES\n%s %.6f %s\n"%(element, mass, fpseudo)
    if forb is not None:
        result += "\nNUMERICAL_ORBITAL\n%s\n\n"%forb
    result += "LATTICE_CONSTANT\n%.6f  // add lattice constant(a.u.)\n"%lattice_constant
    result += "LATTICE_VECTORS\n"
    result += "%10.8f %10.8f %10.8f\n"%(1.0, 0.0, 0.0)
    result += "%10.8f %10.8f %10.8f\n"%(0.0, 1.0, 0.0)
    result += "%10.8f %10.8f %10.8f\n"%(0.0, 0.0, 1.0)
    result += "ATOMIC_POSITIONS\nCartesian_angstrom_center_xyz  //Cartesian or Direct coordinate.\n"
    result += "%s      //Element Label\n"%element
    result += "%.2f     //starting magnetism\n"%starting_magnetization
    result += "4       //number of atoms\n"
    result += "%10.8f %10.8f %10.8f 0 0 0\n"%(0.0, 0.0, 0.0)
    result += "%10.8f %10.8f %10.8f 0 0 0\n"%(0.0, 0.0, bond_length)
    result += "%10.8f %10.8f %10.8f 0 0 0\n"%(bond_length + shift, 0.0 + shift, 0.0 + shift)
    result += "%10.8f %10.8f %10.8f 0 0 0\n"%(bond_length + shift, 0.0 + shift, bond_length + shift)
    return result

def triangular_bipyramid(element, mass, fpseudo, lattice_constant, bond_length, nspin, forb = None):
    """generate triangular bipyramid structure"""
    shift = lattice_constant/2/1.8897259886
    starting_magnetization = 0.0 if nspin == 1 else 2.0
    result = "ATOMIC_SPECIES\n%s %.6f %s\n"%(element, mass, fpseudo)
    if forb is not None:
        result += "\nNUMERICAL_ORBITAL\n%s\n\n"%forb
    result += "LATTICE_CONSTANT\n%.6f  // add lattice constant(a.u.)\n"%lattice_constant
    result += "LATTICE_VECTORS\n"
    result += "%10.8f %10.8f %10.8f\n"%(1.0, 0.0, 0.0)
    result += "%10.8f %10.8f %10.8f\n"%(0.0, 1.0, 0.0)
    result += "%10.8f %10.8f %10.8f\n"%(0.0, 0.0, 1.0)
    result += "ATOMIC_POSITIONS\nCartesian_angstrom_center_xyz  //Cartesian or Direct coordinate.\n"
    result += "%s      //Element Label\n"%element
    result += "%.2f     //starting magnetism\n"%starting_magnetization
    result += "5       //number of atoms\n"
    result += "%10.8f %10.8f %10.8f 0 0 0\n"%(bond_length / 1.73205 + shift, 0.0 + shift, 0.0 + shift)
    result += "%10.8f %10.8f %10.8f 0 0 0\n"%(-bond_length / 1.73205 / 2 + shift, bond_length / 2 + shift, 0.0 + shift)
    result += "%10.8f %10.8f %10.8f 0 0 0\n"%(-bond_length / 1.73205 / 2 + shift, -bond_length / 2 + shift, 0.0 + shift)
    result += "%10.8f %10.8f %10.8f 0 0 0\n"%(0.0 + shift, 0.0 + shift, bond_length * (2/3)**(1/2) + shift)
    result += "%10.8f %10.8f %10.8f 0 0 0\n"%(0.0 + shift, 0.0 + shift, -bond_length * (2/3)**(1/2) + shift)
    return result

def octahedron(element, mass, fpseudo, lattice_constant, bond_length, nspin, forb = None):
    """generate octahedron structure"""
    shift = lattice_constant/2/1.8897259886
    starting_magnetization = 0.0 if nspin == 1 else 2.0
    result = "ATOMIC_SPECIES\n%s %.6f %s\n"%(element, mass, fpseudo)
    if forb is not None:
        result += "\nNUMERICAL_ORBITAL\n%s\n\n"%forb
    result += "LATTICE_CONSTANT\n%.6f  // add lattice constant(a.u.)\n"%lattice_constant
    result += "LATTICE_VECTORS\n"
    result += "%10.8f %10.8f %10.8f\n"%(1.0, 0.0, 0.0)
    result += "%10.8f %10.8f %10.8f\n"%(0.0, 1.0, 0.0)
    result += "%10.8f %10.8f %10.8f\n"%(0.0, 0.0, 1.0)
    result += "ATOMIC_POSITIONS\nCartesian_angstrom_center_xyz  //Cartesian or Direct coordinate.\n"
    result += "%s      //Element Label\n"%element
    result += "%.2f     //starting magnetism\n"%starting_magnetization
    result += "6       //number of atoms\n"
    result += "%10.8f %10.8f %10.8f 0 0 0\n"%(bond_length / 2 + shift, bond_length / 2 + shift, 0.0 + shift)
    result += "%10.8f %10.8f %10.8f 0 0 0\n"%(-bond_length / 2 + shift, -bond_length / 2 + shift, 0.0 + shift)
    result += "%10.8f %10.8f %10.8f 0 0 0\n"%(bond_length / 2 + shift, -bond_length / 2 + shift, 0.0 + shift)
    result += "%10.8f %10.8f %10.8f 0 0 0\n"%(-bond_length / 2 + shift, bond_length / 2 + shift, 0.0 + shift)
    result += "%10.8f %10.8f %10.8f 0 0 0\n"%(0.0 + shift, 0.0 + shift, bond_length / 2**(1/2) + shift)
    result += "%10.8f %10.8f %10.8f 0 0 0\n"%(0.0 + shift, 0.0 + shift, -bond_length / 2**(1/2) + shift)
    return result

def cube(element, mass, fpseudo, lattice_constant, bond_length, nspin, forb = None):
    """generate cube structure"""
    shift = lattice_constant/2/1.8897259886
    starting_magnetization = 0.0 if nspin == 1 else 2.0
    result = "ATOMIC_SPECIES\n%s %.6f %s\n"%(element, mass, fpseudo)
    if forb is not None:
        result += "\nNUMERICAL_ORBITAL\n%s\n\n"%forb
    result += "LATTICE_CONSTANT\n%.6f  // add lattice constant(a.u.)\n"%lattice_constant
    result += "LATTICE_VECTORS\n"
    result += "%10.8f %10.8f %10.8f\n"%(1.0, 0.0, 0.0)
    result += "%10.8f %10.8f %10.8f\n"%(0.0, 1.0, 0.0)
    result += "%10.8f %10.8f %10.8f\n"%(0.0, 0.0, 1.0)
    result += "ATOMIC_POSITIONS\nCartesian_angstrom_center_xyz  //Cartesian or Direct coordinate.\n"
    result += "%s      //Element Label\n"%element
    result += "%.2f     //starting magnetism\n"%starting_magnetization
    result += "8       //number of atoms\n"
    result += "%10.8f %10.8f %10.8f 0 0 0\n"%(bond_length / 2 + shift, bond_length / 2 + shift, bond_length / 2 + shift)
    result += "%10.8f %10.8f %10.8f 0 0 0\n"%(-bond_length / 2 + shift, -bond_length / 2 + shift, bond_length / 2 + shift)
    result += "%10.8f %10.8f %10.8f 0 0 0\n"%(bond_length / 2 + shift, -bond_length / 2 + shift, bond_length / 2 + shift)
    result += "%10.8f %10.8f %10.8f 0 0 0\n"%(-bond_length / 2 + shift, bond_length / 2 + shift, bond_length / 2 + shift)
    result += "%10.8f %10.8f %10.8f 0 0 0\n"%(bond_length / 2 + shift, bond_length / 2 + shift, -bond_length / 2 + shift)
    result += "%10.8f %10.8f %10.8f 0 0 0\n"%(-bond_length / 2 + shift, -bond_length / 2 + shift, -bond_length / 2 + shift)
    result += "%10.8f %10.8f %10.8f 0 0 0\n"%(bond_length / 2 + shift, -bond_length / 2 + shift, -bond_length / 2 + shift)
    result += "%10.8f %10.8f %10.8f 0 0 0\n"%(-bond_length / 2 + shift, bond_length / 2 + shift, -bond_length / 2 + shift)
    return result

class AtomSpecies:
    '''a simple base class to store the information of atomic species'''
    def __init__(self, 
                 elem, 
                 **kwargs):
        '''
        instantiate the class
        
        Parameters
        ----------
        elem: str
            element symbol
        fpseudo: str
            the pseudopotential file, it is only safe to assign as an absolute path
        mass : float
            the atomic mass of the element in amu
        '''
        self.elem = elem
        self.mass = kwargs.get('mass', 1.0)

    def __dict__(self):
        return {'elem': self.elem, 'mass': self.mass}
    
    def __eq__(self, other):
        return self.__dict__() == other.__dict__()

class OrbgenSpecies(AtomSpecies):
    '''a class to store the information of atomic species for orbgen
    -specific tasks'''
    def __init__(self, 
                 elem, 
                 fpseudo,
                 **kwargs):
        '''
        instantiate the class
        
        Parameters
        ----------
        elem: str
            element symbol
        fpseudo: str
            the pseudopotential file, it is only safe to assign as an absolute path
        mass : float
            the atomic mass of the element in amu
        '''
        import os
        super().__init__(elem, **kwargs)
        if os.path.exists(fpseudo):
            self.fpseudo = os.path.abspath(fpseudo) 
            # DFT calculation-specific and compulsory parameters
        else:
            # otherwise throw
            raise FileNotFoundError("The pseudopotential file does not exist.")
        
        self.forb = kwargs.get('forb')
        self.ecutjy = kwargs.get('ecutjy')
        self.rcutjy = kwargs.get('rcutjy')
        
    def jygen(self, outdir, **kwargs):
        '''generate the jy basis file, return the file path
        
        Parameters
        ----------
        outdir: str
            the output directory
        l : int
            the angular momentum quantum number, exclusive with lmax
        lmax : int
            the maximum angular momentum quantum number, exclusive with l
        '''
        from SIAB.io.convention import orb
        
        if 'l' in kwargs and 'lmax' in kwargs:
            raise ValueError('l and lmax are exclusive, only specify one of them')
        
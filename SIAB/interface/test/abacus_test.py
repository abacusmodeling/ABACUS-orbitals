import unittest
import SIAB.interface.abacus as sia
import numpy as np
from scipy.optimize import curve_fit

def morse(r, De, a, re, e0=0.0):
    return De * (1.0 - np.exp(-a*(r-re)))**2.0 + e0

class TestAbacus(unittest.TestCase):

    def test_dimer(self):

        dimer = sia.dimer(element="Si",
                         mass=28.085,
                         fpseudo="Si.pbe-n-kjpaw_psl.1.0.0.UPF",
                         lattice_constant=10.263,
                         bond_length=2.35,
                         nspin=1)
        lines = dimer.split("\n")
        self.assertEqual(lines[0], "ATOMIC_SPECIES")
        self.assertEqual(lines[1], "Si 28.085000 Si.pbe-n-kjpaw_psl.1.0.0.UPF")
        self.assertEqual(lines[2], "LATTICE_CONSTANT")
        self.assertEqual(lines[3], "10.263000  // add lattice constant(a.u.)")
        self.assertEqual(lines[4], "LATTICE_VECTORS")
        self.assertEqual(lines[5], "1.00000000 0.00000000 0.00000000")
        self.assertEqual(lines[6], "0.00000000 1.00000000 0.00000000")
        self.assertEqual(lines[7], "0.00000000 0.00000000 1.00000000")
        self.assertEqual(lines[8], "ATOMIC_POSITIONS")
        self.assertEqual(lines[9], "Cartesian_angstrom  //Cartesian or Direct coordinate.")
        self.assertEqual(lines[10], "Si      //Element Label")
        self.assertEqual(lines[11], "0.00     //starting magnetism")
        self.assertEqual(lines[12], "2       //number of atoms")
        self.assertEqual(lines[13], "0.00000000 0.00000000 0.00000000 0 0 0")
        self.assertEqual(lines[14], "0.00000000 0.00000000 2.35000000 0 0 0")
        self.assertEqual(lines[15], "")

    def test_trimer(self):
            
        trimer = sia.trimer(element="Si",
                        mass=28.085,
                        fpseudo="Si.pbe-n-kjpaw_psl.1.0.0.UPF",
                        lattice_constant=10.263,
                        bond_length=2.35,
                        nspin=1)
        lines = trimer.split("\n")
        self.assertEqual(lines[0], "ATOMIC_SPECIES")
        self.assertEqual(lines[1], "Si 28.085000 Si.pbe-n-kjpaw_psl.1.0.0.UPF")
        self.assertEqual(lines[2], "LATTICE_CONSTANT")
        self.assertEqual(lines[3], "10.263000  // add lattice constant(a.u.)")
        self.assertEqual(lines[4], "LATTICE_VECTORS")
        self.assertEqual(lines[5], "1.00000000 0.00000000 0.00000000")
        self.assertEqual(lines[6], "0.00000000 1.00000000 0.00000000")
        self.assertEqual(lines[7], "0.00000000 0.00000000 1.00000000")
        self.assertEqual(lines[8], "ATOMIC_POSITIONS")
        self.assertEqual(lines[9], "Cartesian_angstrom  //Cartesian or Direct coordinate.")
        self.assertEqual(lines[10], "Si      //Element Label")
        self.assertEqual(lines[11], "0.00     //starting magnetism")
        self.assertEqual(lines[12], "3       //number of atoms")
        self.assertEqual(lines[13], "0.00000000 0.00000000 0.00000000 0 0 0")
        self.assertEqual(lines[14], "0.00000000 0.00000000 2.35000000 0 0 0")
        self.assertEqual(lines[15], "0.00000000 2.03517050 1.17500000 0 0 0")
        self.assertEqual(lines[16], "")

    def test_tetramer(self):

        tetramer = sia.tetramer(element="Si",
                            mass=28.085,
                            fpseudo="Si.pbe-n-kjpaw_psl.1.0.0.UPF",
                            lattice_constant=10.263,
                            bond_length=2.35,
                            nspin=1)
        lines = tetramer.split("\n")
        self.assertEqual(lines[0], "ATOMIC_SPECIES")
        self.assertEqual(lines[1], "Si 28.085000 Si.pbe-n-kjpaw_psl.1.0.0.UPF")
        self.assertEqual(lines[2], "LATTICE_CONSTANT")
        self.assertEqual(lines[3], "10.263000  // add lattice constant(a.u.)")
        self.assertEqual(lines[4], "LATTICE_VECTORS")
        self.assertEqual(lines[5], "1.00000000 0.00000000 0.00000000")
        self.assertEqual(lines[6], "0.00000000 1.00000000 0.00000000")
        self.assertEqual(lines[7], "0.00000000 0.00000000 1.00000000")
        self.assertEqual(lines[8], "ATOMIC_POSITIONS")
        self.assertEqual(lines[9], "Cartesian_angstrom  //Cartesian or Direct coordinate.")
        self.assertEqual(lines[10], "Si      //Element Label")
        self.assertEqual(lines[11], "0.00     //starting magnetism")
        self.assertEqual(lines[12], "4       //number of atoms")
        self.assertEqual(lines[13], "0.00000000 0.00000000 0.00000000 0 0 0")
        self.assertEqual(lines[14], "0.00000000 0.00000000 2.35000000 0 0 0")
        self.assertEqual(lines[15], "0.00000000 2.03517050 1.17500000 0 0 0")
        self.assertEqual(lines[16], "1.91875150 0.67837450 1.17500000 0 0 0")
        self.assertEqual(lines[17], "")    

    def test_STRU(self):
        dimer, natom = sia.STRU(shape="dimer",
                               element="Si",
                               mass=28.085,
                               fpseudo="Si.pbe-n-kjpaw_psl.1.0.0.UPF",
                               lattice_constant=10.263,
                               bond_length=2.35,
                               nspin=1)
        self.assertEqual(natom, 2)
        lines = dimer.split("\n")
        self.assertEqual(lines[0], "ATOMIC_SPECIES")
        self.assertEqual(lines[1], "Si 28.085000 Si.pbe-n-kjpaw_psl.1.0.0.UPF")
        self.assertEqual(lines[2], "LATTICE_CONSTANT")
        self.assertEqual(lines[3], "10.263000  // add lattice constant(a.u.)")
        self.assertEqual(lines[4], "LATTICE_VECTORS")
        self.assertEqual(lines[5], "1.00000000 0.00000000 0.00000000")
        self.assertEqual(lines[6], "0.00000000 1.00000000 0.00000000")
        self.assertEqual(lines[7], "0.00000000 0.00000000 1.00000000")
        self.assertEqual(lines[8], "ATOMIC_POSITIONS")
        self.assertEqual(lines[9], "Cartesian_angstrom  //Cartesian or Direct coordinate.")
        self.assertEqual(lines[10], "Si      //Element Label")
        self.assertEqual(lines[11], "0.00     //starting magnetism")
        self.assertEqual(lines[12], "2       //number of atoms")
        self.assertEqual(lines[13], "0.00000000 0.00000000 0.00000000 0 0 0")
        self.assertEqual(lines[14], "0.00000000 0.00000000 2.35000000 0 0 0")
        self.assertEqual(lines[15], "")

        trimer, natom = sia.STRU(shape="trimer",
                                element="Si",
                                mass=28.085,
                                fpseudo="Si.pbe-n-kjpaw_psl.1.0.0.UPF",
                                lattice_constant=10.263,
                                bond_length=2.35,
                                nspin=1)
        self.assertEqual(natom, 3)
        lines = trimer.split("\n")
        self.assertEqual(lines[0], "ATOMIC_SPECIES")
        self.assertEqual(lines[1], "Si 28.085000 Si.pbe-n-kjpaw_psl.1.0.0.UPF")
        self.assertEqual(lines[2], "LATTICE_CONSTANT")
        self.assertEqual(lines[3], "10.263000  // add lattice constant(a.u.)")
        self.assertEqual(lines[4], "LATTICE_VECTORS")
        self.assertEqual(lines[5], "1.00000000 0.00000000 0.00000000")
        self.assertEqual(lines[6], "0.00000000 1.00000000 0.00000000")
        self.assertEqual(lines[7], "0.00000000 0.00000000 1.00000000")
        self.assertEqual(lines[8], "ATOMIC_POSITIONS")
        self.assertEqual(lines[9], "Cartesian_angstrom  //Cartesian or Direct coordinate.")
        self.assertEqual(lines[10], "Si      //Element Label")
        self.assertEqual(lines[11], "0.00     //starting magnetism")
        self.assertEqual(lines[12], "3       //number of atoms")
        self.assertEqual(lines[13], "0.00000000 0.00000000 0.00000000 0 0 0")
        self.assertEqual(lines[14], "0.00000000 0.00000000 2.35000000 0 0 0")
        self.assertEqual(lines[15], "0.00000000 2.03517050 1.17500000 0 0 0")
        self.assertEqual(lines[16], "")

        tetramer, natom = sia.STRU(shape="tetramer",
                                    element="Si",
                                    mass=28.085,
                                    fpseudo="Si.pbe-n-kjpaw_psl.1.0.0.UPF",
                                    lattice_constant=10.263,
                                    bond_length=2.35,
                                    nspin=1)
        self.assertEqual(natom, 4)
        lines = tetramer.split("\n")
        self.assertEqual(lines[0], "ATOMIC_SPECIES")
        self.assertEqual(lines[1], "Si 28.085000 Si.pbe-n-kjpaw_psl.1.0.0.UPF")
        self.assertEqual(lines[2], "LATTICE_CONSTANT")
        self.assertEqual(lines[3], "10.263000  // add lattice constant(a.u.)")
        self.assertEqual(lines[4], "LATTICE_VECTORS")
        self.assertEqual(lines[5], "1.00000000 0.00000000 0.00000000")
        self.assertEqual(lines[6], "0.00000000 1.00000000 0.00000000")
        self.assertEqual(lines[7], "0.00000000 0.00000000 1.00000000")
        self.assertEqual(lines[8], "ATOMIC_POSITIONS")
        self.assertEqual(lines[9], "Cartesian_angstrom  //Cartesian or Direct coordinate.")
        self.assertEqual(lines[10], "Si      //Element Label")
        self.assertEqual(lines[11], "0.00     //starting magnetism")
        self.assertEqual(lines[12], "4       //number of atoms")
        self.assertEqual(lines[13], "0.00000000 0.00000000 0.00000000 0 0 0")
        self.assertEqual(lines[14], "0.00000000 0.00000000 2.35000000 0 0 0")
        self.assertEqual(lines[15], "0.00000000 2.03517050 1.17500000 0 0 0")
        self.assertEqual(lines[16], "1.91875150 0.67837450 1.17500000 0 0 0")
        self.assertEqual(lines[17], "")    

    def test_KPOINTS(self):

        kpt = sia.KPOINTS()
        lines = kpt.split("\n")
        self.assertEqual(lines[0], "K_POINTS")
        self.assertEqual(lines[1], "0")
        self.assertEqual(lines[2], "Gamma")
        self.assertEqual(lines[3], "1 1 1 0 0 0")
        self.assertEqual(lines[4], "")

    def test_INPUT(self):

        user_settings = {
            "suffix": "unittest",
            "stru_file": "unittest.stru",
            "kpoint_file": "unittest.kpt"
        }
        input = sia.INPUT(calculation_setting=user_settings, suffix="unittest")
        lines = input.split("\n")
        self.assertEqual(lines[0], "INPUT_PARAMETERS")
        self.assertEqual(lines[1], "suffix               unittest")
        self.assertEqual(lines[2], "stru_file            unittest.stru-unittest")
        self.assertEqual(lines[3], "kpoint_file          unittest.kpt-unittest")

    def test_generation(self):
        #print("generation function is not test due to it is only a wrapper contains few file operations")
        pass

    def test_read_INPUT(self):
        result = sia.read_INPUT("./SIAB/test/support")
        self.assertEqual(result["suffix"], "Si-trimer-1.9")
        self.assertEqual(result["stru_file"], "STRU-Si-trimer-1.9")
        self.assertEqual(result["kpoint_file"], "KPT-Si-trimer-1.9")
        self.assertEqual(result["pseudo_dir"], "/root/abacus-develop/pseudopotentials/SG15_ONCV_v1.0_upf")
        self.assertEqual(result["calculation"], "scf")
        self.assertEqual(result["basis_type"], "pw")
        self.assertEqual(result["ecutwfc"], "100")
        self.assertEqual(result["ks_solver"], "dav")
        self.assertEqual(result["nbands"], "10")
        self.assertEqual(result["scf_thr"], "1.0e-7")
        self.assertEqual(result["scf_nmax"], "9000")
        self.assertEqual(result["ntype"], "1")
        self.assertEqual(result["nspin"], "1")
        self.assertEqual(result["lmaxmax"], "2")
        self.assertEqual(result["bessel_nao_rcut"], "6 7")
        self.assertEqual(result["smearing_method"], "gauss")
        self.assertEqual(result["smearing_sigma"], "0.01")
        self.assertEqual(result["mixing_type"], "broyden")
        self.assertEqual(result["mixing_beta"], "0.8")
        self.assertEqual(result["mixing_ndim"], "8")
        self.assertEqual(result["mixing_gg0"], "1")
        self.assertEqual(result["printe"], "1")

    def test_version_compare(self):
        self.assertTrue(sia.version_compare("0.1.0", "0.1.0"))
        self.assertTrue(sia.version_compare("0.1.0", "0.1.1"))
        self.assertTrue(sia.version_compare("0.1.0", "0.2.0"))
        self.assertTrue(sia.version_compare("0.1.0", "1.0.0"))
        self.assertFalse(sia.version_compare("0.1.0", "0.0.1"))
        self.assertFalse(sia.version_compare("0.1.0", "0.0.9"))
        self.assertFalse(sia.version_compare("0.1.0", "0.0.0"))

    def test_blscan_guessbls(self):
        """this function is identical with the one in blscan"""
        self.assertListEqual(
            sia.blscan_guessbls(2.0, [0.1, 0.1]), 
            [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5]
        )

    def test_blscan_fitmorse(self):

        r = np.linspace(1.5, 5.0, 100) # Angstrom
        De = 2.5 # eV
        a = 1.0
        re = 2.0 # Angstrom
        y = morse(r, De, a, re)
        x = r
        e_dis, _, bleq, _ = sia.blscan_fitmorse(x, y)
        self.assertAlmostEqual(e_dis, De, places=2)
        self.assertAlmostEqual(bleq, re, places=2)

        # with noise
        nprecision = 3
        y = y + np.random.normal(0, 10**(-nprecision), 100)
        e_dis, _, bleq, _ = sia.blscan_fitmorse(x, y)
        self.assertAlmostEqual(e_dis, De, places=nprecision-1)
        self.assertAlmostEqual(bleq, re, places=nprecision-1)

    def test_returnbls(self):

        r = np.linspace(1.5, 5.0, 100)
        De = 2.5 # eV
        a = 1.0 # 1/Angstrom
        re = 2.25 # Angstrom
        y = morse(r, De, a, re)
        x = r.tolist()
        e_dis, _, bleq, e0 = sia.blscan_fitmorse(x, y)
        self.assertAlmostEqual(e_dis, De, places=2)
        self.assertAlmostEqual(bleq, re, places=2)
        self.assertAlmostEqual(e0, 0.0, places=2)
        blrange = sia.blscan_returnbls(bl0=bleq, ener0=e0, bond_lengths=x, energies=y, ener_thr=1.0)
        self.assertListEqual(blrange, [1.7474747474747474,
                                       1.9949494949494948,
                                       2.242424242424242,
                                       2.737373737373737,
                                       3.2676767676767673]
        )

    def test_is_duplicate(self):

        calculation_setting = {
            "basis_type": "pw",
            "bessel_nao_rcut": [6, 7],
            "lmaxmax": 2
        }
        self.assertTrue(
            sia.is_duplicate("/root/abacus-develop/orbital_generation/SIAB/interface/test/support/Si-trimer-2.6",
                             calculation_setting)
        )
        self.assertFalse(
            sia.is_duplicate("/root/abacus-develop/orbital_generation/SIAB/interface/test/support/Si-trimer-2.7",
                             calculation_setting)
        )
        calculation_setting = {
            "basis_type": "pw",
            "bessel_nao_rcut": [6, 7],
            "lmaxmax": 3
        }
        self.assertFalse(
            sia.is_duplicate("/root/abacus-develop/orbital_generation/SIAB/interface/test/support/Si-trimer-2.6",
                             calculation_setting)
        )
        self.assertFalse(
            sia.is_duplicate("/root/abacus-develop/orbital_generation/SIAB/interface/test/support/Si-trimer-2.7",
                             calculation_setting)
        )
        calculation_setting = {
            "basis_type": "pw",
            "bessel_nao_rcut": [6, 8],
            "lmaxmax": 2
        }
        self.assertFalse(
            sia.is_duplicate("/root/abacus-develop/orbital_generation/SIAB/interface/test/support/Si-trimer-2.6",
                             calculation_setting)
        )

if __name__ == "__main__":
    unittest.main()
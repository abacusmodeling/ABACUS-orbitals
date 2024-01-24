import unittest
import SIAB.abacus as ab

class TestAbacus(unittest.TestCase):

    def test_dimer(self):

        dimer = ab.dimer(element="Si",
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
            
        trimer = ab.trimer(element="Si",
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

        tetramer = ab.tetramer(element="Si",
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
        dimer, natom = ab.STRU(shape="dimer",
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

        trimer, natom = ab.STRU(shape="trimer",
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

        tetramer, natom = ab.STRU(shape="tetramer",
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

        kpt = ab.KPOINTS()
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
        input = ab.INPUT(user_settings=user_settings, suffix="unittest")
        lines = input.split("\n")
        self.assertEqual(lines[0], "INPUT_PARAMETERS")
        self.assertEqual(lines[1], "suffix               unittest")
        self.assertEqual(lines[2], "stru_file            unittest.stru-unittest")
        self.assertEqual(lines[3], "kpoint_file          unittest.kpt-unittest")

    def test_generation(self):
        #print("generation function is not test due to it is only a wrapper contains few file operations")
        pass

    def test_submit(self):
        jtg = ab.submit(folder="unittest",
                        module_load_command="module load abacus",
                        mpi_command="mpirun -np 1",
                        abacus_command="abacus --version",
                        rcuts=[6, 7],
                        test=True)

    def test_read_INPUT(self):
        result = ab.read_INPUT("./SIAB/test/support")
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

if __name__ == "__main__":
    unittest.main()
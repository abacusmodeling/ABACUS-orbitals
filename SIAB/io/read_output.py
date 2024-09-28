import os
import unittest

def read_energy(folder: str,
                suffix: str,
                calculation: str = "scf"):
    if suffix is None:
        suffix = "ABACUS"
    frunninglog = "%s/OUT.%s/running_%s.log"%(folder, suffix, calculation)
    if not os.path.exists(frunninglog):
        raise FileNotFoundError("running log %s not found."%frunninglog)
    else:
        with open(frunninglog, "r") as f:
            line = "start"
            while line is not None:
                line = f.readline().strip()
                if line.startswith("!FINAL_ETOT_IS"):
                    energy = float(line.split()[-2])
                    return energy
    return -1

def read_natom(folder: str,
               suffix: str,
               calculation: str = "scf"):
    if suffix is None:
        suffix = "ABACUS"
    frunninglog = "%s/OUT.%s/running_%s.log"%(folder, suffix, calculation)
    if not os.path.exists(frunninglog):
        raise FileNotFoundError("running log %s not found."%frunninglog)
    else:
        with open(frunninglog, "r") as f:
            line = "start"
            while line is not None:
                line = f.readline().strip()
                if line.startswith("TOTAL ATOM NUMBER"):
                    natom = int(line.split()[-1])
                    return natom
    return -1


class TestReadOutput(unittest.TestCase):

    def test_read_energy(self):
        energy = -1
        energy = read_energy(folder = "./SIAB/io/test/support",
                                calculation = "scf",
                                suffix = "unittest")
        self.assertNotEqual(energy, -1)
        print("energy = %s eV"%energy)

if __name__ == "__main__":
    unittest.main()
import os
def read_energy(folder: str,
                calculation: str = "scf",
                suffix: str = "ABACUS"):
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

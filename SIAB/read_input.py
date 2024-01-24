import re
import os

def parse(fname: str = ""):

    keyvalue_pattern = r"^(\w+)(\s+)([^#]*)(#.*)?"
    float_pattern = r"^\d+\.\d*$"
    int_pattern = r"^\d+$"
    scalar_keywords = ["Ecut", "sigma", "element"]
    result = {}
    if fname == "":
        raise ValueError("No filename provided")
    with open(fname, "r") as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        _match = re.match(keyvalue_pattern, line)
        if _match:
            key = _match.group(1).strip()
            value = _match.group(3).strip().split()
            value = [float(v) if re.match(float_pattern, v) else int(v) if re.match(int_pattern, v) else v for v in value]
            result[key] = value if key not in scalar_keywords else value[0]
    
    return result

def default(inp: dict):

    if "EXE_opt" not in inp.keys():
        inp["EXE_opt"] = ""
    if inp["EXE_opt"] == "":
        inp["EXE_opt"] = "/opt_orb_pytorch_dpsi/main.py (default)" 
        optimizer_path = "/opt_orb_pytorch_dpsi" 
    else:
        optimizer_path =os.path.dirname(inp["EXE_opt"])
    if "EXE_env" not in inp.keys():
        inp["EXE_env"] = ""
    return inp, optimizer_path

def wash(inp: dict):
    """the parsed input initially might be like:
    {
        "EXE_mpi": ["mpirun", "-np", 1],
        "EXE_pw": ["abacus", "--version"],
        "element": "Si",
        "Ecut": 100,
        "Rcut": [ 6, 7 ],
        "Pseudo_dir": ["/root/abacus-develop/pseudopotentials/SG15_ONCV_v1.0_upf"],
        "Pseudo_name": ["Si_ONCV_PBE-1.0.upf"],
        "sigma": 0.01,
        "STRU1": ["dimer", 8, 2, 1, 1.8, 2.0, 2.3, 2.8, 3.8],
        "STRU2": ["trimer", 10, 2, 1, 1.9, 2.1, 2.6],
        "max_steps": [200],
        "Level1": ["STRU1", 4, "none", "1s1p"],
        "Level2": ["STRU1", 4, "fix", "2s2p1d"],
        "Level3": ["STRU2", 6, "fix", "3s3p2d"],
        "Save1": ["Level1", "Z"],
        "Save2": ["Level2", "DZP"],
        "Save3": ["Level3", "TZDP"]
    }
    """
    inp, optimizer_path = default(inp)

    exe_pw = " ".join([str(word) for word in inp["EXE_pw"]]).replace("\\", "/")
    inp["EXE_pw"] = exe_pw

    exe_mpi = " ".join([str(word) for word in inp["EXE_mpi"]]).replace("\\", "/")
    inp["EXE_mpi"] = exe_mpi

    pseudo_dir = inp["Pseudo_dir"][0].strip().replace("\\", "/")
    if pseudo_dir.endswith("/"):
        pseudo_dir = pseudo_dir[:-1]
    inp["Pseudo_dir"] = pseudo_dir

    fpseudo = inp["Pseudo_name"][0].strip().replace("\\", "").replace("/", "")
    inp["Pseudo_name"] = fpseudo

    return inp

if __name__ == "__main__":
    print(parse("SIAB/example_Si/SIAB_INPUT"))
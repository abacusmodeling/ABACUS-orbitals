import re
import os
import json
import SIAB.io.pseudopotential.tools.basic as siptb
from SIAB.data.interface import PERIODIC_TABLE_TOINDEX
from SIAB.io.pseudopotential.api import ppinfo
from numpy import ceil
import unittest
import uuid

def read_siab_plaintext(fname: str = ""):
    keyvalue_pattern = r"^(\w+)(\s+)([^#]*)(#.*)?"
    float_pattern = r"^\d+\.\d*$"
    int_pattern = r"^\d+$"
    scalar_keywords = ["Ecut", "sigma", "element"]
    result = {"fit_basis": "pw"}
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

def read_siab_json(fname: str = ""):
    """parse SIAB_INPUT file with version 0.2.0 in json format"""
    with open(fname, "r") as f:
        result = json.load(f)
    return result

def read(fname: str = "", version: str = "0.1.0"):
    """default value setting is absent"""
    print(f"""
Parsing SIAB input file {fname} with version {version}
""")
    if fname.endswith(".json"):
        result = read_siab_json(fname)
    else:
        result = read_siab_plaintext(fname)
        result = postprocess_siab_oldinp(result) if version == "0.1.0" else result
        result = convert_oldinp_tojson(result) if version == "0.1.0" else result
        result = convert_plaintext_tojson(result) if version != "0.1.0" else result
    # convert `pseudo_dir` to absolute path
    result["pseudo_dir"] = os.path.abspath(result["pseudo_dir"])
    return result

def postprocess_siab_oldinp(inp: dict):
    """the parsed input initially might be like:
    ```json
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
    ```
    However, the value of EXE_mpi and EXE_pw are not expected to be list, but string.
    """
    # the optimizer path is not really used, impose default values for it
    inp["EXE_opt"] = inp.get("EXE_opt", "")
    inp["EXE_opt"] = "/opt_orb_pytorch_dpsi/main.py (default)" if inp["EXE_opt"] == "" else inp["EXE_opt"]
    # EXE_env
    inp["EXE_env"] = inp.get("EXE_env", "")
    # EXE_pw: concatenate the command
    exe_pw = " ".join([str(word) for word in inp["EXE_pw"]]).replace("\\", "/")
    inp["EXE_pw"] = exe_pw
    # EXE_mpi: concatenate the command
    exe_mpi = " ".join([str(word) for word in inp["EXE_mpi"]]).replace("\\", "/")
    inp["EXE_mpi"] = exe_mpi
    # drop the [] from list
    pseudo_dir = inp["Pseudo_dir"][0].strip().replace("\\", "/")
    pseudo_dir = pseudo_dir[:-1] if pseudo_dir.endswith("/") else pseudo_dir
    inp["Pseudo_dir"] = pseudo_dir
    # drop the [] from list
    fpseudo = inp["Pseudo_name"][0].strip().replace("\\", "").replace("/", "")
    inp["Pseudo_name"] = fpseudo
    # drop the [] from list
    inp["max_steps"] = int(inp["max_steps"][0])
    
    return inp

def translate_oldinp_keyword(keyword: str):
    """translate the old version keywords to new version keywords"""
    dictionary = {"Ecut": "ecutwfc", "Rcut": "bessel_nao_rcut", 
                  "Pseudo_dir": "pseudo_dir", "sigma": "smearing_sigma"}
    return dictionary.get(keyword, keyword)

def convert_oldinp_tojson(inp: dict):
    """convert the old version input contents to new version"""
    result = {
        "reference_systems": [],
        "orbitals": []
    }
    keys = {"EXE_env": "environment", "EXE_mpi": "mpi_command", "EXE_pw": "abacus_command",
            "Pseudo_dir": "pseudo_dir", "Pseudo_name": "pseudo_name", "Ecut": "ecutwfc",
            "Rcut": "bessel_nao_rcut", "sigma": "smearing_sigma", "max_steps": "max_steps"}
    for oldname, newname in keys.items():
        if oldname in inp:
            result[newname] = inp[oldname]
        else:
            print(f"Warning: {oldname} not found in the input file", flush=True)
    for key, val in inp.items():
        if key.startswith("STRU"):
            result["reference_systems"].append({
                "shape": val[0],
                "nbands": int(val[1]), # maxL the useless parameter skipped
                "nspin": int(val[3]),
                "bond_lengths": ["auto"] if val[4] == "auto" else [float(v) for v in val[4:]]
            })
        elif key.startswith("Save"):
            level = val[0]
            result["orbitals"].append({
                "zeta_notation": val[1],
                "shape": [s for s in inp.keys() if s.startswith("STRU")].index(inp[level][0]),
                "nbands_ref": "auto" if inp[level][1] == "auto" else int(inp[level][1]),
                "orb_ref": "none" if inp[level][2] == "none" else inp[
                    "Save"+str(int(level[5:]) - 1)][1]
            })

    return result

def convert_plaintext_tojson(inp: dict):
    """especially for version 0.2.0"""
    
    shapes = ["dimer", "trimer", "tetramer"]

    def is_reference_systems_line(key, value):
        if key in shapes:
            if len(value) == 3:
                if value[0].isdigit():
                    if value[1].isdigit():
                        if isinstance(value[2], list):
                            return True
        return False
    
    def is_orbitals_line(key, value):
        zeta_pattern = r"^(\s*)([SDTQ56789]?Z([SDTQ56789]?P)?)"
        match = re.match(zeta_pattern, key)
        if match:
            if len(value) == 3:
                if value[0] in shapes:
                    if value[1].isdigit():
                        if value[2] == "none" or re.match(zeta_pattern, value[2]):
                            return True
        return False

    result = {
        "reference_systems": [],
        "orbitals": []
    }
    for key, value in inp.items():
        if is_reference_systems_line(key, value):
            result["reference_systems"].append({
                "shape": key,
                "nbands": int(value[0]),
                "nspin": int(value[1]),
                "bond_lengths": value[2]
            })
        elif is_orbitals_line(key, value):
            result["orbitals"].append({
                "zeta_notation": key,
                "shape": value[0],
                "nbands_ref": int(value[1]),
                "orb_ref": value[2]
            })
        else:
            result[key] = value

def natom_from_shape(shape: str):
    natom = {"monomer": 1, "dimer": 2, "trimer": 3, "tetrahedron": 4, 
             "square": 4, "triangular_bipyramid": 5, "octahedron": 6, "cube": 8}
    return natom.get(shape, 0)

def nbands_from_str(option: str|float|int, shape: str, z_val: float):

    if isinstance(option, str):
        assert re.match(r"(auto|occ((\+|-)\d+)?|all)", option), f"option should be auto, occ, occ+/-n or all: {option}"
        if option == "auto":
            return "auto"
        if option == "all":
            return int(max(natom_from_shape(shape)*z_val, 2))
        if option.startswith("occ"):
            occ = int(max(natom_from_shape(shape)*z_val/2, 1))
            return eval(option.replace("occ", str(occ)))
    return int(option)

def abacus_settings(user_settings: dict, minimal_basis: list = None, z_val: float = 0, z_core: float = 0):

    # copy all possible shared parameters (shared by all reference systems)
    all_params = abacus_params()
    template = {key: value for key, value in user_settings.items() if key in all_params}
    # then create copies for each reference system
    refsys = user_settings.get("reference_systems", [])
    autoset_monomer = bool(user_settings.get("spill_guess", "random") == "atomic" \
         and len([True for s in refsys if s["shape"] == "monomer"]) == 0)
    # set `autoset_mononer` to True if spill_guess is atomic and monomer is not in reference systems
    refsys.append({"shape": "monomer"}) if autoset_monomer else None
    nsystem = len(refsys)

    b1 = (nsystem > 0 and not autoset_monomer)
    b2 = (nsystem > 1 and autoset_monomer)
    assert (b1 or b2), "number of reference systems should be at least 2 if spill_guess is atomic, otherwise at least 1"

    result = [template.copy() for _ in range(nsystem)]
    # then parameters cannot share
    shape_index_mapping = [v["shape"] for v in refsys]

    ####################################
    # polarization causes lmaxmax += 1 #
    ####################################
    with_polarization = [False]*nsystem
    for iorb in range(len(user_settings["orbitals"])):
        val = user_settings["orbitals"][iorb]["shape"]
        """DEVELOPE DETAILS
        I recognize there may be the need that defining different set of bond-length lists but
        on the same shape, for example, the dimer with different bond lengths. Also on the other
        hand it is possible for user to use not only one set of reference structures for generating
        one set of orbitals. Therefore the `shape` should support both scalar and list value, of
        both int (the index) and str (the shape name) type.
        
        Changed at Aug 6th, 2024"""
        val = [val] if not isinstance(val, list) else val
        assert all(isinstance(v, (str, int)) for v in val)
        # will not support the mixing of str and int in the list...
        val = [shape_index_mapping.index(v) if isinstance(v, str) else v for v in val]
        if user_settings["orbitals"][iorb]["zeta_notation"][-1] == "P":
            for v in val:
                with_polarization[v] = True

    # monomer is special, it is used as initial guess for spillage optimization, therefore it should have all possible
    # lmax values, therefore the maximal one over all reference systems
    lmax_monomer = 0
    for irs in range(nsystem):
        # auto set nbands if for reference system the nbands is set to "auto"
        nbands = refsys[irs].get("nbands", "auto")
        if nbands == "auto":
            shape = refsys[irs]["shape"]
            nelec_tot = natom_from_shape(shape)*z_val
            if nelec_tot < 1:
                print("WARNING: program possibly cannot grep reasonable `z_valence` from pseudopotential.")
            nbands = int(max(nelec_tot, 2))
        # auto set lmaxmax
        lmaxmax = refsys[irs].get("lmaxmax", "auto")
        if lmaxmax == "auto":
            lmaxmax = len(minimal_basis) \
            if (with_polarization[irs] and [] not in minimal_basis) else len(minimal_basis) - 1
        lmax_monomer = max(lmax_monomer, lmaxmax)
        # set nspin
        nspin = refsys[irs].get("nspin", 1)
        # update
        result[irs].update({"nbands": nbands, "lmaxmax": lmaxmax, "nspin": nspin})
        # for all other parameters in user_settings["reference_systems"][irs], if in abacus_params, overwrite
        for key, value in refsys[irs].items():
            if key in all_params and key not in ["shape", "nbands", "nspin"]:
                result[irs][key] = value
    # set monomer
    if autoset_monomer:
        nbands_monomer = cal_nbands_fill_lmax(z_val, z_core, lmax_monomer) # fill the lmax shell
        result[shape_index_mapping.index("monomer")].update(
            {"lmaxmax": lmax_monomer, "nbands": nbands_monomer})
    return result

def siab_settings(user_settings: dict, minimal_basis: list, z_val: float = 0):
    """convert user_settings to SIAB settings the information needed by spillage optimization
    information is organized as follows:
    
    {
        "optimizer": "pytorch.SWAT",
        "max_steps": 9000,
        "spill_coefs": [2.0, 1.0],
        "orbitals": [
            {
                "nzeta": [1, 1],
                "nzeta_from": "none",
                "nbands_ref": 4,
                "folder": ["dimer"] # this will have exact value after abacus calculation
            },
            {
                "nzeta": [2, 2, 1],
                "nzeta_from": [1, 1],
                "nbands_ref": 4,
                "folder": ["dimer"] # this will have exact value after abacus calculation
            }
        ]
    }
    """
    # allocate
    result = {
        "optimizer": user_settings.get("optimizer", "pytorch.SWAT"),
        "max_steps": user_settings.get("max_steps", 1000),
        "spill_coefs": user_settings.get("spill_coefs", [2.0, 1.0]),
        "spill_thr": user_settings.get("spill_thr", 1e-8),
        "nthreads_rcut": user_settings.get("nthreads_rcut", -1),
        "orbitals": [{} for _ in range(len(user_settings["orbitals"]))],
        "jY_type": user_settings.get("jY_type", "reduced")
    }
    shapes = [rs["shape"] for rs in user_settings["reference_systems"]]

    #####################################################################################
    #                  CHANGE LOG: infer nzeta, since ABACUS-ORBGEN v3.0                #
    #                                        * * *                                      #
    # the `nzeta` is not necessary to be inferred here, but lmaxmax is really           #
    # compulsory. But because the conversion from input to runtime setting is designed  #
    # to be complete as early as possible, so the nzeta is inferred here. For           #
    # determining the nzeta, there are three ways supported:                            #
    # 1. infer from zeta_notation and minimal_basis. The latter is read from            #
    # pseudopotential (and if not possible , will raise some errors).                   #
    # 2. specifying the nzeta directly                                                  #
    # 3. (for jy) decompose bands in range (0, nbands_ref) into contributions from      #
    #    each angular momentum, then make average to get the nzeta.                     #
    #                                                                                   #
    # Here it is before performing calculation by ABACUS, therefore only the former two #
    # are implemented here.                                                             #
    #####################################################################################

    for iorb, orbital in enumerate(user_settings["orbitals"]):
        # here the nzeta is the first time to be inferred. Since SIAB-v3.0, the nzeta can
        # be inferred after DFT calculation, the only thing can always be known now is
        # the dependency between orbitals, therefore the function `nzetagen` is allowed
        # to return index of orbitals if it is for handling `orb_ref` instead of `zeta_
        # notation`.
        tmp = nzetagen(orbital["zeta_notation"], minimal_basis)\
            if orbital["zeta_notation"] != "auto" else "auto"
        assert not isinstance(tmp, int), "`zeta_notation` should not be inferred as an integer"
        result["orbitals"][iorb]["nzeta"] = tmp
        result["orbitals"][iorb]["nzeta_from"] = None \
            if orbital["orb_ref"] == "none" \
            else nzetagen(orbital["orb_ref"], minimal_basis)
        # implement "occ", "occ+%d", "occ-%d" and "all" for nbands_ref
        # support index to link reference structures
        shape = orbital["shape"]
        shape = [shape] if not isinstance(shape, list) else shape
        index = [shapes.index(s) for s in shape if not isinstance(s, int)]
        nbands_ref = orbital["nbands_ref"]
        nbands_ref = [nbands_ref] if not isinstance(nbands_ref, list) else nbands_ref
        assert len(nbands_ref) == len(shape), "nbands_ref should have the same length as shape"
        result["orbitals"][iorb]["nbands_ref"] = [nbands_from_str(n, s, z_val) for n, s in zip(nbands_ref, shape)]
        result["orbitals"][iorb]["folder"] = index
        
    # indexing the nzeta_from -> nzeta
    nzeta = [orb["nzeta"] for orb in result["orbitals"]]
    for orb in result["orbitals"]:
        if isinstance(orb["nzeta_from"], list):
            orb["nzeta_from"] = nzeta.index(orb["nzeta_from"])
    return result

def nzetagen(zeta_notation, minimal_basis: list):
    """generate the nzeta from the zeta_notation. zeta_notation can be of three formats: 
    2s2p1d, DZP or [2, 2, 1].
    
    Parameters
    ----------
    zeta_notation: str or list
        the zeta notation, can be of three formats: 2s2p1d, DZP or [2, 2, 1]
    minimal_basis: list
        the minimal basis read from pseudopotential, like [2, 2, 1]
    
    Returns
    -------
    list of int
        the nzeta, like [2, 2, 1]
    """

    assert isinstance(minimal_basis, list), "minimal_basis should be a list"

    # 
    if isinstance(zeta_notation, list) and all([isinstance(v, int) for v in zeta_notation]):
        return zeta_notation
    if isinstance(zeta_notation, str) and re.match(r"([SDTQ5-9]?Z)(([SDTQ5-9]?P)*)", zeta_notation):
        return siptb.orbconf_fromxzyp(zeta_notation, minimal_basis, as_list=True)
    if isinstance(zeta_notation, str) and re.match(r"(\d+[a-z])+", zeta_notation):
        spectra = ["s", "p", "d", "f", "g", "h", "i", "k", "l", "m", "n", "o"]
        result = {v[-1]: int(v[:-1]) for v in re.findall(r"\d+[a-z]", zeta_notation)}
        result = [result.get(s, 0) for s in spectra]
        while result[-1] == 0:
            result.pop()
        return result
    # change log: the dependency between orbitals now can support "auto" or index, because 
    # the connection is known before performing DFT calculations, while the nzeta might be
    # unknown. Support since SIAB-v3.0
    if isinstance(zeta_notation, int):
        return zeta_notation
    
    assert False, "ERROR: \"zeta_notation\" is not in the correct format. It should be like \"2s2p1d\" or \"DZP\" or [2, 2, 1]"

def environment_settings(user_settings: dict):

    return {
        "environment": user_settings["environment"],
        "mpi_command": user_settings["mpi_command"],
        "abacus_command": user_settings["abacus_command"]
    }

def structure_settings(user_settings: dict):
    """handle with structures needed to calculate their pw wavefunctions as reference
    for fitting numerical atomic orbitals. Special case is inclusion of monomer, if
    keyword spill_guess is set to atomic.
    
    Return:
        list of tuple, each tuple is (shape, bond_lengths)
    """
    refsys = user_settings.get("reference_systems", [])
    need_monomer = user_settings.get("spill_guess", "random") == "atomic" and not any(rs["shape"] == "monomer" for rs in refsys)
    shapes = [rs["shape"] for rs in refsys] # list of str
    shapes.append("monomer") if need_monomer else None
    bond_lengths = [rs.get("bond_lengths", "auto") for rs in refsys] # list of list of float
    bond_lengths.append("auto") if need_monomer else None
    unique_shapes = list(set(shapes))
    _shapemap = [orb["shape"] for orb in user_settings["orbitals"]]
    error_msg = """ERROR: there are reference systems with identical shape. In this case, should
specify the `shape` keyword in `orbitals` section with integar as index of reference,
instead of string like `dimer`.
Raise ValueError, Quit..."""
    if len(unique_shapes) != len(shapes) and not all([isinstance(s, int) for s in _shapemap]):
        print(error_msg, flush=True)
        raise ValueError("ERROR: see error message above")
    return list(zip(shapes, bond_lengths))

def from_pseudopotential(pseudopotential: dict):
    """convert the pseudopotential to SIAB input"""
    symbol = pseudopotential["element"]
    minimal_basis = pseudopotential["val_conf"]
    z_val = pseudopotential["z_val"]
    return {
        "element": symbol,
        "minimal_basis": minimal_basis,
        "z_val": z_val
    }

def description(symbol: str, user_settings: dict):

    return {
        "element": symbol,
        "pseudo_dir": user_settings["pseudo_dir"],
        "pseudo_name": user_settings["pseudo_name"],
        "skip_abacus": user_settings.get("optimizer") in ["none", "restart"]
    }

def skip_ppread(user_settings: dict):
    """check if the pseudopotential read-in can be skipped or not

    Parameters
    ----------
    user_settings: dict
        the user settings
    
    Returns
    -------
    bool
        True if the pseudopotential read-in can be skipped, False otherwise
    """

    skip = True
    # case 0
    # if element is not set, it is not possible to skip the pseudopotential read-in
    if "element" not in user_settings:
        print("AUTOSET: `element` is not specified => AUTOSET", flush=True)
        return False
    # case 1
    # if nbands is specified as auto, occ, all, it must requires the number of valence
    # electrons, therefore it is not possible to skip the pseudopotential read-in
    # case 2
    # for both jy and pw, lmaxmax is compulsory. for pw, lmaxmax is needed in INPUT file
    # , for jy, lmaxmax is needed as LCAO basis. If not specified, it is not possible to
    # skip the pseudopotential read-in
    for shape in user_settings["reference_systems"]:
        if shape["nbands"] in ["auto", "occ", "all"]:
            skip = False
            print("AUTOSET: `nbands` is not specified with a specific value => AUTOSET", flush=True)
            break
        if "lmaxmax" not in shape:
            skip = False
            print("AUTOSET: `lmaxmax` is not specified => AUTOSET", flush=True)
            break
    # case 3
    # there is a strategy from SIAB-v2.0 that taking the valence electron configuration
    # as minimal basis, then generate orbitals based on it like SZ is itself, DZ is its
    # doubled, DZP is DZ with an additional orbital with angular momentum plus one. In
    # SIAB-v1.0, one should always specify something like SsPpDd..., in which S, P, D 
    # stands the number of orbitals with angular momentum s(0), p(1), d(2) respectively.
    # The conversion from SsPpDd... to list [S, P, D, ...] is trivial, thus if the
    # zeta_notation is not specified as either a list of integers or a string like SsPpDd,
    # it is not possible to skip the pseudopotential read-in
    orbpat = r"(\d+[spdfgh])+" # like 2s2p1d
    for orbital in user_settings["orbitals"]:
        z = orbital["zeta_notation"]
        if isinstance(z, list):
            if not all([isinstance(v, int) for v in z]):
                print("AUTOSET: `zeta_notation` is not specified with a list of integers => AUTOSET", flush=True)
                skip = False
                break
        # changlog: if `fit_basis jy`, the nzeta can be inferred by doing angular momentum
        # band decomposition. Accumulating the occ of each l and divide by degeneracy (1, 3, 5
        # ... for s, p, d), taking powered averge between kpoints and algebraic average between
        # structures, then we can get the nzeta. This is implemented in SIAB-v3.0
        elif isinstance(z, str) and not re.match(orbpat, z) and z != "auto":
            skip = False
            print("AUTOSET: `zeta_notation` is not specified with a string like 2s2p1d => AUTOSET", flush=True)
            break
    # case 4
    # the same as case 3, but for `orb_ref`.
    for orbital in user_settings["orbitals"]:
        z = orbital["orb_ref"]
        if isinstance(z, list):
            if not all([isinstance(v, int) for v in z]):
                print("AUTOSET: `orb_ref` is not specified with a list of integers => AUTOSET", flush=True)
                skip = False
                break
        elif isinstance(z, str) and z not in ["none", "auto"] and not re.match(orbpat, z):
            print("AUTOSET: `orb_ref` is not specified with a string like 2s2p1d => AUTOSET", flush=True)
            skip = False
            break
        elif isinstance(z, int) and (z < 0 or z >= len(user_settings["orbitals"])):
            print("AUTOSET: `orb_ref` is specified as an index, but the index is out of range => AUTOSET", flush=True)
            skip = False
            break
        # changelog: with `fit_basis jy` and `zeta_notation auto`, the nzeta can be inferred,
        # the hierarchy of orbitals now is defined by `nbands_ref` and not-known before the
        # DFT calculation being performed. We should allow the case that the link/dependency
        # between orbitals is "auto" or specified as an index (because the orbital is defined
        # in a list in input, therefore the index is known).
    return skip

def _validate_param(user_settings: dict):
    """validate the input parameters
    
    Parameters
    ----------
    user_settings: dict
        the user settings
    
    Returns
    -------
    None
    """
    # check if the shape assigned to orbitals is valid
    shape2index = {rs["shape"]: i for i, rs in enumerate(user_settings["reference_systems"])}
    for iorb, orb in enumerate(user_settings["orbitals"]):
        shape = orb["shape"]
        shape = [shape] if not isinstance(shape, list) else shape
        for s in shape:
            assert isinstance(s, (str, int)), f"shape {s} is not a valid shape"
            if isinstance(s, str):
                assert s in shape2index, f"shape {s} is not found in reference systems"

    # check if the nbands set for reference system is smaller than
    # bands needed for fitting orbitals
    shape2index = {rs["shape"]: i for i, rs in enumerate(user_settings["reference_systems"])}
    for iorb, orb in enumerate(user_settings["orbitals"]):
        shape = orb["shape"]
        shape = [shape] if not isinstance(shape, list) else shape
        for s in shape:
            if isinstance(s, str):
                assert orb["nbands_ref"] <= user_settings["reference_systems"][shape2index[s]]["nbands"], \
                    f"ERROR: `nbands_ref` for orbital {iorb} is larger than the number of bands set for\
 reference system `{s}`"
            elif isinstance(s, int):
                assert orb["nbands_ref"] <= user_settings["reference_systems"][s]["nbands"], \
                    f"ERROR: `nbands_ref` for orbital {iorb} is larger than the number of bands set for\
 reference system `{s}`"


def parse(user_settings: dict):
    """unpack the SIAB input to structure (shape as key and bond lengths are list as value),
    input setting of abacus, orbital generation settings, environmental settings and general description
    """
    _validate_param(user_settings)

    # move the information fetch from pseudopotential from front.py here...
    # get value from the dict returned by function from_pseudopotential

    if skip_ppread(user_settings):
        """the logic here is, because there are pseudopotential can be parsed automatically,
        but the range of supported are limited. For those pseudopotential that is not with
        the format expected, user has to specify information manually. In this case there
        are many steps not needed to proceed, so we skip the pseudopotential read-in here.
        
        Also, there are cases even user provides a pseudopotential that can be parsed automatically,
        user might still want to set all things manually, in this case, the pseudopotential
        read-in is also uncessary and therefore can be skipped."""
        symbol = user_settings.get("element", "X")
        if symbol == "X":
            print("WARNING: `element` keyword is not specified in input, will use `X` as placeholder", flush=True)
        minimal_basis = user_settings.get("minimal_basis", [])
        z_val = user_settings.get("z_val", 0)
    else:
        ppinfo_ = ppinfo(os.path.join(user_settings["pseudo_dir"], user_settings["pseudo_name"]))
        properties = ["element", "minimal_basis", "z_val"]
        symbol, minimal_basis, z_val = map(from_pseudopotential(ppinfo_).get, properties)

    z_core = PERIODIC_TABLE_TOINDEX.get(symbol, z_val) - z_val
    structures = structure_settings(user_settings)
    # the following call is for generating the INPUT setting for each structure
    abacus = abacus_settings(user_settings, minimal_basis, z_val, z_core)
    siab = siab_settings(user_settings, minimal_basis, z_val)
    env = environment_settings(user_settings)
    general = description(symbol, user_settings)
    return structures, abacus, siab, env, general

def abacus_params():
    pattern = r"^([\w]+)(\s+)([^#]+)(\s*)(#.*)?"
    keys = []
    for line in ABACUS_INPUT_TEMPLATE.split("\n"):
        match = re.match(pattern, line)
        if match:
            key = match.group(1).strip()
            #value = match.group(3).strip()
            keys.append(key)
    return keys

def cal_nbands_fill_lmax(zval: int, zcore: int, lmax: int, fill_lmax: bool = True) -> int:
    """
    WARNING: Only for use from single isolated atom case!

    according to Hund's rule, select an appropriate value to include explicit calculation on states involving hydrogen-like
    orbitals with angular momentum up to lmax.

    Args:
        zval (int): the number of valence electrons, always can be founded in PP_HEADER.attrib.z_valence
        zcore (int): the number of core electrons, equals to Z - zval
        lmax (int): the maximal angular momentum to include in the calculation
        fill_lmax (bool): if True, fill up to lmax, otherwise reach to lmax
    
    Returns:
        int: the number of bands to include in the calculation
    """
    
    # the first and the last element that fills shell of lmax.
    # S: 1, Hydrogen; P: 5, Boron; D: 21, Scandium; F: 58, Cerium;
    # S: 2, Helium;   P: 10, Neon; D: 30, Zinc;     F: 70, Ytterbium
    # nelec_max: the maximal number of electrons for each l and each period (can fill up to lmax orbitals)
    # nelec_min: the minimal number of electrons for each l and each period (can reach to lmax orbitals)
    z_max = [[2, 4, 12, 20, 38, 56, 88, 120], [10, 18, 36, 54, 86, 118], 
             [30, 48, 80, 112], [70, 102]]
    z_min = [[1, 3, 11, 19, 37, 55, 87, 119], [5, 13, 31, 49, 81, 113],
             [21, 39, 57, 89], [58, 90]]
    z_ref = z_max if fill_lmax else z_min
    print(f"AUTOSET: adding orbitals... will add electrons to l = {lmax} orbitals according to Hund's rule", flush=True)
    strategy = "\'fill up to\'" if fill_lmax else "\'reach to\'"
    print(f"AUTOSET: strategy {strategy} lmax = {lmax} subshell", flush=True)
    if lmax > 3:
        print(f"WARNING: lmax > 3, will add g-orbital, which is not possible for all ground state atoms", flush=True)
    # for all l <= lmax, find the minimal number of electrons that can fill up to lmax orbitals
    nelec = zval
    for l in range(min(lmax, 3) + 1):
        nelec_l = min([z for z in z_ref[l] if z >= zcore]) - zcore
        nelec = max(nelec, nelec_l)
    print(f"AUTOSET: minimal number of electrons to fill up to l = {min(lmax, 3)} is {nelec}", flush=True)
    nbands = ceil(nelec/2)
    nbands += 5 if nelec == zval else 0 # for the case low lmax is specified
    nbands *= 5 if lmax > 3 else 1 # for g-orbital which is definitely not possible for all ground state atoms
    
    return int(nbands)

ABACUS_INPUT_TEMPLATE = """INPUT_PARAMETERS
#Parameters (1.General)
suffix                         ABACUS #the name of main output directory
latname                        none #the name of lattice name
stru_file                      STRU #the filename of file containing atom positions
kpoint_file                    KPT #the name of file containing k points
pseudo_dir                     ../../../tests/PP_ORB/ #the directory containing pseudo files
orbital_dir                    ../../../tests/PP_ORB/ #the directory containing orbital files
pseudo_rcut                    15 #cut-off radius for radial integration
pseudo_mesh                    0 #0: use our own mesh to do radial renormalization; 1: use mesh as in QE
lmaxmax                        2 #maximum of l channels used
dft_functional                 default #exchange correlation functional
xc_temperature                 0 #temperature for finite temperature functionals
calculation                    scf #test; scf; relax; nscf; get_wf; get_pchg
esolver_type                   ksdft #the energy solver: ksdft, sdft, ofdft, tddft, lj, dp
ntype                          1 #atom species number
nspin                          1 #1: single spin; 2: up and down spin; 4: noncollinear spin
kspacing                       0 0 0  #unit in 1/bohr, should be > 0, default is 0 which means read KPT file
min_dist_coef                  0.2 #factor related to the allowed minimum distance between two atoms
nbands                         0 #number of bands
nbands_sto                     256 #number of stochastic bands
nbands_istate                  5 #number of bands around Fermi level for get_pchg calulation
symmetry                       1 #the control of symmetry
init_vel                       0 #read velocity from STRU or not
symmetry_prec                  1e-06 #accuracy for symmetry
symmetry_autoclose             1 #whether to close symmetry automatically when error occurs in symmetry analysis
nelec                          0 #input number of electrons
out_mul                        0 # mulliken  charge or not
noncolin                       0 #using non-collinear-spin
lspinorb                       0 #consider the spin-orbit interaction
kpar                           1 #devide all processors into kpar groups and k points will be distributed among each group
bndpar                         1 #devide all processors into bndpar groups and bands will be distributed among each group
out_freq_elec                  0 #the frequency ( >= 0) of electronic iter to output charge density and wavefunction. 0: output only when converged
dft_plus_dmft                  0 #true:DFT+DMFT; false: standard DFT calcullation(default)
rpa                            0 #true:generate output files used in rpa calculation; false:(default)
printe                         100 #Print out energy for each band for every printe steps
mem_saver                      0 #Only for nscf calculations. if set to 1, then a memory saving technique will be used for many k point calculations.
diago_proc                     12 #the number of procs used to do diagonalization
nbspline                       -1 #the order of B-spline basis
wannier_card                   none #input card for wannier functions
soc_lambda                     1 #The fraction of averaged SOC pseudopotential is given by (1-soc_lambda)
cal_force                      0 #if calculate the force at the end of the electronic iteration
out_freq_ion                   0 #the frequency ( >= 0 ) of ionic step to output charge density and wavefunction. 0: output only when ion steps are finished
device                         cpu #the computing device for ABACUS

#Parameters (2.PW)
ecutwfc                        50 ##energy cutoff for wave functions
ecutrho                        200 ##energy cutoff for charge density and potential
erf_ecut                       0 ##the value of the constant energy cutoff
erf_height                     0 ##the height of the energy step for reciprocal vectors
erf_sigma                      0.1 ##the width of the energy step for reciprocal vectors
fft_mode                       0 ##mode of FFTW
pw_diag_thr                    0.01 #threshold for eigenvalues is cg electron iterations
scf_thr                        1e-06 #charge density error
scf_thr_type                   2 #type of the criterion of scf_thr, 1: reci drho for pw, 2: real drho for lcao
init_wfc                       atomic #start wave functions are from 'atomic', 'atomic+random', 'random' or 'file'
init_chg                       atomic #start charge is from 'atomic' or file
chg_extrap                     atomic #atomic; first-order; second-order; dm:coefficients of SIA
out_chg                        0 #>0 output charge density for selected electron steps
out_pot                        0 #output realspace potential
out_wfc_pw                     0 #output wave functions
out_wfc_r                      0 #output wave functions in realspace
out_dos                        0 #output energy and dos
out_band                       0 #output energy and band structure
out_proj_band                  0 #output projected band structure
restart_save                   0 #print to disk every step for restart
restart_load                   0 #restart from disk
read_file_dir                  auto #directory of files for reading
nx                             0 #number of points along x axis for FFT grid
ny                             0 #number of points along y axis for FFT grid
nz                             0 #number of points along z axis for FFT grid
ndx                            0 #number of points along x axis for FFT smooth grid
ndy                            0 #number of points along y axis for FFT smooth grid
ndz                            0 #number of points along z axis for FFT smooth grid
cell_factor                    1.2 #used in the construction of the pseudopotential tables
pw_seed                        1 #random seed for initializing wave functions

#Parameters (3.Stochastic DFT)
method_sto                     2 #1: slow and save memory, 2: fast and waste memory
npart_sto                      1 #Reduce memory when calculating Stochastic DOS
nbands_sto                     256 #number of stochstic orbitals
nche_sto                       100 #Chebyshev expansion orders
emin_sto                       0 #trial energy to guess the lower bound of eigen energies of the Hamitonian operator
emax_sto                       0 #trial energy to guess the upper bound of eigen energies of the Hamitonian operator
seed_sto                       0 #the random seed to generate stochastic orbitals
initsto_ecut                   0 #maximum ecut to init stochastic bands
initsto_freq                   0 #frequency to generate new stochastic orbitals when running md
cal_cond                       0 #calculate electronic conductivities
cond_che_thr                   1e-08 #control the error of Chebyshev expansions for conductivities
cond_dw                        0.1 #frequency interval for conductivities
cond_wcut                      10 #cutoff frequency (omega) for conductivities
cond_dt                        0.02 #t interval to integrate Onsager coefficiencies
cond_dtbatch                   0 #exp(iH*dt*cond_dtbatch) is expanded with Chebyshev expansion.
cond_smear                     1 #Smearing method for conductivities
cond_fwhm                      0.4 #FWHM for conductivities
cond_nonlocal                  1 #Nonlocal effects for conductivities

#Parameters (4.Relaxation)
ks_solver                      dav #cg; dav; lapack; genelpa; scalapack_gvx; cusolver
scf_nmax                       100 ##number of electron iterations
relax_nmax                     1 #number of ion iteration steps
out_stru                       0 #output the structure files after each ion step
force_thr                      0.001 #force threshold, unit: Ry/Bohr
force_thr_ev                   0.0257112 #force threshold, unit: eV/Angstrom
force_thr_ev2                  0 #force invalid threshold, unit: eV/Angstrom
relax_cg_thr                   0.5 #threshold for switching from cg to bfgs, unit: eV/Angstrom
stress_thr                     0.5 #stress threshold
press1                         0 #target pressure, unit: KBar
press2                         0 #target pressure, unit: KBar
press3                         0 #target pressure, unit: KBar
relax_bfgs_w1                  0.01 #wolfe condition 1 for bfgs
relax_bfgs_w2                  0.5 #wolfe condition 2 for bfgs
relax_bfgs_rmax                0.8 #maximal trust radius, unit: Bohr
relax_bfgs_rmin                1e-05 #minimal trust radius, unit: Bohr
relax_bfgs_init                0.5 #initial trust radius, unit: Bohr
cal_stress                     0 #calculate the stress or not
fixed_axes                     None #which axes are fixed
fixed_ibrav                    0 #whether to preseve lattice type during relaxation
fixed_atoms                    0 #whether to preseve direct coordinates of atoms during relaxation
relax_method                   cg #bfgs; sd; cg; cg_bfgs;
relax_new                      1 #whether to use the new relaxation method
relax_scale_force              0.5 #controls the size of the first CG step if relax_new is true
out_level                      ie #ie(for electrons); i(for ions);
out_dm                         0 #>0 output density matrix
out_bandgap                    0 #if true, print out bandgap
use_paw                        0 #whether to use PAW in pw calculation
deepks_out_labels              0 #>0 compute descriptor for deepks
deepks_scf                     0 #>0 add V_delta to Hamiltonian
deepks_bandgap                 0 #>0 for bandgap label
deepks_out_unittest            0 #if set 1, prints intermediate quantities that shall be used for making unit test
deepks_model                    #file dir of traced pytorch model: 'model.ptg

#Parameters (5.LCAO)
basis_type                     pw #PW; LCAO in pw; LCAO
nb2d                           0 #2d distribution of atoms
gamma_only                     0 #Only for localized orbitals set and gamma point. If set to 1, a fast algorithm is used
search_radius                  -1 #input search radius (Bohr)
search_pbc                     1 #input periodic boundary condition
lcao_ecut                      50 #energy cutoff for LCAO
lcao_dk                        0.01 #delta k for 1D integration in LCAO
lcao_dr                        0.01 #delta r for 1D integration in LCAO
lcao_rmax                      30 #max R for 1D two-center integration table
out_mat_hs                     0 #output H and S matrix
out_mat_hs2                    0 #output H(R) and S(R) matrix
out_mat_dh                     0 #output of derivative of H(R) matrix
out_interval                   1 #interval for printing H(R) and S(R) matrix during MD
out_app_flag                   1 #whether output r(R), H(R), S(R), T(R), and dH(R) matrices in an append manner during MD
out_mat_t                      0 #output T(R) matrix
out_element_info               0 #output (projected) wavefunction of each element
out_mat_r                      0 #output r(R) matrix
out_wfc_lcao                   0 #ouput LCAO wave functions, 0, no output 1: text, 2: binary
bx                             0 #division of an element grid in FFT grid along x
by                             0 #division of an element grid in FFT grid along y
bz                             0 #division of an element grid in FFT grid along z

#Parameters (6.Smearing)
smearing_method                fixed #type of smearing_method: gauss; fd; fixed; mp; mp2; mv
smearing_sigma                 0.01 #energy range for smearing

#Parameters (7.Charge Mixing)
mixing_type                    broyden #plain; pulay; broyden
mixing_beta                    0.8 #mixing parameter: 0 means no new charge
mixing_ndim                    8 #mixing dimension in pulay or broyden
mixing_gg0                     1 #mixing parameter in kerker
mixing_beta_mag                -10 #mixing parameter for magnetic density
mixing_gg0_mag                 0 #mixing parameter in kerker
mixing_gg0_min                 0.1 #the minimum kerker coefficient
mixing_tau                     0 #whether to mix tau in mGGA calculation
mixing_dftu                    0 #whether to mix locale in DFT+U calculation

#Parameters (8.DOS)
dos_emin_ev                    -15 #minimal range for dos
dos_emax_ev                    15 #maximal range for dos
dos_edelta_ev                  0.01 #delta energy for dos
dos_scale                      0.01 #scale dos range by
dos_sigma                      0.07 #gauss b coefficeinet(default=0.07)
dos_nche                       100 #orders of Chebyshev expansions for dos

#Parameters (9.Molecular dynamics)
md_type                        nvt #choose ensemble
md_thermostat                  nhc #choose thermostat
md_nstep                       10 #md steps
md_dt                          1 #time step
md_tchain                      1 #number of Nose-Hoover chains
md_tfirst                      -1 #temperature first
md_tlast                       -1 #temperature last
md_dumpfreq                    1 #The period to dump MD information
md_restartfreq                 5 #The period to output MD restart information
md_seed                        -1 #random seed for MD
md_prec_level                  0 #precision level for vc-md
ref_cell_factor                1 #construct a reference cell bigger than the initial cell
md_restart                     0 #whether restart
lj_rcut                        8.5 #cutoff radius of LJ potential
lj_epsilon                     0.01032 #the value of epsilon for LJ potential
lj_sigma                       3.405 #the value of sigma for LJ potential
pot_file                       graph.pb #the filename of potential files for CMD such as DP
msst_direction                 2 #the direction of shock wave
msst_vel                       0 #the velocity of shock wave
msst_vis                       0 #artificial viscosity
msst_tscale                    0.01 #reduction in initial temperature
msst_qmass                     -1 #mass of thermostat
md_tfreq                       0 #oscillation frequency, used to determine qmass of NHC
md_damp                        1 #damping parameter (time units) used to add force in Langevin method
md_nraise                      1 #parameters used when md_type=nvt
cal_syns                       0 #calculate asynchronous overlap matrix to output for Hefei-NAMD
dmax                           0.01 #maximum displacement of all atoms in one step (bohr)
md_tolerance                   100 #tolerance for velocity rescaling (K)
md_pmode                       iso #NPT ensemble mode: iso, aniso, tri
md_pcouple                     none #whether couple different components: xyz, xy, yz, xz, none
md_pchain                      1 #num of thermostats coupled with barostat
md_pfirst                      -1 #initial target pressure
md_plast                       -1 #final target pressure
md_pfreq                       0 #oscillation frequency, used to determine qmass of thermostats coupled with barostat
dump_force                     1 #output atomic forces into the file MD_dump or not
dump_vel                       1 #output atomic velocities into the file MD_dump or not
dump_virial                    1 #output lattice virial into the file MD_dump or not

#Parameters (10.Electric field and dipole correction)
efield_flag                    0 #add electric field
dip_cor_flag                   0 #dipole correction
efield_dir                     2 #the direction of the electric field or dipole correction
efield_pos_max                 0.5 #position of the maximum of the saw-like potential along crystal axis efield_dir
efield_pos_dec                 0.1 #zone in the unit cell where the saw-like potential decreases
efield_amp                     0 #amplitude of the electric field

#Parameters (11.Gate field)
gate_flag                      0 #compensating charge or not
zgate                          0.5 #position of charged plate
relax                          0 #allow relaxation along the specific direction
block                          0 #add a block potential or not
block_down                     0.45 #low bound of the block
block_up                       0.55 #high bound of the block
block_height                   0.1 #height of the block

#Parameters (12.Test)
out_alllog                     0 #output information for each processor, when parallel
nurse                          0 #for coders
colour                         0 #for coders, make their live colourful
t_in_h                         1 #calculate the kinetic energy or not
vl_in_h                        1 #calculate the local potential or not
vnl_in_h                       1 #calculate the nonlocal potential or not
vh_in_h                        1 #calculate the hartree potential or not
vion_in_h                      1 #calculate the local ionic potential or not
test_force                     0 #test the force
test_stress                    0 #test the force
test_skip_ewald                0 #skip ewald energy

#Parameters (13.vdW Correction)
vdw_method                     none #the method of calculating vdw (none ; d2 ; d3_0 ; d3_bj
vdw_s6                         default #scale parameter of d2/d3_0/d3_bj
vdw_s8                         default #scale parameter of d3_0/d3_bj
vdw_a1                         default #damping parameter of d3_0/d3_bj
vdw_a2                         default #damping parameter of d3_bj
vdw_d                          20 #damping parameter of d2
vdw_abc                        0 #third-order term?
vdw_C6_file                    default #filename of C6
vdw_C6_unit                    Jnm6/mol #unit of C6, Jnm6/mol or eVA6
vdw_R0_file                    default #filename of R0
vdw_R0_unit                    A #unit of R0, A or Bohr
vdw_cutoff_type                radius #expression model of periodic structure, radius or period
vdw_cutoff_radius              default #radius cutoff for periodic structure
vdw_radius_unit                Bohr #unit of radius cutoff for periodic structure
vdw_cn_thr                     40 #radius cutoff for cn
vdw_cn_thr_unit                Bohr #unit of cn_thr, Bohr or Angstrom
vdw_cutoff_period   3 3 3 #periods of periodic structure

#Parameters (14.exx)
exx_hybrid_alpha               default #fraction of Fock exchange in hybrid functionals
exx_hse_omega                  0.11 #range-separation parameter in HSE functional
exx_separate_loop              1 #if 1, a two-step method is employed, else it will start with a GGA-Loop, and then Hybrid-Loop
exx_hybrid_step                100 #the maximal electronic iteration number in the evaluation of Fock exchange
exx_mixing_beta                1 #mixing_beta for outer-loop when exx_separate_loop=1
exx_lambda                     0.3 #used to compensate for divergence points at G=0 in the evaluation of Fock exchange using lcao_in_pw method
exx_real_number                0 #exx calculated in real or complex
exx_pca_threshold              0.0001 #threshold to screen on-site ABFs in exx
exx_c_threshold                0.0001 #threshold to screen C matrix in exx
exx_v_threshold                0.1 #threshold to screen C matrix in exx
exx_dm_threshold               0.0001 #threshold to screen density matrix in exx
exx_cauchy_threshold           1e-07 #threshold to screen exx using Cauchy-Schwartz inequality
exx_c_grad_threshold           0.0001 #threshold to screen nabla C matrix in exx
exx_v_grad_threshold           0.1 #threshold to screen nabla V matrix in exx
exx_cauchy_force_threshold     1e-07 #threshold to screen exx force using Cauchy-Schwartz inequality
exx_cauchy_stress_threshold    1e-07 #threshold to screen exx stress using Cauchy-Schwartz inequality
exx_ccp_rmesh_times            default #how many times larger the radial mesh required for calculating Columb potential is to that of atomic orbitals
exx_opt_orb_lmax               0 #the maximum l of the spherical Bessel functions for opt ABFs
exx_opt_orb_ecut               0 #the cut-off of plane wave expansion for opt ABFs
exx_opt_orb_tolerence          0 #the threshold when solving for the zeros of spherical Bessel functions for opt ABFs

#Parameters (16.tddft)
td_force_dt                    0.02 #time of force change
td_vext                        0 #add extern potential or not
td_vext_dire                   1 #extern potential direction
out_dipole                     0 #output dipole or not
out_efield                     0 #output dipole or not
ocp                            0 #change occupation or not
ocp_set                         #set occupation

#Parameters (17.berry_wannier)
berry_phase                    0 #calculate berry phase or not
gdir                           3 #calculate the polarization in the direction of the lattice vector
towannier90                    0 #use wannier90 code interface or not
nnkpfile                       seedname.nnkp #the wannier90 code nnkp file name
wannier_spin                   up #calculate spin in wannier90 code interface
out_wannier_mmn                1 #output .mmn file or not
out_wannier_amn                1 #output .amn file or not
out_wannier_unk                1 #output UNK. file or not
out_wannier_eig                1 #output .eig file or not
out_wannier_wvfn_formatted     1 #output UNK. file in text format or in binary format

#Parameters (18.implicit_solvation)
imp_sol                        0 #calculate implicit solvation correction or not
eb_k                           80 #the relative permittivity of the bulk solvent
tau                            1.0798e-05 #the effective surface tension parameter
sigma_k                        0.6 # the width of the diffuse cavity
nc_k                           0.00037 # the cut-off charge density

#Parameters (19.orbital free density functional theory)
of_kinetic                     wt #kinetic energy functional, such as tf, vw, wt
of_method                      tn #optimization method used in OFDFT, including cg1, cg2, tn (default)
of_conv                        energy #the convergence criterion, potential, energy (default), or both
of_tole                        1e-06 #tolerance of the energy change (in Ry) for determining the convergence, default=2e-6 Ry
of_tolp                        1e-05 #tolerance of potential for determining the convergence, default=1e-5 in a.u.
of_tf_weight                   1 #weight of TF KEDF
of_vw_weight                   1 #weight of vW KEDF
of_wt_alpha                    0.833333 #parameter alpha of WT KEDF
of_wt_beta                     0.833333 #parameter beta of WT KEDF
of_wt_rho0                     0 #the average density of system, used in WT KEDF, in Bohr^-3
of_hold_rho0                   0 #If set to 1, the rho0 will be fixed even if the volume of system has changed, it will be set to 1 automaticly if of_wt_rho0 is not zero
of_lkt_a                       1.3 #parameter a of LKT KEDF
of_full_pw                     1 #If set to 1, ecut will be ignored when collect planewaves, so that all planewaves will be used
of_full_pw_dim                 0 #If of_full_pw = true, dimention of FFT is testricted to be (0) either odd or even; (1) odd only; (2) even only
of_read_kernel                 0 #If set to 1, the kernel of WT KEDF will be filled from file of_kernel_file, not from formula. Only usable for WT KEDF
of_kernel_file                 WTkernel.txt #The name of WT kernel file.

#Parameters (20.dft+u)
dft_plus_u                     0 #true:DFT+U correction; false: standard DFT calcullation(default)
yukawa_lambda                  -1 #default:0.0
yukawa_potential               0 #default: false
omc                            0 #the mode of occupation matrix control
hubbard_u           0 #Hubbard Coulomb interaction parameter U(ev)
orbital_corr        -1 #which correlated orbitals need corrected ; d:2 ,f:3, do not need correction:-1

#Parameters (21.spherical bessel)
bessel_nao_ecut                50.000000 #energy cutoff for spherical bessel functions(Ry)
bessel_nao_tolerence           1e-12 #tolerence for spherical bessel root
bessel_nao_rcut                6 #radial cutoff for spherical bessel functions(a.u.)
bessel_nao_smooth              1 #spherical bessel smooth or not
bessel_nao_sigma               0.1 #spherical bessel smearing_sigma
bessel_descriptor_lmax         2 #lmax used in generating spherical bessel functions
bessel_descriptor_ecut         50.000000 #energy cutoff for spherical bessel functions(Ry)
bessel_descriptor_tolerence    1e-12 #tolerence for spherical bessel root
bessel_descriptor_rcut         6 #radial cutoff for spherical bessel functions(a.u.)
bessel_descriptor_smooth       1 #spherical bessel smooth or not
bessel_descriptor_sigma        0.1 #spherical bessel smearing_sigma

#Parameters (22.non-collinear spin-constrained DFT)
sc_mag_switch                  0 #0: no spin-constrained DFT; 1: constrain atomic magnetization
decay_grad_switch              0 #switch to control gradient break condition
sc_thr                         1e-06 #Convergence criterion of spin-constrained iteration (RMS) in uB
nsc                            100 #Maximal number of spin-constrained iteration
nsc_min                        2 #Minimum number of spin-constrained iteration
sc_scf_nmin                    2 #Minimum number of outer scf loop before initializing lambda loop
alpha_trial                    0.01 #Initial trial step size for lambda in eV/uB^2
sccut                          3 #Maximal step size for lambda in eV/uB
sc_file                        none #file name for parameters used in non-collinear spin-constrained DFT (json format)

#Parameters (23.Quasiatomic Orbital analysis)
qo_switch                      0 #0: no QO analysis; 1: QO analysis
qo_basis                       hydrogen #type of QO basis function: hydrogen: hydrogen-like basis, pswfc: read basis from pseudopotential
qo_thr                         1e-06 #accuracy for evaluating cutoff radius of QO basis function"""

class TestReadInput(unittest.TestCase):

    """use example input as test material
    ./SIAB/example_Si/SIAB_INPUT
    """
    
    def test_parse(self):
        
        self.maxDiff = None
        example = """
#--------------------------------------------------------------------------------
#1. CMD & ENV
#EXE_env    module purge && module load anaconda3_nompi gcc/9.2.0 elpa/2021.05.002/intelmpi2018 intelmpi/2018.update4 2>&1 && source activate pytorch110
 EXE_mpi    mpirun -np 1
 EXE_pw     abacus
#EXE_pw     /home/nic/wszhang/abacus/abacus222_intel-2018u4/ABACUS.mpi
#EXE_opt    /home/nic/wszhang/abacus/wszhang@github/abacus-develop/tools/SIAB/PyTorchGradient_dpsi/main.py

#-------------------------------------------------------------------------------- 
#2. Electronic calculatation
 element     Si          # Element Name
 Ecut        100         # in Ry
 Rcut        6 7         # in Bohr
 Pseudo_dir  /root/abacus-develop/pseudopotentials/SG15_ONCV_v1.0_upf
 Pseudo_name Si_ONCV_PBE-1.0.upf 
 sigma       0.01        # energy range for gauss smearing (in Ry) 

#--------------------------------------------------------------------------------
#3. Reference structure related parameters for PW calculation
#For the built-in structure types (including 'dimer', 'trimer' and 'tetramer'):
#STRU Name   #STRU Type  #nbands #MaxL   #nspin  #Bond Length list
 STRU1       dimer       8       2       1       1.8 2.0 2.3 2.8 3.8
 STRU2       trimer      10      2       1       1.9 2.1 2.6

#-------------------------------------------------------------------------------- 
#4. SIAB calculatation
 max_steps    200
# Orbital configure and reference target for each level
#LevelIndex   #Ref STRU Name    #Ref Bands   #InputOrb    #OrbitalConf
 Level1       STRU1             4            none         1s1p      
 Level2       STRU1             4            fix          2s2p1d    
 Level3       STRU2             6            fix          3s3p2d    

#--------------------------------------------------------------------------------
#5. Save Orbitals
#Index       #LevelNum    #OrbitalType
 Save1       Level1       Z           
 Save2       Level2       DZP         
 Save3       Level3       TZDP   
"""
        fsiab = str(uuid.uuid4())
        with open(fsiab, "w") as f:
            f.write(example)
        result = read(fsiab)
        os.remove(fsiab)
        self.assertDictEqual(result,
                            {'environment': '', 
                             'mpi_command': 'mpirun -np 1', 
                             'abacus_command': 'abacus', 
                             'pseudo_dir': '/root/abacus-develop/pseudopotentials/SG15_ONCV_v1.0_upf', 
                             'pseudo_name': 'Si_ONCV_PBE-1.0.upf', 
                             'ecutwfc': 100, 
                             'bessel_nao_rcut': [6, 7], 
                             'smearing_sigma': 0.01, 
                             'max_steps': 200, 
                             'reference_systems': [
                                 {'shape': 'dimer', 'nbands': 8, 'nspin': 1, 'bond_lengths': [1.8, 2.0, 2.3, 2.8, 3.8]}, 
                                 {'shape': 'trimer', 'nbands': 10, 'nspin': 1, 'bond_lengths': [1.9, 2.1, 2.6]}
                            ], 
                             'orbitals': [
                                 {'zeta_notation': 'Z', 'shape': 0, 'nbands_ref': 4, 'orb_ref': 'none'}, 
                                 {'zeta_notation': 'DZP', 'shape': 0, 'nbands_ref': 4, 'orb_ref': 'Z'}, 
                                 {'zeta_notation': 'TZDP', 'shape': 1, 'nbands_ref': 6, 'orb_ref': 'DZP'}]})

        example = """
 
#--------------------------------------------------------------------------------
#1. CMD & ENV
 EXE_mpi      mpirun -np 16
 EXE_pw       abacus

#-------------------------------------------------------------------------------- 
#2. Electronic calculatation
 element     Na  # element name 
 Ecut        100  # cutoff energy (in Ry)
 Rcut        10  # cutoff radius (in a.u.)
 Pseudo_dir  /root/deepmodeling/ABACUS-Pseudopot-Nao-Square/download/pseudopotentials/sg15_oncv_upf_2020-02-06/
 Pseudo_name Na_ONCV_PBE-1.0.upf
 sigma       0.01 # energy range for gauss smearing (in Ry)

#--------------------------------------------------------------------------------
#3. Reference structure related parameters for PW calculation
#For the built-in structure types (including 'dimer', 'trimer' and 'tetramer'):
#STRU Name   #STRU Type  #nbands #MaxL   #nspin  #Bond Length list 
STRU1       dimer       13      2       1      2.1 2.6 3.1 3.7 4.3
STRU2       trimer      18      2       1      2.8 3.4 4.1
STRU3       trimer      18      2       1      2.7 3.0 3.6
STRU4       trimer      18      2       1      2.6 3.2 3.8

#-------------------------------------------------------------------------------- 
#4. SIAB calculatation
 max_steps    3000
#Orbital configure and reference target for each level
#LevelIndex  #Ref STRU name  #Ref Bands  #InputOrb    #OrbitalConf 
 Level1      STRU1           auto        none        2s1p
 Level2      STRU1           auto        fix         4s2p1d
 Level3      STRU2           auto        fix         6s3p2d  
 Level4      STRU3           auto        fix         8s4p3d
 Level5      STRU4           auto        fix         10s5p4d

#--------------------------------------------------------------------------------
#5. Save Orbitals
#Index    #LevelNum   #OrbitalType 
 Save1    Level1      Z
 Save2    Level2      DZP
 Save3    Level3      TZDP
 Save4    Level4      QZTP
 Save5    Level5      5ZQP
"""     # thanks to Huanjing GONG provides this example
        fsiab = str(uuid.uuid4())
        with open(fsiab, "w") as f:
            f.write(example)
        result = read(fsiab)
        os.remove(fsiab)
        print(result)

    def test_keywords_translate(self):

        self.assertEqual(translate_oldinp_keyword("Ecut"), "ecutwfc")
        self.assertEqual(translate_oldinp_keyword("Rcut"), "bessel_nao_rcut")
        self.assertEqual(translate_oldinp_keyword("Pseudo_dir"), "pseudo_dir")
        self.assertEqual(translate_oldinp_keyword("sigma"), "smearing_sigma")      

    def test_unpack_siab_settings(self):
        self.maxDiff = None
        pseudo = {
            "element": "Si",
            "val_conf": [["1S"], ["1P"]],
            "z_val": 4.0
        }
        result = read("SIAB/example_Si/SIAB_INPUT")
        result = parse(result, pseudo)
        self.assertEqual(len(result), 5)
        self.assertEqual(result[0], list(zip(['dimer', 'trimer'], [[1.8, 2.0, 2.3, 2.8, 3.8], [1.9, 2.1, 2.6]])))
        self.assertListEqual(result[1], [
            {'pseudo_dir': '/root/abacus-develop/pseudopotentials/SG15_ONCV_v1.0_upf', 
             'ecutwfc': 100, 'bessel_nao_rcut': [6, 7], 'smearing_sigma': 0.01, 
             'nbands': 8, 'lmaxmax': 2, 'nspin': 1}, 
             {'pseudo_dir': '/root/abacus-develop/pseudopotentials/SG15_ONCV_v1.0_upf', 
              'ecutwfc': 100, 'bessel_nao_rcut': [6, 7], 'smearing_sigma': 0.01, 
              'nbands': 10, 'lmaxmax': 2, 'nspin': 1}
              ])
        self.assertDictEqual(result[2], {
            'jY_type': 'reduced',
            'optimizer': 'pytorch.SWAT', 
            'nthreads_rcut': -1,
            'max_steps': 200, 
            'spill_coefs': [2.0, 1.0], 
            'spill_thr': 1e-08,
            'orbitals': [
                {'nzeta': [1, 1], 
                 'nzeta_from': None, 
                 'nbands_ref': 4, 
                 'folder': 0}, 
                {'nzeta': [2, 2, 1], 
                 'nzeta_from': [1, 1], 
                 'nbands_ref': 4, 
                 'folder': 0}, 
                {'nzeta': [3, 3, 2], 
                 'nzeta_from': [2, 2, 1], 
                 'nbands_ref': 6, 
                 'folder': 1}]
        })
        self.assertDictEqual(result[3], {'environment': '', 
                                         'mpi_command': 'mpirun -np 1', 
                                         'abacus_command': 'abacus'})
        self.assertDictEqual(result[4], {'element': 'Si', 
                                         'pseudo_dir': '/root/abacus-develop/pseudopotentials/SG15_ONCV_v1.0_upf', 
                                         'pseudo_name': 'Si_ONCV_PBE-1.0.upf'})

    def test_abacus_params(self):
        params = abacus_params()
        self.assertGreater(len(params), 0)

    def test_compatibility_convert(self):
        clean_oldversion_input = {
            'EXE_mpi': 'mpirun -np 1', 
            'EXE_pw': 'abacus --version', 
            'element': 'Si', 
            'Ecut': 100, 
            'Rcut': [6, 7], 
            'Pseudo_dir': '/root/abacus-develop/pseudopotentials/SG15_ONCV_v1.0_upf', 
            'Pseudo_name': 'Si_ONCV_PBE-1.0.upf', 
            'sigma': 0.01, 
            'STRU1': ['dimer', 8, 2, 1, 1.8, 2.0, 2.3, 2.8, 3.8], 
            'STRU2': ['trimer', 10, 2, 1, 1.9, 2.1, 2.6], 
            'max_steps': 200, 
            'Level1': ['STRU1', 4, 'none', '1s1p'], 
            'Level2': ['STRU1', 4, 'fix', '2s2p1d'], 
            'Level3': ['STRU2', 6, 'fix', '3s3p2d'], 
            'Save1': ['Level1', 'SZ'], 
            'Save2': ['Level2', 'DZP'], 
            'Save3': ['Level3', 'TZDP'], 
            'EXE_opt': '/opt_orb_pytorch_dpsi/main.py (default)', 
            'EXE_env': '', 
            'environment': '', 
            'mpi_command': 'mpirun -np 1', 
            'abacus_command': 'abacus', 
            'optimizer': 'pytorch.SWAT', 
            'spill_coefs': [2.0, 1.0]
        }
        result = convert_oldinp_tojson(clean_oldversion_input)
        clean_oldversion_input = {
            'EXE_mpi': 'mpirun -np 1', 
            'EXE_pw': 'abacus --version', 
            'element': 'Si', 
            'Ecut': 100, 
            'Rcut': [6, 7], 
            'Pseudo_dir': '/root/abacus-develop/pseudopotentials/SG15_ONCV_v1.0_upf', 
            'Pseudo_name': 'Si_ONCV_PBE-1.0.upf', 
            'sigma': 0.01, 
            'STRU1': ['dimer', 8, 2, 1, "auto"], 
            'STRU2': ['trimer', 10, 2, 1, 1.9, 2.1, 2.6], 
            'max_steps': 200, 
            'Level1': ['STRU1', 4, 'none', '1s1p'], 
            'Level2': ['STRU1', 4, 'fix', '2s2p1d'], 
            'Level3': ['STRU2', 6, 'fix', '3s3p2d'], 
            'Save1': ['Level1', 'SZ'], 
            'Save2': ['Level2', 'DZP'], 
            'Save3': ['Level3', 'TZDP'], 
            'EXE_opt': '/opt_orb_pytorch_dpsi/main.py (default)', 
            'EXE_env': '', 
            'environment': '', 
            'mpi_command': 'mpirun -np 1', 
            'abacus_command': 'abacus', 
            'optimizer': 'pytorch.SWAT', 
            'spill_coefs': [2.0, 1.0]
        }
        result = convert_oldinp_tojson(clean_oldversion_input)

    def test_abacus_settings(self):
        user_settings = read("SIAB/example_Si/SIAB_INPUT")
        result = abacus_settings(user_settings, 
                                 minimal_basis=[["2S"], ["2P"]],
                                 z_val=4.0)
        self.assertEqual(len(result), 2)
        result = abacus_settings(user_settings, 
                                    minimal_basis=[["2S"], [], ["3D"]],
                                    z_val=4.0)
        self.assertEqual(len(result), 2)
        for i in result:
            self.assertEqual(i["lmaxmax"], 2)
        result = abacus_settings(user_settings, 
                                    minimal_basis=[["2S"], ["2P"], ["3D"]],
                                    z_val=4.0)
        self.assertEqual(len(result), 2)
        for i in result:
            self.assertEqual(i["lmaxmax"], 3)

    def test_nzetagen(self):
        result = nzetagen("DZP", [2, 1, 1])
        self.assertEqual(result, [4, 2, 2, 1])
        result = nzetagen("1s1p", [2, 1, 1])
        self.assertEqual(result, [1, 1])
        result = nzetagen("2s2p1d", [2, 1, 1])
        self.assertEqual(result, [2, 2, 1])
        result = nzetagen("1s0p1d", [2, 1, 1])
        self.assertEqual(result, [1, 0, 1])
        result = nzetagen([2, 2, 1], [2, 1, 1])
        self.assertEqual(result, [2, 2, 1])
    
    def test_nbands_from_str(self):
        result = nbands_from_str(4, "dimer", 100)
        self.assertEqual(result, 4)
        result = nbands_from_str(6, "trimer", 100)
        self.assertEqual(result, 6)
        result = nbands_from_str("occ", "dimer", 4)
        self.assertEqual(result, 4)
        result = nbands_from_str("occ", "trimer", 6)
        self.assertEqual(result, 9)
        result = nbands_from_str("occ+5", "dimer", 4)
        self.assertEqual(result, 9)
        result = nbands_from_str("occ-2", "trimer", 6)
        self.assertEqual(result, 7)
        result = nbands_from_str("auto", "dimer", 4)
        self.assertEqual(result, "auto")
        result = nbands_from_str("auto", "trimer", 6)
        self.assertEqual(result, "auto")
        result = nbands_from_str("all", "dimer", 4)
        self.assertEqual(result, 8)
        result = nbands_from_str("all", "trimer", 6)
        self.assertEqual(result, 18)

    def test_cal_nbands_fill_lmax(self):
        result = cal_nbands_fill_lmax(1, 0, 1) # Hydrogen, lmax = 1
        self.assertEqual(result, 5)
        result = cal_nbands_fill_lmax(10, 46, 3) # Barium, lmax = 3
        self.assertEqual(result, 12)
        # what is strange is Barium pseudopotential take reference state as 6s1 5d1 instead of 6s2...

if __name__ == "__main__":
    unittest.main()

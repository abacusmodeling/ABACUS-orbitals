"""This module is for making compatible with previous version of PTG_dpsi, 
The optimizer needs the "INPUT.json" file for performing optimization task. """

"""the parsed parameters are organized inside like this:
{
    "environment": "module load intel/2019.5.281 openmpi/3.1.4 intel-mkl/2019.5.281 intel-mpi/2019.5.281",
    "mpi_command": "mpirun -np 1",
    "abacus_command": "abacus",

    "pseudo_dir": "./download/pseudopotentials/sg15_oncv_upf_2020-02-06/1.2",
    "pesudo_name": "Fe_ONCV_PBE-1.2.upf",
    "ecutwfc": 100,
    "bessel_nao_rcut": [6, 7, 8, 9, 10],
    "smearing_sigma": 0.015,

    "optimizer": "pytorch.SWAT",
    "spill_coefs": [2.0, 1.0],
    "max_steps": 9000,

    "reference_systems": [
        {
            "shape": "dimer",
            "nbands": 8,
            "nspin": 1,
            "bond_lengths": "auto"
        },
        {
            "shape": "trimer",
            "nbands": 10,
            "nspin": 1,
            "bond_lengths": [1.9, 2.1, 2.6]
        }
    ],
    "orbitals": [
        {
            "zeta_notation": "SZ",
            "shape": "dimer",
            "nbands_ref": 4,
            "orb_ref": "none"
        },
        {
            "zeta_notation": "DZP",
            "shape": "dimer",
            "nbands_ref": 4,
            "orb_ref": "SZ"
        },
        {
            "zeta_notation": "TZDP",
            "shape": "trimer",
            "nbands_ref": 6,
            "orb_ref": "DZP"
        }
    ]
}
"""

"""
prepare_SIAB_INPUT() seems will be called each time when a
new rcut and new level of orbitals, say each task that can
define one unique numerical atomic orbital (NAO)

variables appear in function prepare_SIAB_INPUT()
refSTRU_Level: ["STRU1", "STRU2"]
orbConf_Level: [
#   element1, element2, ...
    [2s2p1d, 4s2p2d1f, ...], # level1
    [2s2p1d, 4s2p2d1f, ...], # level2
    ...
]
refBands_Level: [
# bl: bond length
#   bl1, bl2, ...
    [18, 18, 18, 18, 18], # level1
    [18, 18, 18, 18, 18], # level2
    [27, 27, 27]          # level3
]
BL_STRU: {
    "STRU1": [bl1, bl2, ...],
    "STRU2": [bl1, bl2, ...]
}
nBL_STRU: {
    "STRU1": 5,
    "STRU2": 3
}
fixPre_Level: [
    "none", "fix", "fix", ... 
]
"""

def scan_folder_consistency(folders: list):
    """check if all folders have the same shape, returns 
    [True/False, element/None, shape/None, bond_lengths/None]"""
    elements = []
    shapes = []
    bond_lengths = []
    for folder in folders:
        elements.append(folder.split("-")[0])
        shapes.append(folder.split("-")[1])
        bond_lengths.append(float(folder.split("-")[2]))
    if len(set(shapes)) == 1 and len(set(elements)) == 1:
        return True, elements[0], shapes[0], bond_lengths
    else:
        print("check Folders return false: ", folders)
        return False, None, None, None

def ov_parameters(element: str,
                  orbital_config: list,
                  bessel_nao_rcut: float,
                  lmax: int,
                  ecutwfc: float,
                  dr: float = 0.01,
                  opt_maxsteps: int = 1000,
                  lr: float = 0.03,
                  calc_kinetic_ener: bool = False,
                  calc_smooth: bool = True
                  ):
    """this function is for dumping the old "INPUT.json" file for performing optimization
    parameters for SINGLE ELEMENT, to get identical "info" section contents, call merge_param
    function and feed list of what this function returns for elements"""

    keys = ["Nt_all", "Nu", "Rcut", "dr", "Ecut", "lr", "cal_T", "cal_smooth", "max_steps"]
    return dict(zip(keys, [[element], 
                           {element: orbital_config+[0]*(lmax+1-len(orbital_config))}, # this is very ugly
                           {element: bessel_nao_rcut}, 
                           {element: dr}, 
                           {element: ecutwfc},
                           lr, calc_kinetic_ener, calc_smooth, opt_maxsteps]
                    ))

def merge_ovparam(params: list):
    """this function is for merging all parameters of each element packed in dicts organized
    in one list as input, into one dict. 
    Except the "Nt_all", which would be a list of element after merging, and lr, cal_T, 
     cal_smooth and max_steps whose values would be shared by all elements and will still be
    scalar values, all other keys' values would be dicts with element as key and value 
    as the corresponding value of that element."""
    """this function corresponds to original version SIAB.py line 411 to 418"""
    keys_to_merge = ["Nu", "Rcut", "dr", "Ecut"]
    """check if all dicts in params have the same values for lr, cal_T, cal_smooth and max_steps"""
    for key in ["lr", "cal_T", "cal_smooth", "max_steps"]:
        for param in params:
            if param[key] != params[0][key]:
                raise ValueError(f"params have different values for {key}")
    """merge the dicts in params"""
    merged = {
        "Nt_all": [param["Nt_all"] for param in params],
        "lr": params[0]["lr"],
        "cal_T": params[0]["cal_T"],
        "cal_smooth": params[0]["cal_smooth"],
        "max_steps": params[0]["max_steps"]
    }
    for key in keys_to_merge:
        merged[key] = dict(zip(merged["Nt_all"], [param[key] for param in params]))
    return merged

def ov_weights(reference_states: list):
    """for each level, generate the "weights" section in the "INPUT.json" file
    but actually the "weight" is all the same as 1 for reference structures of
    one level of numerical orbitals. But is that always reasonable?
    """
    result = []
    def type_trais_isallsame(obj: list):
        """check if all elements in obj are the same type"""
        if len(obj) == 0:
            return True, None
        else:
            return all(isinstance(obj[0], type(item)) for item in obj), type(obj[0])
    for reference_state in reference_states:
        is_same, type_ = type_trais_isallsame(reference_state)
        if not is_same:
            raise TypeError("elements in reference_states should be all str or all int")
        # 20240204: seems should provide the first branch all the time, the second is not
        # directly valid
        if type_ == str:
        #if True:
            result.append(
                dict(zip(["stru", "bands_file"], [
                    [1]*len(reference_state),
                    reference_state
                ]))
            )
        elif type_ == int:
            result.append(
                dict(zip(["stru", "bands_range"], [
                    [1]*len(reference_state),
                    reference_state
                ]))
            )
        else:
            raise TypeError("elements in reference_states should be all str or all int")
    return result

def ov_V():
    """generate the "V_info" section (shared by all (rcut, level)-pairs)
    in the "INPUT.json" file for performing optimization task"""
    return {
        "init_from_file": True,
        "same_band": True
    }

def ov_ovlps_qsv(element: str,
                 reference_shapes: list,
                 bond_lengths: list,
                 orbitals: list,
                 om_shortname: bool,
                 rcut: float = 10.0):
    """set path of file where overlap_q, s and v are stored for ALL LEVELS OF ORBITALS IN
    ONE SHOT, organized in one list, each element corresponds to one level of orbitals

    For example this function returns:
    ```
    [
        [# level1
            ["H-dimer-0.8/orb_matrix.0.dat", "H-dimer-1.2/orb_matrix.0.dat"], # origin key
            [["H-dimer-0.8/orb_matrix.1.dat", "H-dimer-1.2/orb_matrix.1.dat"]] # linear key
        ],
        [
        ...
        ]
        ...
    ]
    ```
    Above is the case for old abacus version. For abacus version >=3.5.1, there are some
    trivial differences because different bessel_nao_rcut know is calculated in one shot
    by abacus. For example this function returns:
    ```
    [
        [# level1
            ["H-dimer-0.8/orb_matrix_rcut10.0deriv0.dat", "H-dimer-1.2/orb_matrix_rcut10.0deriv0.dat"], # origin key
            [["H-dimer-0.8/orb_matrix_rcut10.0deriv1.dat", "H-dimer-1.2/orb_matrix_rcut10.0deriv1.dat"]] # linear key
        ],
        [
        ...
        ]
        ...
    ]
    ```
    """
    ovlp_qsv_header = "orb_matrix"

    if om_shortname:
        ovlp_qsv0 = ovlp_qsv_header + ".0.dat"
        ovlp_qsv1 = ovlp_qsv_header + ".1.dat"
    else:
        ovlp_qsv0 = ovlp_qsv_header + "_rcut" + str(rcut) + "deriv0.dat"
        ovlp_qsv1 = ovlp_qsv_header + "_rcut" + str(rcut) + "deriv1.dat"

    result = [[] for _ in range(len(orbitals))]
    for iorb, orbital in enumerate(orbitals): # for each level
        shape = scan_folder_consistency(orbital["folder"])[2]
        # 20240204: seems should provide the first branch all the time, the second is not
        # directly valid
        #if orbital["nbands_ref"] == "auto":
        if True:
            folder_header = "-".join([element, shape])
            ishape = reference_shapes.index(shape)
            result[iorb] = [
                ["-".join(
                    [folder_header, str(bond_length)]
                    ) + "/" + ovlp_qsv0 for bond_length in bond_lengths[ishape]],
                [[
                "-".join(
                    [folder_header, str(bond_length)]
                    ) + "/" + ovlp_qsv1 for bond_length in bond_lengths[ishape]]]
            ]
        elif isinstance(orbital["nbands_ref"], int):
            ishape = reference_shapes.index(shape)
            result[iorb] = [
                [orbital["nbands_ref"]]*len(bond_lengths[ishape]),
                [[orbital["nbands_ref"]]*len(bond_lengths[ishape])]
            ]
        else:
            raise TypeError("nbands_ref should be 'auto' or exact number of bands")
    return result

def ov_reference_states(element: str,
                        reference_shapes: list,
                        bond_lengths: list,
                        orbitals: list):
    """set reference states for ALL LEVELS OF ORBITALS IN ONE SHOT, organized
    in one list, each element corresponds to one level of orbitals
    
    For example this function returns:
    ```
    [
        ['H-dimer-0.8/istate.info', 'H-dimer-1.2/istate.info'], 
        [4, 4], 
        ['H-trimer-0.8/istate.info', 'H-trimer-1.2/istate.info', 'H-trimer-1.6/istate.info']
    ]
    ```
    it means for level1, the reference state is read from files in the first list, and
    from the name of folder can be known that the reference state is for dimer with bond
    length of 0.8 and 1.2, 
    and for level2, the number of reference states is read explicitly specified by users,
    as 4, 
    and for level3, the reference state is read from files in the third list, and from
    the name of folder can be known that the reference state is for trimer with bond
    length of 0.8, 1.2 and 1.6.
    """
    # allocate
    result = [[] for _ in range(len(orbitals))]
    for iorb, orbital in enumerate(orbitals):
        shape = scan_folder_consistency(orbital["folder"])[2]
        # 20240204: seems should provide the first branch all the time, the second is not
        # directly valid
        if orbital["nbands_ref"] == "auto":
        #if True:
            # actually it is not the correct choice for "auto", instead, a more reasonable
            # choice is including all occupied bands, and also unoccupied bands as many
            # as the occupied. In future version, "auto" will be changed to this.
            folder_header = element + "-" + shape
            ishape = reference_shapes.index(shape)
            folders = [folder_header + "-" + str(bond_length) for bond_length in bond_lengths[ishape]]
            result[iorb] = [folder + "/OUT.%s/istate.info"%folder for folder in folders]

        elif isinstance(orbital["nbands_ref"], int):
            ishape = reference_shapes.index(shape)
            result[iorb] = [orbital["nbands_ref"]]*len(bond_lengths[ishape])
        else:
            raise TypeError("nbands_ref should be 'auto' or exact number of bands")
    
    return result, ov_weights(result)

def ov_c_init(orbitals: list, folder: str = None):
    """this function generates setting for initial guess. The outer loop to generate orbitals will
    proceed level by level then rcut by rcut, which means only if all orbitals with the same rcut
    have been generated, will program start the generation of the next rcut. However, it is more
    appropriate to create folder and save all temporary files into that folder. What is the most
    important temporary file is the ORBITAL_RESULTS.txt, which contains the optimized orbitals.
    
    """
    # add "/" to the end of folder if it is not None
    if folder is not None:
        folder = folder + "/" if folder[-1] != "/" else folder
    result = [{} for _ in range(len(orbitals))]
    for il, level in enumerate(orbitals):
        if level["nzeta_from"] is None or il == 0:
            result[il] = {
                "init_from_file": False,
            }
        else:
            orbital_results = folder if folder is not None else ""
            orbital_results += "Level%s.ORBITAL_RESULTS.txt"%(il - 1)
            result[il] = {"init_from_file": True, "C_init_file": orbital_results, "opt_C_read": False} 
            # discard the "opt_C_read" further support
    return result

import uuid
import time
def convert(calculation_setting: dict,
            siab_settings: dict):
    """
    convert from the new version of SIAB input to the old version of SIAB input, to make
    compatible with the old version of SIAB optimizer. This works in the way same as
    generator, therefore use
    ```python
    for PytorchSWAT_inp, rcut, iorb in convert(calculation_setting, siab_settings):
        pass
    ```
    to get the old SIAB input in the form of dict, rcut value for present orbital and index
    of orbital within present rcut.

    Args:
        `calculation_settings`: because the old version does not organize information well,
        there are many information that can be obtained from external files it requires, 
        many variables in pytorch_swat are initialized in "design-less" way, the calculation
        setting is therefore an *ad hoc*.
    """
    element = None
    reference_shapes = []
    bond_lengths = []
    for orbital in siab_settings["orbitals"]:
        result = scan_folder_consistency(orbital["folder"])
        if not result[0]:
            raise ValueError("all folders for one orbital should have the same shape")
        if element is None:
            element = result[1]
        elif element != result[1]:
            raise ValueError("all folders should have the same element")
        reference_shapes.append(result[2])
        bond_lengths.append(result[3])
    # ----------------------------------------------------------------------------------------------
    # for SIAB and PTG_dpsi original version, can only accept the case where only one bessel_nao_rcut
    # is given each ABACUS run, in this case, the overlap matrix will be output like orb_matrix.0.dat
    # and orb_matrix.1.dat. However since ABACUS 3.5.1, bessel_nao_rcut can support multiple values
    # and can calculate for all values in one-shot. In this case overlap matrices for different rcut
    # are distinguished by another naming convention: orb_matrix_rcut[R]deriv[D].dat, where R will be
    # corresponding rcut and D will be 0 or 1.
    om_shortname = False if len(calculation_setting["bessel_nao_rcut"]) > 1 else True
    # ----------------------------------------------------------------------------------------------
    # for each rcut and each orbital configuration, yield tuple ("input", rcut, iorb), where the 
    # "input" is the one for pytorch_swat spillage submodule.
    for rcut in calculation_setting["bessel_nao_rcut"]:
        ovlp_qsv = ov_ovlps_qsv(element=element,
                                reference_shapes=reference_shapes,
                                bond_lengths=bond_lengths,
                                orbitals=siab_settings["orbitals"],
                                om_shortname=om_shortname,
                                rcut=rcut)
        istates = ov_reference_states(element=element,
                                      reference_shapes=reference_shapes,
                                      bond_lengths=bond_lengths,
                                      orbitals=siab_settings["orbitals"])
        # to make compatible with parallelization in process level, should save the ORBITAL_RESULTS.txt
        # in different folders for different rcut, a combination of element and rcut with timestamp
        # can be used to identify the folder.
        ecutwfc = calculation_setting["ecutwfc"]
        foldername = str(uuid.uuid3(uuid.NAMESPACE_DNS, f"{element}_gga_{ecutwfc}Ry_{rcut}au.orbgen"))
        # then generate input file for old version orbital optimizer
        c_init = ov_c_init(orbitals=siab_settings["orbitals"], folder=foldername)
        v = ov_V()
        info = [ov_parameters(element=element,
                              orbital_config=orbital["nzeta"],
                              bessel_nao_rcut=rcut,
                              lmax=orbital["lmax"],
                              ecutwfc=ecutwfc,
                              opt_maxsteps=siab_settings["max_steps"])
                for orbital in siab_settings["orbitals"]]
        # level by level, yield the input for pytorch_swat
        for iorb in range(len(siab_settings["orbitals"])):
            """there is correlation between different sections, not happy with this"""
            if "opt_C_read" in c_init[iorb]:
                if c_init[iorb]["opt_C_read"]:
                    if c_init[iorb]["init_from_file"]:
                        info[iorb]["lr"] = 0.001
            yield {
                "file_list": dict(zip(["origin", "linear"], ovlp_qsv[iorb])),
                "info": info[iorb],
                "weight": istates[1][iorb],
                "C_init_info": c_init[iorb],
                "V_info": v,
                "spill_coefs": siab_settings["spill_coefs"]
            }, foldername, iorb

def unpack(orb_gen: dict) -> dict:
    """convert information collect in old version input to dict like
    ```python
    {
        "element": "Co",
        "ecutwfc": 100.0,
        "rcut": 6,
        "zeta_notation": "2s1p1d"
    }
    ```
    , this dict can identify explicitly an orbital
    """
    symbols = ["s", "p", "d", "f", "g", "h", "i", "j", "k", "l"]
    
    element = orb_gen["info"]["Nt_all"][0]
    ecutwfc = orb_gen["info"]["Ecut"][element]
    rcut = orb_gen["info"]["Rcut"][element]

    orbital_config = orb_gen["info"]["Nu"][element]
    zeta_notation = ""
    for i, n in enumerate(orbital_config):
        if n != 0:
            zeta_notation += str(n) + symbols[i]
    
    return {
        "element": element,
        "ecutwfc": ecutwfc,
        "rcut": rcut,
        "zeta_notation": zeta_notation,
    }

def folder(unpacked_orb: dict, additional_suffix: str = None) -> str:
    """this will generate folder path like:
    `Co_2s1p1d/6au_100Ry`
    """
    element = unpacked_orb["element"]
    ecutwfc = str(unpacked_orb["ecutwfc"]) + "Ry"
    rcut = str(unpacked_orb["rcut"]) + "au"
    zeta_notation = unpacked_orb["zeta_notation"]
    
    if additional_suffix is not None:
        zeta_notation = additional_suffix + "_" + zeta_notation
    layer1 = "_".join([element, zeta_notation])
    layer2 = "_".join([rcut, ecutwfc])
    return "/".join([layer1, layer2])
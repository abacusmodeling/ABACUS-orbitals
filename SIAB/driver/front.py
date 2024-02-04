# interface to initialize
import os
import SIAB.io.read_input as siri
import SIAB.io.pseudopotential.api as sipa
def initialize(version: str = "0.1.0",
               fname: str = "./SIAB_INPUT", 
               pseudopotential_check: bool = True):
    """initialization of numerical atomic orbitals generation task, 
    1. read input file named as fname, will convert to new version inside package
    2. check the existence of pseudopotential file, if pseudopotential_check is True, go on
    3. unpack the input file, return set of parameters:
        - reference_shapes: list of reference shapes, e.g. ["dimer", "trimer"]
        - bond_lengths: list of bond lengths for each reference shape, e.g. [[1.0, 1.1], [1.0, 1.1, 1.2]]
        - calculation_settings: list of calculation settings, e.g. [{"bessel_nao_rcut": 5.0, "bessel_nao_lmax": 5}, {"bessel_nao_rcut": 5.0, "bessel_nao_lmax": 5}]
            for each reference shape
        - siab_settings: dict of SIAB settings specially for SIAB optimization tasks.
        - env_settings: tuple of environment settings, contain environment variables, abacus executable, and mpi executable
        - general: dict of general settings, contain element symbol, pseudo_dir, pseudo_name
    """

    fname = fname.strip().replace("\\", "/")
    user_settings = siri.parse(fname=fname, version=version)
    # pseudopotential check
    fpseudo = user_settings["pseudo_dir"]+"/"+user_settings["pseudo_name"]
    if not os.path.exists(fpseudo) and pseudopotential_check: # check the existence of pseudopotential file
        raise FileNotFoundError(
            "Pseudopotential file %s not found"%fpseudo)
    else:
        pseudo = sipa.towards_siab(fname=fpseudo)
    unpacked = siri.unpack_siab_input(user_settings, 
                                      symbol=pseudo["element"], 
                                      minimal_basis=pseudo["valence_electron_configuration"])
    return unpacked

# interface to abacus
import SIAB.interface.submit as sis
def abacus(general: dict,
           reference_shapes: list,
           bond_lengths: list,
           calculation_settings: list,
           env_settings: dict,
           test: bool = True):
    """abacus interface for calling iteratively the abacus executable, generate reference
    wavefunctions and overlap between KS states and Truncated Spherical Bessel Functions.
    Will save as orb_matrix.0.dat and orb_matrix.1.dat.
    However for new version of abacus, which can accept multiple bessel_nao_rcut input,
    output matrices will be saved as orb_matrix_rcutRderivD.0.dat and orb_matrix_rcutRderivD.1.dat
    where R and D are the corresponding bessel_nao_rcut and order of derivatives of the
    wavefunctions.
    
    For introduction of input params, see annotation of initialize() function in this file.
    
    Iteration will return a list of list of folders in the save sequence of reference shapes,
    according to present implemented nomenclature of folders, if, element is Si, what returns
    would be like:
    [
        ["Si_dimer_1.0", "Si_dimer_1.1"], ["Si_trimer_1.0", "Si_trimer_1.1", "Si_trimer_1.2"]
    ]
    if the bond_lengths is given as [[1.0, 1.1], [1.0, 1.1, 1.2]]
    """
    return sis.iterate(general=general,
                       reference_shapes=reference_shapes,
                       bond_lengths=bond_lengths,
                       calculation_settings=calculation_settings,
                       env_settings=env_settings,
                       test=test)

# interface to Spillage optimization
import SIAB.interface.old_version as siov
import SIAB.opt_orb_pytorch_dpsi.main as soopdm # old version of backend
import SIAB.io.restart as sisrt
import SIAB.spillage as spill                   # new version of backend
def spillage(folders: list,
             calculation_settings: list,
             siab_settings: dict,
             abacus_version: str = "<3.5.1",
             siab_version: str = "0.1.0"):
    """spillage interface
    For being compatible with old version, the one without refactor, there exposes
    a parameter siab_version, which is the version of SIAB, default is "0.1.0".
    """
    for orbital in siab_settings["orbitals"]:
        orbital["folder"] = folders[orbital["folder"]]
    if siab_version == "0.1.0":
        for orb_gen in siov.convert(calculation_setting=calculation_settings[0],
                                    siab_settings=siab_settings,
                                    abacus_version=abacus_version):
            """the iteration here will be processed first by rcut and second by zeta notation"""
            print(orb_gen)
            soopdm.main(params=orb_gen)
            sisrt.checkpoint(src="./", dst = "./", )
            exit()
    else:
        raise NotImplementedError("SIAB version %s is not supported yet"%siab_version)

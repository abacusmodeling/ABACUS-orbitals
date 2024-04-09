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
        pseudopotential = sipa.towards_siab(fname=fpseudo)
    unpacked = siri.unpack_siab_input(user_settings, pseudopotential)
    return unpacked

# interface to abacus
import SIAB.interface.abacus as sia
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
        ["Si-dimer-1.0", "Si-dimer-1.1"], ["Si-trimer-1.0", "Si-trimer-1.1", "Si-trimer-1.2"]
    ]
    if the bond_lengths is given as [[1.0, 1.1], [1.0, 1.1, 1.2]]
    """
    return sia.run_all(general=general,
                       reference_shapes=reference_shapes,
                       bond_lengths=bond_lengths,
                       calculation_settings=calculation_settings,
                       env_settings=env_settings,
                       test=test)

# interface to Spillage optimization
import SIAB.interface.old_version as siov
import SIAB.spillage.pytorch_swat.api as SPS_api  # old version of backend
import SIAB.include.citation as sicite
# import SIAB.spillage.something.api as SpillageSomething_api  # new version of backend
def spillage(folders: list,
             calculation_settings: list,
             siab_settings: dict,
             siab_version: str = "0.1.0"):
    """spillage interface
    For being compatible with old version, the one without refactor, there exposes
    a parameter siab_version, which is the version of SIAB, default is "0.1.0".

    Parameters:
    folders: list of list of str, the folders for each reference shape and bond length,
    like
    ```python
    [
        ["Si-dimer-1.0", "Si-dimer-1.1"],
        ["Si-trimer-1.0", "Si-trimer-1.1", "Si-trimer-1.2"]
    ]
    ```
    calculation_settings: list of dict, the contents of INPUT for each reference shape,
    like
    ```python
    [
        {'pseudo_dir': '/root/abacus-develop/pseudopotentials/SG15_ONCV_v1.0_upf', 
            'ecutwfc': 100, 
            'bessel_nao_rcut': [6, 7], 
            'smearing_sigma': 0.01, 
            'nbands': 8, 
            'lmaxmax': 2, 
            'nspin': 1}, 
        {'pseudo_dir': '/root/abacus-develop/pseudopotentials/SG15_ONCV_v1.0_upf', 
            'ecutwfc': 100, 
            'bessel_nao_rcut': [6, 7], 
            'smearing_sigma': 0.01, 
            'nbands': 10, 
            'lmaxmax': 2, 
            'nspin': 1}
    ]
    ```
    siab_settings: dict, the settings for SIAB optimization tasks, including informations
    all about the orbitals, like
    ```python
    {
        'optimizer': 'pytorch.SWAT', 
        'max_steps': [200], 
        'spillage_coeff': [0.5, 0.5], 
        'orbitals': [
            {'nzeta': [1, 1], 'nzeta_from': None, 'nbands_ref': 4, 'folder': 0}, 
            {'nzeta': [2, 2, 1], 'nzeta_from': [1, 1], 'nbands_ref': 4, 'folder': 0}, 
            {'nzeta': [3, 3, 2], 'nzeta_from': [2, 2, 1], 'nbands_ref': 6, 'folder': 1}
        ]
    }
    ```
    """
    # because it is orbital that is generated one-by-one, it is reasonable to iterate
    # the orbitals...
    # NOTE: the following loop move necessary information for orbital generation from 
    # calculation_settings to siab_settings, so to make decouple between these
    # two dicts -> but this can be done earlier. The reason why do it here is because
    # the conversion from newly designed data structure, calculation_settings, and
    # siab_settings, to the old version of SIAB, is taken place here.
    for orbital in siab_settings["orbitals"]:
        # MOVE: copy lmax information from calculation_settings to siab_settings
        # for old version SIAB, but lmax can be read from orb_matrix*.dat for better design
        # in that way, not needed to import lmaxmax information from calculation_settings
        orbital["lmax"] = calculation_settings[orbital["folder"]]["lmaxmax"]
        # the key "folder" has a more reasonable alternative: "abacus_setup"

        # MOVE: let siab_settings know exactly what folders are used for orbital generation
        # in folders, folders are stored collectively according to reference structure
        # and the "abacus_setup" is then expanded to the corresponding folder
        # therefore this setup is to expand the "folder" key to the corresponding folder
        # list
        orbital["folder"] = folders[orbital["folder"]]
    
    # iteratively generate numerical atomic orbitals here
    if siab_version == "0.1.0":
        nlevel=len(siab_settings["orbitals"])
        for orb_gen, _, ilevel in siov.convert(calculation_setting=calculation_settings[0],
                                               siab_settings=siab_settings):
            """the iteration here will be processed first by rcut and second by zeta notation"""
            result = SPS_api.run(params=orb_gen, ilevel=ilevel, nlevel=nlevel)
            if result is not None:
                forb, quality = result
                # instantly print the quality of the orbital generated
                print("Report: quality of the orbital %s is:"%forb, flush=True)
                for l in range(len(quality)):
                    print("l = %d: %s"%(l, " ".join(["%10.8e"%q for q in quality[l] if q is not None])), flush=True)

    else:
        # reserve for new implementation of orbital optimization
        raise NotImplementedError("SIAB version %s is not supported yet"%siab_version)
    return sicite.citation()

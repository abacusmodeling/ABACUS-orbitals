# interface to initialize
import os
import SIAB.io.read_input as siri
import SIAB.io.pseudopotential.api as sipa
def initialize(version: str = "0.1.0",
               fname: str = "./SIAB_INPUT"):
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
    user_settings = siri.read_siab_inp(fname=fname, version=version)
    # pseudopotential check
    fpseudo = user_settings["pseudo_dir"]+"/"+user_settings["pseudo_name"]
    if not os.path.exists(fpseudo): # check the existence of pseudopotential file
        raise FileNotFoundError("Pseudopotential file %s not found"%fpseudo)
    
    pseudopotential = sipa.extract_ppinfo_forsiab(fname=fpseudo)
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
import SIAB.spillage.util as ssu
import SIAB.spillage.pytorch_swat.api as ssps_api  # old version of backend
#import SIAB.spillage.api as ss_api  # new version of backend
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
    [["Si-dimer-1.0", "Si-dimer-1.1"], ["Si-trimer-1.0", "Si-trimer-1.1", "Si-trimer-1.2"]]
    ```
    calculation_settings: list of dict, the contents of INPUT for each reference shape,
    like
    ```python
    [
        {'pseudo_dir': '/root/abacus-develop/pseudopotentials/SG15_ONCV_v1.0_upf', 
         'ecutwfc': 100, 'bessel_nao_rcut': [6, 7], 'smearing_sigma': 0.01, 
         'nbands': 8, 'lmaxmax': 2, 'nspin': 1}, 
        {'pseudo_dir': '/root/abacus-develop/pseudopotentials/SG15_ONCV_v1.0_upf', 
         'ecutwfc': 100, 'bessel_nao_rcut': [6, 7], 'smearing_sigma': 0.01, 
         'nbands': 10, 'lmaxmax': 2, 'nspin': 1}
    ]
    ```
    siab_settings: dict, the settings for SIAB optimization tasks, including informations
    all about the orbitals, like
    ```python
    {
        'nthreads_per_rcut': 1,
        'optimizer': 'pytorch.SWAT', 
        'max_steps': 200, 
        'spillage_coeff': [0.5, 0.5], 
        'orbitals': [
            {'nzeta': [1, 1], 'nzeta_from': None, 'nbands_ref': 4, 'folder': 0}, 
            {'nzeta': [2, 2, 1], 'nzeta_from': [1, 1], 'nbands_ref': 4, 'folder': 0}, 
            {'nzeta': [3, 3, 2], 'nzeta_from': [2, 2, 1], 'nbands_ref': 6, 'folder': 1}
        ]
    }
    ```
    """
    siab_settings = ssu.initialize(calculation_settings, siab_settings, folders)
    """after initialization, siab_settings will have the following structure:
    ```python
    {
        'nthreads_per_rcut': 1,
        'optimizer': 'pytorch.SWAT', 
        'max_steps': 200, 
        'spillage_coeff': [0.5, 0.5], 
        'orbitals': [
            {'nzeta': [1, 1], 'nzeta_from': None, 'nbands_ref': 4, 
             'folder': ['Si-dimer-1.0', 'Si-dimer-1.1'], 'lmax': 2}, 
            {'nzeta': [2, 2, 1], 'nzeta_from': [1, 1], 'nbands_ref': 4, 
             'folder': ['Si-dimer-1.0', 'Si-dimer-1.1'], 'lmax': 2}, 
            {'nzeta': [3, 3, 2], 'nzeta_from': [2, 2, 1], 'nbands_ref': 6, 
             'folder': ['Si-trimer-1.0', 'Si-trimer-1.1', 'Si-trimer-1.2'], 'lmax': 2}
        ]
    }
    ```
    """
    # iteratively generate numerical atomic orbitals here
    if siab_version == "0.1.0":
        ssps_api.iter(siab_settings=siab_settings, calculation_settings=calculation_settings)
    else:
        # reserve for new implementation of orbital optimization
        raise NotImplementedError("SIAB version %s is not supported yet"%siab_version)
        #ss_api.run(siab_settings=siab_settings)
    
    return

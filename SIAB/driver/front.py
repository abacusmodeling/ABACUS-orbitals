# interface to initialize
import SIAB.io.read_input as siri
def initialize(version: str = "0.1.0",
               fname: str = "./SIAB_INPUT"):
    """initialization of numerical atomic orbitals generation task, 
    1. read input file named as fname, will convert to new version inside package
    2. check the existence of pseudopotential file, if pseudopotential_check is True, go on
    3. unpack the input file, return set of parameters:
        - reference_shapes: list of reference shapes, e.g. ["dimer", "trimer"]
        - bond_lengths: list of bond lengths for each reference shape, e.g. [[1.0, 1.1], [1.0, 1.1, 1.2]]
        - calculation_settings: list of calculation settings, e.g. 
          [{"bessel_nao_rcut": 5.0, "bessel_nao_lmax": 5}, {"bessel_nao_rcut": 5.0, "bessel_nao_lmax": 5}]
          for each reference shape
        - siab_settings: dict of SIAB settings specially for SIAB optimization tasks.
        - env_settings: tuple of environment settings, contain environment variables, abacus executable, and mpi executable
        - general: dict of general settings, contain element symbol, pseudo_dir, pseudo_name
    
    Parameters
    ----------
    version: str
        version of SIAB, this is not used anymore
    fname: str
        input filename, default is "./SIAB_INPUT"
    """
    import os
    fname = fname.strip().replace("\\", "/")
    user_settings = siri.read_siab_inp(fname=fname, version=version)
    
    # pseudopotential existence check
    fpseudo = user_settings["pseudo_dir"]+"/"+user_settings["pseudo_name"]
    if not os.path.exists(fpseudo): # check the existence of pseudopotential file
        raise FileNotFoundError("Pseudopotential file %s not found"%fpseudo)
    structures, abacus, siab, env, general = siri.unpack_siab_input(user_settings)
    lmaxmax = max([dftparam.get("lmaxmax", 1) for dftparam in abacus])

    ##################################################
    # NEW FEATURE in SIAB-v3.0: support for jy basis #
    ##################################################
    from SIAB.spillage.api import _coef_gen, _save_orb
    use_jy = user_settings.get("basis_type", "jy") == "jy" \
        and user_settings.get("optimizer", "pytorch.SWAT") != "none" 
    # if user only want jy, will generate elsewhere
    ecut = user_settings.get("ecutwfc", 100)
    rcuts = user_settings.get("bessel_nao_rcut", [6.0])
    if use_jy: # only if use_jy, will generate jy basis
        fjy = [_save_orb(
            _coef_gen(rcut, ecut, lmaxmax)[0], general["element"], ecut, rcut, "jy", 
            user_settings.get("jy_type", "reduced")) for rcut in rcuts]
        abacus = [dftparam|{"orbital_dir": fjy, 
                            "basis_type": "lcao",
                            "ks_solver": "genelpa"} for dftparam in abacus]
    # the code above will change the abacus setting: INPUT:
    # basis_type: pw -> lcao
    # ks_solver: dav -> genelpa
    # and add a new key: orbital_dir, which is the directory of jy basis
    return structures, abacus, siab, env, general

# interface to abacus
import SIAB.interface.abacus as sia
def abacus(general: dict,
           structures: list,
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

    Parameters
    ----------
    general: dict
        general settings, contain element symbol, pseudo_dir, pseudo_name

    structures: list of dict
        list of structures, each structure is a dict, contain the information of the structure
    
    calculation_settings: list of dict
        list of calculation settings, for each reference shape
    
    env_settings: dict
        settings for calculation environment configuration, important in HPC case
    
    test: bool
        whether to run in test mode, default is True
    """
    return sia.run_all(general=general,
                       structures=structures,
                       calculation_settings=calculation_settings,
                       env_settings=env_settings,
                       test=test)

# interface to Spillage optimization

import SIAB.spillage.pytorch_swat.api as ssps_api  # old version of backend
import SIAB.spillage.api as ss_api  # new version of backend
def spillage(folders: list,
             calculation_settings: list,
             siab_settings: dict):
    """spillage interface

    Parameters
    ----------
    folders: list of list of str
        the folders for each reference shape and bond length,
    like
    ```python
    [["Si-dimer-1.0", "Si-dimer-1.1"], 
     ["Si-trimer-1.0", "Si-trimer-1.1", "Si-trimer-1.2"]]
    ```
    calculation_settings: list of dict
        the contents of INPUT for each reference shape, like
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
    siab_settings: dict
        the settings for SIAB optimization tasks, including informations all about the orbitals, 
        like
    ```python
    {
        'nthreads_per_rcut': 1,
        'optimizer': 'pytorch.SWAT', 
        'max_steps': 200, 
        'spill_coefs': [2.0, 1.0], 
        'orbitals': [
            {'nzeta': [1, 1], 'nzeta_from': None, 'nbands_ref': 4, 'folder': 0}, 
            {'nzeta': [2, 2, 1], 'nzeta_from': [1, 1], 'nbands_ref': 4, 'folder': 0}, 
            {'nzeta': [3, 3, 2], 'nzeta_from': [2, 2, 1], 'nbands_ref': 6, 'folder': 1}
        ]
    }
    ```
    """
    # iteratively generate numerical atomic orbitals here
    optimizer = siab_settings.get("optimizer", "none").lower()
    caller_map = {
        "pytorch.swat": ssps_api.iter,
        "none": ss_api.iter,
        "restart": ss_api.iter,
        "bfgs": ss_api.iter
    }
    caller_map[optimizer](siab_settings, calculation_settings, folders)


"""this file defines interface between newly implemented spillage optimization algorithm
with the driver of SIAB"""
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from SIAB.spillage.orbio import read_param, write_nao, write_param
from SIAB.spillage.spillage import Spillage_jy, Spillage_pw, flatten,\
    initgen_jy, initgen_pw
from SIAB.spillage.listmanip import merge
from SIAB.spillage.plot import plot_chi
from SIAB.spillage.radial import coeff_normalized2raw, coeff_reduced2raw,\
    build_raw, build_reduced, _nbes
from SIAB.spillage.datparse import read_wfc_lcao_txt, read_triu, \
    read_running_scf_log, read_input_script, read_orb_mat,\
    read_istate_info
from SIAB.spillage.lcao_wfc_analysis import _wll
from SIAB.spillage.lcao_wfc_analysis import api as wfc_analysis_api
import unittest
from SIAB.spillage.jy_expmt import _coef_init as _coef_init_jy
from SIAB.spillage.jy_expmt import _ibands
from SIAB.spillage.util import neo_spilopt_params_from_dft

def _coef_gen(rcut: float, ecut: float, lmax: int, value: str = "eye"):
    """Directly generate the coefficients of the orbitals instead of performing optimization
    
    Parameters
    ----------
    rcut: float
        the cutoff radius
    ecut: float
        the energy cutoff
    lmax: int
        the maximum angular momentum
    value: str
        the value to be returned, can be "eye"
    
    Returns
    -------
    list: the coefficients of the orbitals, indexed by [type][l][zeta][q]
    """

    if not isinstance(ecut, (int, float)):
        raise ValueError("Expected ecut to be an integer or float")
    if not isinstance(lmax, int):
        raise ValueError("Expected lmax to be an integer")
    if not isinstance(value, str):
        raise ValueError("Expected value to be a string")
    
    nbes_rcut = [_nbes(l, rcut, ecut) for l in range(lmax + 1)]
    if value == "eye":
        return [[np.eye(nbes_rcut[l]).tolist() for l in range(lmax + 1)]]
    else:
        raise ValueError("Only 'eye' is supported for value presently")

def _coef_restart(fcoef: str):
    """get coefficients from SIAB dumped jy coefficients file
    
    Parameters
    ----------
    fcoef: str
        the file name of the coefficients
    
    Returns
    -------
    list: the coefficients of the orbitals
    """
    return read_param(fcoef)

def _coef_opt_jy(rcut, orbparams, folders, options, nthreads, spill_coefs = None):
    """for fit_basis jy case, optimize Spillage function to get contraction coefficients of jy for one single rcut value
    
    Parameters
    ----------
    rcut: float
        the cutoff radius
    orbparams: list
        list of settings for generation of each orbital
    folders: list[list[str]]
        the folders where the ABACUS run information are stored. The first level of list
        is the geometry like dimer, trimer, etc, the second level is the folders of the
        deformation of the geometry, such as stretching, bending, etc.
    options: dict
        the options for optimization
    nthreads: int
        the number of threads used in optimization
    
    Returns
    -------
    list[list[list[float]]]: the coefficients of the orbitals
    """
    minimizer = Spillage_jy()
    print(f"ORBGEN: Optimizing orbitals for rcut = {rcut} au", flush = True)
    if spill_coefs:
        print("ORBGEN: For jy, spill_coefs is deprecated.", flush = True)

    configs = [folders[indf] for orb in orbparams for indf in orb['folder']]
    configs = list(set([folder for f in configs for folder in f]))

    iconfs = [[] for _ in range(len(orbparams))]
    for iorb, orb in enumerate(orbparams):
        iconfs[iorb] = [configs.index(folder) for f in orb['folder'] for folder in folders[f]]
    
    for folder in configs:
        minimizer.config_add(os.path.join(folder, f"OUT.{os.path.basename(folder)}"))
    
    nbands_ref = [[orb['nbands_ref']] if not isinstance(orb['nbands_ref'], list) else orb['nbands_ref']\
                   for orb in orbparams]
    nbands_ref = [[nbands_ref[iorb]*len(folders[i]) for i in orb['folder']]
                    for iorb, orb in enumerate(orbparams)]
    
    # nzeta infer...
    nzeta = [_nzeta_mean_conf(flatten(nbands_ref[iorb]), 
                              flatten([folders[i] for i in orb['folder']]),
                              'max',
                              'svd-fold',
                              [orb['nzeta']]) # currently only one atomtype
             for iorb, orb in enumerate(orbparams)]
    # use int(ceil()) to filter nzeta values
    nzeta = [[int(np.ceil(nz)) for nz in nzeta_orb] for nzeta_orb in nzeta]

    # prepare opt params
    elem = flatten(folders)[0].split("-")[0]
    initdir = "-".join([elem, "monomer", f"{rcut}au"])
    initdir = os.path.join(initdir, f"OUT.{os.path.basename(initdir)}")
    guess = _plan_guess(nzeta, initdir, True, True)

    ibands = [_ibands(orb['nbands_ref'], orb['folder'], [len(f) for f in folders if f]) 
              for orb in orbparams]
    deps = [orb['nzeta_from'] for orb in orbparams]
    
    return _do_onion_opt(minimizer, 
                         nzeta, 
                         iconfs, 
                         ibands, 
                         deps, 
                         nthreads, 
                         options, 
                         guess)

def _coef_opt_pw(rcut, orbparams, folders, options, nthreads, spill_coefs = None):
    """for fit_basis pw case, optimize Spillage function to get contraction coefficients of pw for one single rcut value
    
    Parameters
    ----------
    rcut: float
        the cutoff radius
    orbparams: list
        list of settings for generation of each orbital
    folders: list[list[str]]
        the folders where the ABACUS run information are stored. The first level of list
        is the geometry like dimer, trimer, etc, the second level is the folders of the
        deformation of the geometry, such as stretching, bending, etc.
    options: dict
        the options for optimization
    nthreads: int
        the number of threads used in optimization
    
    Returns
    -------
    list[list[list[float]]]: the coefficients of the orbitals
    """

    minimizer = Spillage_pw()
    print(f"ORBGEN: Optimizing orbitals for rcut = {rcut} au", flush = True)

    spill_coefs = [0.0, 1.0] if spill_coefs is None else spill_coefs

    configs = [folders[indf] for orb in orbparams for indf in orb['folder']]
    configs = list(set([folder for f in configs for folder in f]))

    iconfs = [[] for _ in range(len(orbparams))]
    for iorb, orb in enumerate(orbparams):
        iconfs[iorb] = [configs.index(folder) for f in orb['folder'] for folder in folders[f]]
    
    fov = None
    for folder in configs:
        for fov_, fop_ in _orb_matrices(folder):
            ov, op = map(read_orb_mat, [fov_, fop_])
            assert ov['rcut'] == op['rcut'], "Data violation: rcut of ov and op matrices are different"
            if np.abs(ov['rcut'] - rcut) < 1e-10:
                print(f"ORBGEN: jy_jy, mo_jy and mo_mo matrices loaded from {fov_} and {fop_}", flush = True)
                minimizer.config_add(fov_, fop_, spill_coefs)
                fov = fov_ if fov is None else fov
        
    nzeta = [orb['nzeta'] for orb in orbparams]

    # prepare opt params
    elem = flatten(folders)[0].split("-")[0]
    initdir = "-".join([elem, "monomer"])
    initdir = os.path.join(initdir, os.path.basename(fov))
    guess = _plan_guess(nzeta, initdir, False, True)
    ibands = [_ibands(orb['nbands_ref'], orb['folder'], [len(f) for f in folders if f]) for orb in orbparams]
    ideps = [orb['nzeta_from'] for orb in orbparams]

    return _do_onion_opt(minimizer,
                         nzeta,
                         iconfs,
                         ibands,
                         ideps,
                         nthreads,
                         options,
                         guess)

def _plan_guess(nzeta, folder, jy = True, diagnosis = True):
    """initialize the coef_guess for both jy and pw basis. calculate the maximal
    number of zeta func needed by each angular momentum.
    
    Parameters
    ----------
    nzeta: list[list[int]]
        the number of zeta functions for each l
    folder: str
        the folder where the ABACUS run of initial guess is stored
    jy: bool
        whether the jY basis is used
    diagnosis: bool
        whether the diagnosis information is printed
    
    Returns
    -------
    dict: the initial guess configuration
    """
    # calculate the first param of function initgen
    lmax = max([len(nz_orb) for nz_orb in nzeta]) - 1

    # calculate maxial number of zeta for each l
    nzeta_max = [(lambda nzeta: nzeta + (lmax + 1 - len(nzeta))*[-1])(nz_orb) for nz_orb in nzeta]
    nzeta_max = [max([orb[i] for orb in nzeta_max]) for i in range(lmax + 1)]
    keys = ["nzeta", "diagnosis"]
    keys += ["outdir"] if jy else ["orb_mat"]
    return dict(zip(keys, [nzeta_max, diagnosis, folder]))

def _do_onion_opt(minimizer,
                  nzeta, 
                  iconfs, 
                  ibands, 
                  deps, 
                  nthreads, 
                  options, 
                  guess):
    """Onion! optimize the contraction coefficients of jy from inner to outer step by step.
    Based on the contraction coefficients of jy finding problem to the Spillage function
    minimization problem, the optimization is performed in a hierarchical way: from
    core, shell to outer shell. Once the optimization of one shell is done, during the
    optimization of the next shell, the coefficients of the previous shell will be fixed.

    Parameters
    ----------
    minimizer: Spillage
        the Spillage object, can be Spillage_jy or Spillage_pw
    nzeta: list[list[int]]
        the number of zeta functions for each l for each shell of orbitals
    iconfs: list[list[int]]
        the indexes of configurations for each shell of orbitals to ref
    ibands: list[list[int]]
        the band indexes for each shell of orbitals
    deps: list[int]
        the shells are not guaranteed to be arranged like from inner to outer or vice 
        versa, this list stores the mapping of each shell to its previous shell
    nthreads: int
        the number of threads used in optimization
    options: dict
        the options for optimization
    guess: dict
        the initial guess configuration, see function _plan_guess for details

    Returns
    -------
    list[list[list[float]]]: the coefficients of the orbitals
    """
    
    norb = len(nzeta)
    coefs = [None for _ in range(norb)]

    # optimize the orbital layer-by-layer, from inner to outer
    for iorb, index_, nzeta_, iconfs_, ibands_ in zip(range(norb), deps, nzeta, iconfs, ibands):
        nzeta_inner = None if index_ is None else nzeta[index_]
        print(f"""ORBGEN: optimization on level {iorb + 1} (with # of zeta functions for each l: {nzeta_}), 
        based on orbital ({nzeta_inner})""", flush = True)
        
        # initial guess of coefficients of present layer
        coef_init = _coef_subset(nzeta, nzeta_inner, initgen_pw(**guess)) if 'orb_mat' in guess \
            else [_coef_init_jy(guess['outdir'], nzeta_, nzeta_inner, True)]
        
        # and select what coefficients to be frozen during opt
        coef_inner = coefs[deps[iorb]] if deps[iorb] is not None else None
        
        # opt
        coefs_shell = minimizer.opt(coef_init, 
                                    coef_inner, 
                                    iconfs_, 
                                    ibands_, 
                                    options, 
                                    nthreads)
        
        # merge the coefficients of present layer with the previous layer
        coefs[iorb] = merge(coef_inner, coefs_shell, 2) if coef_inner is not None \
            else coefs_shell
        print(f"ORBGEN: End optimization on level {iorb + 1} orbital, merge with previous orbital shell(s).", flush = True)
    return coefs

def _coef_opt(rcut, orbparams, folders, maxiter, nthreads, jy, spill_coefs = None):
    """optimize Spillage function to get contraction coefficients of jy for one single rcut value
    
    Parameters
    ----------
    rcut: float
        the cutoff radius
    orbparams: list
        list of settings for generation of each orbital
    folders: list[list[str]]
        the folders where the ABACUS run information are stored, indexed by [geom][deform]
    maxiter: int
        the maximum number of iterations
    nthreads: int
        the number of threads used in optimization
    jy: bool
        whether the jY basis is used
    spill_coefs: list[float]
        the spillage coefficients, not used when set fit_basis as jy
    
    Returns
    -------
    list[list[list[float]]]: the coefficients of the orbitals
    """
    call = _coef_opt_jy if jy else _coef_opt_pw
    option = {"maxiter": maxiter,
              "disp": False, "ftol": 0, "gtol": 1e-6, 'maxcor': 20}
        
    return call(rcut, 
                orbparams, 
                folders, 
                option, 
                nthreads, 
                spill_coefs)

def _coef_subset(nzeta, nzeta0, data):
    """
    Compare `nzeta` and `nzeta0`, get the subset of `data` that `nzeta` has but
    `nzeta0` does not have. Returned nested list has dimension:
    [t][l][z][q]
    t: atom type
    l: angular momentum
    z: zeta
    q: q of j(qr)Y(q)

    Parameters
    ----------
    nzeta: list[int]
        the number of zeta functions for each l
    nzeta0: list[int]
        the number of zeta functions for each l
    data: list[list[list[float]]]
        the coefficients of the orbitals.
    
    Returns
    -------
    list[list[list[float]]]: the subset
    """
    if nzeta0 is None:
        return [[[ data[l][iz] for iz in range(nz) ] for l, nz in enumerate(nzeta)]]
    assert len(nzeta) >= len(nzeta0), f"(at least) size error of nzeta and nzeta0: {len(nzeta)} and {len(nzeta0)}"
    nzeta0 = nzeta0 + (len(nzeta) - len(nzeta0))*[0]
    for nz, nz0 in zip(nzeta, nzeta0):
        assert nz >= nz0, f"not hirarcal structure of these two nzeta set: {nzeta0} and {nzeta}"
    iz_subset = [list(range(iz, jz)) for iz, jz in zip(nzeta0, nzeta)]
    # then get coefs from data with iz_subset, the first dim of iz_subset is l
    #               l  zeta                 l  list of zeta
    return [[[ data[l][j] for j in jz ] for l, jz in enumerate(iz_subset)]]

def _orb_matrices(folder: str):
    """
    Parameters
    ----------
    folder: str
        the folder where the orb_matrix files are stored
    
    Returns
    -------
    tuple of str: the file names of orb_matrix and its derivative (absolute path)

    Notes
    -----
    on the refactor of ABACUS Numerical_Basis class
    
    This function provides a temporary solution for getting correct file name
    of orb_matrix from the folder path. There are discrepancies between the
    resulting orb_matrix files yielded with single bessel_nao_rcut parameter
    and multiple. The former will yield orb_matrix.0.dat and orb_matrix.1.dat,
    while the latter will yield orb_matrix_rcutRderivD.dat, in which R and D
    are the corresponding bessel_nao_rcut and order of derivatives of the
    wavefunctions, presently ranges from 6 to 10 and 0 to 1, respectively.
    """

    old = r'orb_matrix.([01]).dat'
    new = r'orb_matrix_rcut(\d+)deriv([01]).dat'

    files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    # convert to absolute path
    old_files = [os.path.join(folder, f) for f in files if re.match(old, f)]
    new_files = [os.path.join(folder, f) for f in files if re.match(new, f)]
    # not allowed to have both old and new files
    assert not (old_files and new_files)
    assert len(old_files) == 2 or not old_files
    assert len(new_files) % 2 == 0 or not new_files

    # make old_files to be list of tuples, if not None
    old_files = [(old_files[0], old_files[1])] if old_files else None
    # new files are sorted by rcut and deriv
    new_files = sorted(new_files) if new_files else None
    # similarly, make new_files to be list of tuples, if not None
    if new_files:
        new_files = [(new_files[i], new_files[i+1]) for i in range(0, len(new_files), 2)]
    
    # yield
    files = old_files or new_files
    for f in files:
        yield f

def _save_orb(coefs, elem, ecut, rcut, folder, jY_type: str = "reduced"):
    """
    Plot the orbital and save .orb file
    Parameter
    ---------
    coefs_tot: list of list of list of float
        the coefficients of the orbitals [l][zeta][q]: c (float)
    elem: str
        the element symbol
    ecut: float
        the energy cutoff
    rcut: float
        the cutoff radius
    folder: str
        the folder to save the orbitals
    jY_type: str
        the type of jY basis, can be "reduced", "nullspace", "svd" or "raw"
    
    Return
    ------
    str: the file name of the orbital    
    """

    coeff_converter_map = {"reduced": coeff_reduced2raw, 
                           "normalized": coeff_normalized2raw}
    syms = "SPDFGHIKLMNOQRTUVWXYZ".lower()
    dr = 0.01
    r = np.linspace(0, rcut, int(rcut/dr)+1)

    folder = os.path.abspath(folder)
    os.makedirs(folder, exist_ok=True)

    chi = _coef_griddata([coefs], rcut, 0.01, jY_type)
    # however, we should not bundle orbitals of different atomtypes together

    suffix = "".join([f"{len(coef)}{sym}" for coef, sym in zip(coefs, syms)])

    fpng = os.path.join(folder, f"{elem}_gga_{rcut}au_{ecut}Ry_{suffix}.png")
    plot_chi(chi, r, save=fpng)
    plt.close()

    forb = fpng[:-4] + ".orb"
    write_nao(forb, elem, ecut, rcut, len(r), dr, chi)

    # fparam = os.path.join(folder, "ORBITAL_RESULTS.txt")
    fparam = forb[:-4] + ".param"
    write_param(fparam, coeff_converter_map[jY_type](coefs, rcut), rcut, 0.0, elem)
    print(f"orbital saved as {forb}")

    return forb

def _coef_griddata(coefs, rcut, dr: float = 0.01, jY_type: str = "reduced"):
    """build real space grid orbital based on the coefficients of the orbitals,
    rcut and grid spacing dr. The coefficients should be in the form of
    [it][l][zeta][q].
    
    Args:
    coefs: list of list of list of list of float
        the coefficients of the orbitals
    rcut: float
        the cutoff radius
    dr: float
        the grid spacing
    jY_type: str
        the type of jY basis, can be "reduced", "nullspace", "svd" or "raw"

    Returns:
    np.ndarray: the real space grid data
    """

    r = np.linspace(0, rcut, int(rcut/dr)+1) # hard code dr to be 0.01? no...
    if jY_type in ["reduced", "nullspace", "svd"]:
        chi = build_reduced(coefs[0], rcut, r, True)
    else:
        coefs = coeff_normalized2raw(coefs, rcut)
        chi = build_raw(coefs[0], rcut, r, 0.0, True, True)
    return chi

def run(siab_settings: dict, calculation_settings: list, folders: list):
    """Loop over rcut values and yield orbitals
    
    Parameters
    ----------
    siab_settings: dict
        the settings for SIAB optimization
    calculation_settings: list
        the settings for ABACUS calculation
    folders: list of list of str
        the folders where the ABACUS run information are stored. folders are indexed
        by [geom][deform], in which the `geom` is index of geometry like dimer, trimer,
        etc, and `deform` is index of deformation of the geometry, such as stretching,
        etc.
    """
    rcuts, ecut, elem, jy_type, run_type, spil_option = \
        neo_spilopt_params_from_dft(calculation_settings, siab_settings, folders)
    for rcut in rcuts: # can be parallelized here
        ##############
        # Generation #
        ##############
        # for jy basis calculation, only matched rcut folders are needed, so there is
        # a filter needed.
        if run_type == "opt":
            # REFACTOR: SIAB-v3.0, get folders with matched rcut
            confs = [[f for f in fgrp \
                      if len(f.split("-")) == 3\
                      or float(f.split("-")[-1].replace("au", "")) == rcut] # jy case 
                      for fgrp in folders]
            coefs_tot = _coef_opt(**spil_option, rcut = rcut, folders=confs)
        elif run_type == "restart":
            raise NotImplementedError("restart is not implemented yet")
        else: # run_type == "none", used to generate jY basis
            coefs_tot = [_coef_gen(rcut, ecut, len(orb['nzeta']) - 1) for orb in siab_settings['orbitals']]

        #################
        # save orbitals #
        #################
        for ilev, coefs in enumerate(coefs_tot): # loop over different levels...
            folder = "_".join([elem, f"{rcut}au", f"{ecut}Ry"]) # because the concept of "level" is not clear
            for coefs_it in coefs: # loop over different atom types
                _ = _save_orb(coefs_it, elem, ecut, rcut, folder, jy_type)
    return

def _nzeta_mean_conf(nbands, 
                     folders, 
                     statistics = 'max',
                     pop = 'svd-fold',
                     filter = 1.0):
    """infer the nzeta from given folders with some strategy. If there are
    multiple kpoints calculated, for each folder the result will be firstly
    averaged and used to represent the `nzeta` inferred from the whole folder
    , then the `nzeta` will be averaged again over all folders to get the
    final result
    
    Parameters
    ----------
    nbands: int|list[int]
        the number of bands for each folder. If it is a list, the range of
        bands will be set individually for each folder.
    folders: list[str]
        the folders where the ABACUS run information are stored.
    statistics: str
        the statistics method used to infer nzeta, can be 'max' or 'mean'
    pop: str
        the population analysis method used to infer nzeta, can be 'svd-fold'
        or 'svd-max'
    filter: float|list[float]|list[list[int]]|None
        can be either float or list of floats, which means the threshold for 
        the significance for each atomtype. 
        If it is a list[list[int]], it should be the nzeta for each atomtype, 
        then the loss will be evaluated based on it. If it is None, will be 
        overwritten to the list[float] case in which all elements are 1.0

    Returns
    -------
    nzeta: list[int]
        the inferred nzeta for each folder
    """
    assert statistics in ['max', 'mean']
    assert isinstance(folders, list), f"folders should be a list: {folders}"
    assert all([isinstance(f, str) for f in folders]), f"folders should be a list of strings: {folders}"

    # nzeta = np.array([0])
    nzeta = []
    nbands = [nbands] * len(folders) if not isinstance(nbands, list) else nbands

    max_shape = (0,) # 1D array
    for folder, nband in zip(folders, nbands):
        nzeta_ = np.array(_nzeta_infer(folder, nband, pop, filter=filter))
        max_shape = np.maximum(max_shape, nzeta_.shape)
        nzeta.append(nzeta_)

    nzeta = np.array([nz.reshape(max_shape).tolist() for nz in nzeta])
    return np.max(nzeta, axis=0).tolist() if statistics == 'max' \
        else np.mean(nzeta, axis=0).tolist()

def _nzeta_infer(folder, nband, pop = 'svd-fold', filter = 1.0):
    """infer nzeta based on one structure whose calculation result is stored
    in the folder
    
    Parameters
    ----------
    folder: str
        the folder where the ABACUS run information are stored
    nband: int|list[int]|range
        if specified as int, it is the highest band index to be considered. 
        if specified as list or range, it is the list of band indexes to be
        considered
    pop: str, optional
        the population analysis method used to infer nzeta, can be 'svd-fold'
        or 'svd-max'. The default is 'svd-fold'.
    filter: float|list[float]|list[list[int]]|None
        can be either float or list of floats, which means the threshold for 
        the significance for each atomtype. 
        If it is a list[list[int]], it should be the nzeta for each atomtype, 
        then the loss will be evaluated based on it. If it is None, will be 
        overwritten to the list[float] case in which all elements are 1.0

    Returns
    -------
    np.ndarray: the inferred nzeta for the folder, like list[int]
    """

    # read INPUT and running_*.log
    params = read_input_script(os.path.join(folder, "INPUT"))
    outdir = os.path.abspath(os.path.join(folder, "OUT." + params.get("suffix", "ABACUS")))
    nspin = int(params.get("nspin", 1))
    fwfc = "WFC_NAO_GAMMA" if params.get("gamma_only", False) else "WFC_NAO_K"
    running = read_running_scf_log(os.path.join(outdir, 
                                                f"running_{params.get('calculation', 'scf')}.log"))
    kpts, ener, occ = read_istate_info(os.path.join(outdir, "istate.info"))

    assert nspin == running["nspin"], \
        f"nspin in INPUT and running_scf.log are different: {nspin} and {running['nspin']}"
    
    # if nspin == 2, the "spin-up" kpoints will be listed first, then "spin-down"
    wk = running["wk"]

    print(f'\nPerform nzeta_infer/evaluation task for {folder}', flush=True)
    nzeta = np.array([0])
    for isk in range(nspin*len(wk)): # loop over (ispin, ik)
        ik, is_ = isk % len(wk), isk % nspin
        w = wk[ik] # spin-up and spin-down share the wk
        wfc, _, _, kpt = read_wfc_lcao_txt(os.path.join(outdir, f"{fwfc}{isk+1}.txt"))

        ik_ = np.where(np.all(kpts == kpt, axis=1))[0][0]
        nocc = len(np.where(np.array(occ[is_][ik_]) > 0)[0])
        nall = len(occ[is_][ik_])
        assert isinstance(nband, (int, str))
        nband = nband if isinstance(nband, int) else eval('n' + nband)
        assert wfc.shape[1] >= nband, \
            f"ERROR: number of bands for orbgen is larger than calculated: {nband} > {wfc.shape[1]}"

        # the complete return list is (wfc.T, e, occ, k)
        ovlp = read_triu(os.path.join(outdir, f"data-{isk}-S"))
        # the number of non-zeros of each l is the maximal number of zeta functions
        sigma, nz, loss = wfc_analysis_api(wfc, 
                                           ovlp,
                                           running['natom'], 
                                           running['nzeta'], 
                                           pop, 
                                           nband=nband, 
                                           filter=filter)
        # print out the result
        print(f'k = {ik}, ispin = {is_}', flush=True)
        for it, (st_, nzt_) in enumerate(zip(sigma, nz)):
            print(f'For type {it}, the complete list of sigma values shown as:', flush=True)
            for l, s in enumerate(st_):
                print(f'l = {l}:', flush=True)
                for i, s_ in enumerate(s):
                    print(f'{s_:>8.4f}', end = ' ', flush=True)
                    if i % 5 == 4:
                        print('', flush=True)
                print('', flush=True)
            print(f'Number of zeta functions for type {it} is {nzt_}', flush=True)
        print(f'jy space truncation (subspace) loss: {loss:8.4e}\n', flush=True)

        nz = np.array(nz[0])
        nzeta = np.resize(nzeta, np.maximum(nzeta.shape, nz.shape)) + nz * w / nspin

    # count the number of atoms
    assert len(running["natom"]) == 1, f"multiple atom types are not supported: {running['natom']}"

    return nzeta

def _wll_fold(wll, nband):
    """One of strategy for inferring nzeta from wll matrix. This function
    fold the wll to the number of bands, and return the folded wll in shape 
    of 1*(lmax+1)
    
    Parameters
    ----------
    wll: np.ndarray
        the wll matrix, with shape of nband x lmax+1 x lmax+1
    nband: int|list[int]|range
        if specified as int, it is the highest band index to be considered. 
        if specified as list or range, it is the list of band indexes to be
        considered
    
    Returns
    -------
    np.ndarray: the folded wll matrix in shape of 1*(lmax+1)
    """

    nband = range(nband) if isinstance(nband, int) else nband
    _, lmax_plus_1, _ = wll.shape
    wll_fold = np.array([0]*lmax_plus_1, dtype=float)
    degen = np.array([2*i + 1 for i in range(lmax_plus_1)], dtype=float)
    for ib in nband:
        wlb = np.sum(wll[ib].real, 1)
        wll_fold += wlb / degen

    return wll_fold

def _nzeta_analysis(folder, count_thr = 1e-1, itype = 0):
    """analyze the initial guess, distinguishing n and l (principal and angular quantum number)
    
    Parameters
    ----------
    folder: str
        the folder where the ABACUS run information are stored
    
    Returns
    -------
    list[list[list[int]]]: [l][n][ib] storing the band index(es)
    """


    params = read_input_script(os.path.join(folder, "INPUT"))
    outdir = os.path.abspath(os.path.join(folder, "OUT." + params.get("suffix", "ABACUS")))
    nspin = int(params.get("nspin", 1))
    fwfc = "WFC_NAO_GAMMA" if params.get("gamma_only", False) else "WFC_NAO_K"
    running = read_running_scf_log(os.path.join(outdir,
                                                f"running_{params.get('calculation', 'scf')}.log"))
    
    assert nspin == running["nspin"], \
        f"nspin in INPUT and running_scf.log are different: {nspin} and {running['nspin']}"
    
    lmaxmax = len(running["nzeta"][itype]) - 1
    assert lmaxmax >= 0, f"lmaxmax should be at least 0: {lmaxmax}"
    out = [[[] for _ in range(lmaxmax + 1)] for _ in range(nspin)]

    # if nspin == 2, the "spin-up" kpoints will be listed first, then "spin-down"
    wk = running["wk"]
    for isk in range(nspin*len(wk)): 
        # loop over (ispin, ik), but usually initial guess is a gamma point calculation
        # so the following w is not used
        # w = wk[isk % len(wk)]

        wfc, _, _, _ = read_wfc_lcao_txt(os.path.join(outdir, f"{fwfc}{isk+1}.txt"))
        # the complete return list is (wfc.T, e, occ, k)
        ovlp = read_triu(os.path.join(outdir, f"data-{isk}-S"))

        # for monomer, it is okay to use wll method to decompose its components
        wll = _wll(wfc, ovlp, running["natom"], running["nzeta"])

        for ib, wb in enumerate(wll): # loop over bands
            wlb = np.sum(wb.real, 1) # sum over one dimension, get dim 1 x lmax matrix
            for l, wl in enumerate(wlb):
                if wl >= count_thr:
                    out[isk % len(wk)][l].append(ib)
    return out

class TestAPI(unittest.TestCase):

    # @unittest.skip('Skip for developement')
    def test_coef_gen(self):

        rcut = 3.0
        ecut = 10.0
        lmax = 3
        coefs = _coef_gen(rcut, ecut, lmax)
        self.assertEqual(len(coefs), 1) # always only one atomtype
        self.assertEqual(len(coefs[0]), lmax + 1) # lmax + 1 orbitals
        for l in range(lmax + 1):
            dim1 = len(coefs[0][l])
            dim2 = len(coefs[0][l][0])
            self.assertEqual(dim1, dim2)
            self.assertEqual(coefs[0][l], np.eye(dim1).tolist())
    
    # @unittest.skip('Skip for developement')
    def test_plan_guess(self):

        nzeta = [[3, 3, 2], [2, 2, 1], [1, 1]]
        initdir = "initdir"
        guess = _plan_guess(nzeta, initdir, True, True)
        self.assertEqual(guess["nzeta"], [3, 3, 2])
        self.assertEqual(guess["diagnosis"], True)
        self.assertEqual(guess["outdir"], "initdir")

        nzeta = [[3, 1, 2], [2, 2, 1], [4, 1]]
        guess = _plan_guess(nzeta, initdir, False, True)
        self.assertEqual(guess["nzeta"], [4, 2, 2])
        self.assertEqual(guess["diagnosis"], True)
        self.assertEqual(guess["orb_mat"], "initdir")

    # @unittest.skip('Skip for developement')
    def test_orb_matrices(self):

        test_folder = "test_orb_matrices"
        os.makedirs(test_folder, exist_ok=True)

        # test old version
        with open(os.path.join(test_folder, "orb_matrix.0.dat"), "w") as f:
            f.write("old version")
        # expect an assertion error
        with self.assertRaises(AssertionError):
            for files in _orb_matrices(test_folder):
                pass

        # continue to write orb_matrix.1.dat
        with open(os.path.join(test_folder, "orb_matrix.1.dat"), "w") as f:
            f.write("old version")
        # will not raise an error, but return orb_matrix.0.dat and orb_matrix.1.dat
        for files in _orb_matrices(test_folder):
            self.assertSetEqual(set(files), 
            {os.path.join(test_folder, "orb_matrix.0.dat"), 
             os.path.join(test_folder, "orb_matrix.1.dat")})
            
        # write new version
        with open(os.path.join(test_folder, "orb_matrix_rcut6deriv0.dat"), "w") as f:
            f.write("new version")
        # expect an assertion error due to coexistence of old and new files
        with self.assertRaises(AssertionError):
            for files in _orb_matrices(test_folder):
                pass

        # continue to write new files
        with open(os.path.join(test_folder, "orb_matrix_rcut6deriv1.dat"), "w") as f:
            f.write("new version")
        # expect an assertion error due to coexistence of old and new files
        with self.assertRaises(AssertionError):
            for files in _orb_matrices(test_folder):
                pass

        # remove old files
        os.remove(os.path.join(test_folder, "orb_matrix.0.dat"))
        os.remove(os.path.join(test_folder, "orb_matrix.1.dat"))

        # now will return new files
        for files in _orb_matrices(test_folder):
            self.assertSetEqual(set(files), 
            {os.path.join(test_folder, "orb_matrix_rcut6deriv0.dat"), 
             os.path.join(test_folder, "orb_matrix_rcut6deriv1.dat")})
            
        # continue to write new files
        with open(os.path.join(test_folder, "orb_matrix_rcut7deriv0.dat"), "w") as f:
            f.write("new version")
        # expect an assertion error due to odd number of new files
        with self.assertRaises(AssertionError):
            for files in _orb_matrices(test_folder):
                pass
        # continue to write new files
        with open(os.path.join(test_folder, "orb_matrix_rcut7deriv1.dat"), "w") as f:
            f.write("new version")
        # now will return new files
        for ifmats, fmats in enumerate(_orb_matrices(test_folder)):
            if ifmats == 0:
                self.assertSetEqual(set(fmats), 
            {os.path.join(test_folder, "orb_matrix_rcut6deriv0.dat"), 
             os.path.join(test_folder, "orb_matrix_rcut6deriv1.dat")})
            elif ifmats == 1:
                self.assertSetEqual(set(fmats), 
            {os.path.join(test_folder, "orb_matrix_rcut7deriv0.dat"), 
             os.path.join(test_folder, "orb_matrix_rcut7deriv1.dat")})
            else:
                self.fail("too many files")
        # remove new files
        os.remove(os.path.join(test_folder, "orb_matrix_rcut6deriv0.dat"))
        os.remove(os.path.join(test_folder, "orb_matrix_rcut6deriv1.dat"))
        os.remove(os.path.join(test_folder, "orb_matrix_rcut7deriv0.dat"))
        os.remove(os.path.join(test_folder, "orb_matrix_rcut7deriv1.dat"))
        # remove the folder
        os.rmdir(test_folder)

    # @unittest.skip('Skip for developement')
    def test_nzeta_to_initgen(self):

        nz1 = np.random.randint(0, 5, 2).tolist()
        nz2 = np.random.randint(0, 5, 3).tolist()
        nz3 = np.random.randint(0, 5, 4).tolist()
        nz4 = np.random.randint(0, 5, 5).tolist()
        lmax = max([len(nz) for nz in [nz1, nz2, nz3, nz4]]) - 1
        total_init = [(lambda nzeta: nzeta + (lmax + 1 - len(nzeta))*[-1])(nz) for nz in [nz1, nz2, nz3, nz4]]
        total_init = [max([orb[i] for orb in total_init]) for i in range(lmax + 1)]
        for iz in range(lmax + 1):
            self.assertEqual(total_init[iz], max([
                nz[iz] if iz < len(nz) else -1 for nz in [nz1, nz2, nz3, nz4]]))

    # @unittest.skip('Skip for developement')
    def test_coefs_subset(self):

        nz3 = [3, 3, 2]
        nz2 = [2, 2, 1]
        nz1 = [1, 1]
        data = [np.random.random(i).tolist() for i in nz3]
        
        subset = _coef_subset(nz1, None, data)
        self.assertEqual(subset, [[[data[0][0]], 
                                   [data[1][0]]]])

        subset = _coef_subset(nz2, None, data)
        self.assertEqual(subset, [[[data[0][0], data[0][1]], 
                                   [data[1][0], data[1][1]],
                                   [data[2][0]]
                                  ]])

        subset = _coef_subset(nz3, None, data)
        self.assertEqual(subset, [[[data[0][0], data[0][1], data[0][2]], 
                                   [data[1][0], data[1][1], data[1][2]],
                                   [data[2][0], data[2][1]]
                                  ]])

        subset = _coef_subset(nz3, nz2, data)
        self.assertEqual(subset, [[[data[0][2]], 
                                   [data[1][2]], 
                                   [data[2][1]]]])
        
        subset = _coef_subset(nz2, nz1, data)
        self.assertEqual(subset, [[[data[0][1]], 
                                   [data[1][1]], 
                                   [data[2][0]]]])
        
        subset = _coef_subset(nz3, nz1, data)
        self.assertEqual(subset, [[[data[0][1], data[0][2]], 
                                   [data[1][1], data[1][2]], 
                                   [data[2][0], data[2][1]]]])
        
        nz2 = [1, 2, 1]
        subset = _coef_subset(nz3, nz2, data)
        self.assertEqual(subset, [[[data[0][1], data[0][2]], 
                                   [data[1][2]], 
                                   [data[2][1]]]])

        subset = _coef_subset(nz3, nz3, data)
        self.assertEqual(subset, [[[], [], []]])
        
        with self.assertRaises(AssertionError):
            subset = _coef_subset(nz1, nz3, data) # nz1 < nz3

        with self.assertRaises(AssertionError):
            subset = _coef_subset(nz2, nz3, data) # nz2 < nz3

        subset = _coef_subset([2, 2, 0], [2, 1, 0], data)
        self.assertEqual(subset, [[[],
                                   [data[1][1]],
                                   []]])

    @unittest.skip('wll_fold is deprecated, therefore not needed to\
        test this function')
    def test_wll_fold(self):

        here = os.path.dirname(__file__)

        fpath = os.path.join(here, "testfiles/Si/jy-7au/monomer-gamma/")
        params = read_input_script(os.path.join(fpath, "INPUT"))
        outdir = os.path.abspath(os.path.join(fpath, "OUT." + params.get("suffix", "ABACUS")))
        fwfc = "WFC_NAO_GAMMA"
        running = read_running_scf_log(os.path.join(outdir,
                                                    f"running_{params.get('calculation', 'scf')}.log"))
        
        wfc = read_wfc_lcao_txt(os.path.join(outdir, f"{fwfc}1.txt"))[0]
        ovlp = read_triu(os.path.join(outdir, f"data-0-S"))
        wll = _wll(wfc, ovlp, running["natom"], running["nzeta"])
        wll_fold = _wll_fold(wll, 4)
        ref = [1, 1, 0]
        self.assertTrue(all([abs(wl - ref[i]) < 1e-8 for i, wl in enumerate(wll_fold)]))
        wll_fold = _wll_fold(wll, 5)
        ref = [2, 1, 0]
        self.assertTrue(all([abs(wl - ref[i]) < 1e-8 for i, wl in enumerate(wll_fold)]))
        wll_fold = _wll_fold(wll, 10)
        ref = [2, 1, 1]
        self.assertTrue(all([abs(wl - ref[i]) < 1e-8 for i, wl in enumerate(wll_fold)]))

    @unittest.skip('still under development')
    def test_nzeta_infer(self):

        here = os.path.dirname(__file__)
        # gamma case is easy, multi-k case is more difficult
        fpath = os.path.join(here, "testfiles/Si/jy-7au/monomer-gamma/")

        nzeta = _nzeta_infer(fpath, 4, 'svd-max')
        ref = [1, 1, 0]
        self.assertTrue(all([abs(nz - ref[i]) < 1e-8 for i, nz in enumerate(nzeta)]))
        nzeta = _nzeta_infer(fpath, 5, 'svd-max')
        ref = [2, 1, 0]
        self.assertTrue(all([abs(nz - ref[i]) < 1e-8 for i, nz in enumerate(nzeta)]))
        nzeta = _nzeta_infer(fpath, 10, 'svd-max')
        ref = [2, 1, 1]
        self.assertTrue(all([abs(nz - ref[i]) < 1e-8 for i, nz in enumerate(nzeta)]))

        # multi-k case
        fpath = os.path.join(here, "testfiles/Si/jy-7au/monomer-k/")
        testref = """
  1.000  0.000  0.000   
  0.000  1.000  0.000   
  0.000  1.000  0.000   
  0.000  1.000  0.000   
  1.000  0.000  0.000   
  0.000  0.000  1.000   
  0.000  0.000  1.000   
  0.000  0.000  1.000   
  0.000  0.000  1.000   
   0.000  0.000  1.000
  0.999  0.000  0.001   
  0.004  0.991  0.005   
  0.000  1.000  0.000   
  0.000  1.000  0.000   
  0.719  0.004  0.277   
 -0.000 -0.000  1.000   
  0.129  0.427  0.444   
  0.000  0.008  0.992   
  0.000  0.008  0.992   
   0.000  0.000  1.000
  0.999  0.001  0.001   
  0.007  0.989  0.004   
 -0.000  0.991  0.009   
  0.000  0.999  0.001   
  0.392  0.005  0.603   
  0.014  0.079  0.907   
 -0.000  0.357  0.643   
  0.310  0.336  0.354   
  0.000  0.012  0.988   
   0.000  0.000  1.000
  0.999  0.001 -0.000   
  0.011  0.987  0.002   
  0.000  0.990  0.010   
  0.000  0.990  0.010   
  0.049  0.116  0.836   
  0.000  0.031  0.969   
  0.000  0.031  0.969   
  0.000  0.286  0.714   
  0.000  0.286  0.714   
   0.547  0.303  0.150
"""
        refdata = [list(map(float, line.split())) for line in testref.strip().split("\n")]
        refdata = np.array(refdata).reshape(4, -1, 3) # reshape to (nks, nbands, lmax+1)
        wk = [0.0370, 0.2222, 0.4444, 0.2963]
        degen = np.array([2*i + 1 for i in range(3)], dtype=float)

    #@unittest.skip('still under development')
    def test_nzeta_infer_occ(self):
        here = os.path.dirname(__file__)

        # gamma case is easy, multi-k case is more difficult
        fpath = os.path.join(here, "testfiles/Si/jy-7au/monomer-gamma/")
        with self.assertRaises(ValueError):
            nzeta = _nzeta_infer(fpath, 4, 'wll')
        nzeta = _nzeta_infer(fpath, 4)
        print(nzeta)

    @unittest.skip('still under development')
    def test_nzeta_mean_conf(self):

        here = os.path.dirname(__file__)

        # first test the monomer case at gamma
        fpath = os.path.join(here, "testfiles/Si/jy-7au/monomer-gamma/")
        folders = [fpath]
        """Here I post the accumulated wll matrix from unittest test_wll_gamma
        at SIAB/spillage/lcao_wfc_analysis.py:61
        
        Band 1     1.000  0.000  0.000     sum =  1.000
        Band 2     0.000  1.000  0.000     sum =  1.000
        Band 3     0.000  1.000  0.000     sum =  1.000
        Band 4     0.000  1.000  0.000     sum =  1.000
        Band 5     1.000  0.000  0.000     sum =  1.000
        Band 6     0.000  0.000  1.000     sum =  1.000
        Band 7     0.000  0.000  1.000     sum =  1.000
        Band 8     0.000  0.000  1.000     sum =  1.000
        Band 9     0.000  0.000  1.000     sum =  1.000
        Band 10     0.000  0.000  1.000     sum =  1.000
        """
        # nbands = 4, should yield 1s1p as [1, 1, 0]
        nzeta = _nzeta_mean_conf(4, folders, 'mean')
        nzeta = [int(np.round(nz)) for nz in nzeta] # filter to integer
        self.assertEqual(nzeta, [1, 1, 0])
        # nbands = 5, should yield 2s1p as [2, 1, 0]
        nzeta = _nzeta_mean_conf(5, folders)
        nzeta = [int(np.round(nz)) for nz in nzeta] # filter to integer
        self.assertEqual(nzeta, [2, 1, 0])
        # nbands = 10, should yield 2s1p1d as [2, 1, 1]
        nzeta = _nzeta_mean_conf(10, folders)
        nzeta = [int(np.round(nz)) for nz in nzeta] # filter to integer
        self.assertEqual(nzeta, [2, 1, 1])

        # then test the multi-k case
        fpath = os.path.join(here, "testfiles/Si/jy-7au/monomer-k/")
        folders = [fpath]
        """Here I post the accumulated wll matrix from unittest test_wll_multi_k
        at SIAB/spillage/lcao_wfc_analysis.py:87
        
        ik = 0, wk = 0.0370
        Band 1     1.000  0.000  0.000     sum =  1.000
        Band 2     0.000  1.000  0.000     sum =  1.000
        Band 3     0.000  1.000  0.000     sum =  1.000
        Band 4     0.000  1.000  0.000     sum =  1.000
        Band 5     1.000  0.000  0.000     sum =  1.000
        Band 6     0.000  0.000  1.000     sum =  1.000
        Band 7     0.000  0.000  1.000     sum =  1.000
        Band 8     0.000  0.000  1.000     sum =  1.000
        Band 9     0.000  0.000  1.000     sum =  1.000
        Band 10     0.000  0.000  1.000     sum =  1.000
        ik = 1, wk = 0.2222
        Band 1     0.999  0.000  0.001     sum =  1.000
        Band 2     0.004  0.991  0.005     sum =  1.000
        Band 3     0.000  1.000  0.000     sum =  1.000
        Band 4     0.000  1.000  0.000     sum =  1.000
        Band 5     0.719  0.004  0.277     sum =  1.000
        Band 6    -0.000 -0.000  1.000     sum =  1.000
        Band 7     0.129  0.427  0.444     sum =  1.000
        Band 8     0.000  0.008  0.992     sum =  1.000
        Band 9     0.000  0.008  0.992     sum =  1.000
        Band 10     0.000  0.000  1.000     sum =  1.000
        ik = 2, wk = 0.4444
        Band 1     0.999  0.001  0.001     sum =  1.000
        Band 2     0.007  0.989  0.004     sum =  1.000
        Band 3    -0.000  0.991  0.009     sum =  1.000
        Band 4     0.000  0.999  0.001     sum =  1.000
        Band 5     0.392  0.005  0.603     sum =  1.000
        Band 6     0.014  0.079  0.907     sum =  1.000
        Band 7    -0.000  0.357  0.643     sum =  1.000
        Band 8     0.310  0.336  0.354     sum =  1.000
        Band 9     0.000  0.012  0.988     sum =  1.000
        Band 10     0.000  0.000  1.000     sum =  1.000
        ik = 3, wk = 0.2963
        Band 1     0.999  0.001 -0.000     sum =  1.000
        Band 2     0.011  0.987  0.002     sum =  1.000
        Band 3     0.000  0.990  0.010     sum =  1.000
        Band 4     0.000  0.990  0.010     sum =  1.000
        Band 5     0.049  0.116  0.836     sum =  1.000
        Band 6     0.000  0.031  0.969     sum =  1.000
        Band 7     0.000  0.031  0.969     sum =  1.000
        Band 8     0.000  0.286  0.714     sum =  1.000
        Band 9     0.000  0.286  0.714     sum =  1.000
        Band 10     0.547  0.303  0.150     sum =  1.000
        """
        # nbands = 4, should yield 
        #   [1/1, 3/3, 0]*0.0370 
        # + [1/1, 3/3, 0]*0.2222 
        # + [1/1, 3/3, 0]*0.4444 
        # + [1/1, 3/3, 0]*0.2963
        # = [1, 1, 0]
        nzeta = _nzeta_mean_conf(4, folders, 'mean', 'svd-max')
        nzeta = [int(np.round(nz)) for nz in nzeta] # filter to integer
        self.assertEqual(nzeta, [1, 1, 0])
        # nbands = 5, should yield
        #   [2/1 + 0,     3/3 + 0,       0]*0.0370 
        # + [1/1 + 0.719, 3/3 + 0,       0.283/5]*0.2222 
        # + [1/1 + 0.392, 3/3 + 0,       0.618/5]*0.4444 
        # + [1/1 + 0.049, 3/3 + 0.116/3, 0.858/5]*0.2963
        # = [1, 1, 0]
        nzeta = _nzeta_mean_conf(5, folders, 'mean', 'wll')
        nzeta = [int(np.round(nz)) for nz in nzeta] # filter to integer
        self.assertEqual(nzeta, [1, 1, 0])
        # nbands = 10, should yield
        #   [2/1, 3/3, 5/5] * 0.0370
        # + [1.848, 3.438/3, 4.711/5] * 0.2222
        # + [1.722, 3.769/3, 4.51/5] * 0.4444
        # + [1.607, 4.021/3, 4.37/5] * 0.2963
        # = [1.726, 1.247, 0.906] = [2, 1, 1]
        nzeta = _nzeta_mean_conf(10, folders, 'mean', 'wll')
        nzeta = [int(np.round(nz)) for nz in nzeta] # filter to integer
        self.assertEqual(nzeta, [2, 1, 1])

        # test the two folder mixed case, monomer-gamma and monomer-k
        fpath1 = os.path.join(here, "testfiles/Si/jy-7au/monomer-gamma/")
        fpath2 = os.path.join(here, "testfiles/Si/jy-7au/monomer-k/")
        folders = [fpath1, fpath2]
        # nbands = 4, should yield [1, 1, 0]
        nzeta = _nzeta_mean_conf(4, folders, 'mean', 'wll')
        nzeta = [int(np.round(nz)) for nz in nzeta] # filter to integer
        self.assertEqual(nzeta, [1, 1, 0])
        # nbands = 5, should yield [2, 1, 0]
        nzeta = _nzeta_mean_conf(5, folders, 'mean', 'wll')
        nzeta = [int(np.round(nz)) for nz in nzeta] # filter to integer
        self.assertEqual(nzeta, [2, 1, 0])
        # nbands = 10, should yield [2, 1, 1]
        nzeta = _nzeta_mean_conf(10, folders, 'mean', 'wll')
        nzeta = [int(np.round(nz)) for nz in nzeta] # filter to integer
        self.assertEqual(nzeta, [2, 1, 1])

        # test the dimer-1.8-gamma
        fpath_dimer = os.path.join(here, "testfiles/Si/jy-7au/dimer-1.8-gamma/")
        folders = [fpath_dimer]
        nzeta_dimer_nbnd4 = _nzeta_mean_conf(4, folders, 'mean', 'wll')
        nzeta_dimer_nbnd5 = _nzeta_mean_conf(5, folders, 'mean', 'wll')
        nzeta_dimer_nbnd10 = _nzeta_mean_conf(10, folders, 'mean', 'wll')
        # also get the monomer-gamma result
        fpath_mono = os.path.join(here, "testfiles/Si/jy-7au/monomer-gamma/")
        folders = [fpath_mono]
        nzeta_mono_nbnd4 = _nzeta_mean_conf(4, folders, 'mean', 'wll')
        nzeta_mono_nbnd5 = _nzeta_mean_conf(5, folders, 'mean', 'wll')
        nzeta_mono_nbnd10 = _nzeta_mean_conf(10, folders, 'mean', 'wll')
        # the mixed case should return average of the two
        nzeta_mixed_nbnd4 = _nzeta_mean_conf(4, [fpath_dimer, fpath_mono], 'mean', 'wll')
        self.assertEqual(nzeta_mixed_nbnd4, 
                         [(a+b)/2 for a, b in\
                          zip(nzeta_dimer_nbnd4, nzeta_mono_nbnd4)])
        nzeta_mixed_nbnd5 = _nzeta_mean_conf(5, [fpath_dimer, fpath_mono], 'mean', 'wll')
        self.assertEqual(nzeta_mixed_nbnd5, 
                         [(a+b)/2 for a, b in\
                          zip(nzeta_dimer_nbnd5, nzeta_mono_nbnd5)])
        nzeta_mixed_nbnd10 = _nzeta_mean_conf(10, [fpath_dimer, fpath_mono], 'mean', 'wll')
        self.assertEqual(nzeta_mixed_nbnd10, 
                         [(a+b)/2 for a, b in\
                          zip(nzeta_dimer_nbnd10, nzeta_mono_nbnd10)])
    
    # @unittest.skip('skip for development')
    def test_nzeta_analysis(self):

        here = os.path.dirname(__file__)
        fpath = os.path.join(here, "testfiles/Si/jy-7au/monomer-gamma/")
        out = _nzeta_analysis(fpath)
        # print(out)
        self.assertEqual(out, 
                         [[[0, 4, 18], 
                           [1, 2, 3, 10, 11, 12, 19, 20, 21], 
                           [5, 6, 7, 8, 9, 13, 14, 15, 16, 17, 22, 23, 24]]])

    @unittest.skip('This is not a unittest. Instead, this\
     is a minimal example to investigate the synergetic\
     effect of the two parameters, nzeta and nband on the\
     orbital generation task.')
    def test_sigma_nzeta_nbands(self):
        '''for doing numerical experiments, according to sigma value,
        determine the nzeta that can produce the best orbital genreation
        results. Possible adjustable parameters are mainly in two 
        aspects:
        - nband
        - nzeta
        
        that is, the range of bands to be considered
        and the number of zeta to be generated.

        The initial guess also matters but it is clear.
        '''
        from SIAB.spillage.datparse import read_istate_info
        from SIAB.spillage.lcao_wfc_analysis import _svdfold

        rcut = 6
        nzeta = [2, 2, 0]
        # Al: 2s 2p valence electrons
        # 1s2, 2s2, 2p6, 3s2, 3p1
        ibands_atom = [0, 4, 1, 2, 3, 5, 6, 7]
        jobdir = '/root/documents/simulation/orbgen/Test1Aluminum-20241011'
        outdir = [f'Al-dimer-2.00-{rcut}au',
                  f'Al-dimer-2.50-{rcut}au',
                  f'Al-dimer-3.00-{rcut}au',
                  f'Al-dimer-3.75-{rcut}au',
                  f'Al-dimer-4.50-{rcut}au']
        occ_thr = 1e-1 # threshold on occ to determine nbands

        nthreads = 4
        option = {"maxiter": 2000, "disp": False, "ftol": 0, 
                  "gtol": 1e-6, 'maxcor': 20}

        #######
        # Run #
        #######
        minimizer = Spillage_jy()
        nbands = [0] * len(outdir)
        for i, f in enumerate(outdir):
            suffix = f
            d = os.path.join(jobdir, f, f'OUT.{suffix}')
            _, _, occ = read_istate_info(os.path.join(d, 'istate.info'))
            print(f'For {f}, occ:')
            for isp, occ_sp in enumerate(occ):
                print(f'spin = {isp}')
                for ik, occ_k in enumerate(occ_sp):
                    print(f'k = {ik}')
                    for io, o in enumerate(occ_k):
                        print(f'{o:.4e}', end=' ')
                        if io % 5 == 4:
                            print('')
                    print('')
                print('')
            print('')
            nbands[i] = len(np.where(np.array(occ) > occ_thr)[0])
            minimizer.config_add(d)

        # svd
        for i, f in enumerate(outdir):
            nbnd = nbands[i]
            d = os.path.join(jobdir, f, f'OUT.{f}')
            wfc = read_wfc_lcao_txt(os.path.join(d, "WFC_NAO_GAMMA1.txt"))[0]
            ovlp = read_triu(os.path.join(d, "data-0-S"))
            running = read_running_scf_log(os.path.join(d, "running_scf.log"))
            sigma = _svdfold(wfc, ovlp, nbnd, running["natom"],
                                running["nzeta"], 1.0)
            print(f'For {f}, nbnd = {nbnd}, sigma:')
            for l, s in enumerate(sigma[0]):
                print(f'l = {l}')
                for i, si in enumerate(s):
                    print(f'{si:.4e}', end=' ')
                    if i % 5 == 4:
                        print('')
                print('')
            print('')
        ibands = [range(n) for n in nbands]

        ###########################################################
        # the following should be a new way to give initial guess #
        ###########################################################
        suffix = f'Al-monomer-{rcut}au'
        ibands_atom = [[[0], [4]], 
                       [[1, 2, 3], [5, 6, 7]]]
        
        coef_init = [[] for _ in nzeta] # without type dimension
        for l, nz in enumerate(nzeta):
            for iz in range(nz):
                ib = ibands_atom[l][iz]
                nz_ = np.zeros_like(np.array(nzeta))
                nz_[l] = 1
                c = initgen_jy(os.path.join(jobdir, suffix, f'OUT.{suffix}'),
                                       nz_,
                                       ibands=ib,
                                       diagnosis=True)
                coef_init[l].append(c[l][0])
        
        # suppress the plotting of initial guess
        # _save_orb(coef_init, 'Al(init)', 100, rcut, os.getcwd())
        coefs = minimizer.opt([coef_init], None, 'all', ibands, option, nthreads)

        _save_orb(coefs[0], 'Al', 100, rcut, os.getcwd())


if __name__ == "__main__":
    unittest.main()

"""this file defines interface between newly implemented spillage optimization algorithm
with the driver of SIAB"""

def _coef_gen(rcut: float, ecut: float, lmax: int, value: str = "eye"):
    """Directly generate the coefficients of the orbitals instead of performing optimization
    
    Args:
    rcut: float
        the cutoff radius
    ecut: float
        the energy cutoff
    lmax: int
        the maximum angular momentum
    value: str
        the value to be generated, can be "eye"

    Returns:
    list of list of list of list of float: the coefficients of the orbitals
    """
    from SIAB.spillage.radial import _nbes
    import numpy as np
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
    """get coefficients from SIAB dumped ORBITAL_RESULTS.txt file"""
    from SIAB.spillage.orbio import read_param
    return read_param(fcoef)

def _coef_opt(rcut: float, siab_settings: dict, folders: list, jy: bool = False):
    """generate orbitals for one single rcut value
    
    Parameters
    ----------
    rcut: float
        the cutoff radius
    siab_settings: dict
        the settings for SIAB optimization
    folders: list[list[str]]
        the folders where the ABACUS run information are stored
    jy: bool
        whether the jY basis is used
    
    Returns
    -------
    list[list[list[float]]]: the coefficients of the orbitals
    """
    import os
    import numpy as np
    from SIAB.spillage.spillage import Spillage_jy, Spillage_pw, flatten
    from SIAB.spillage.datparse import read_orb_mat
    from SIAB.spillage.listmanip import merge

    Spillage = Spillage_jy if jy else Spillage_pw

    print(f"ORBGEN: Optimizing orbitals for rcut = {rcut} au", flush = True)
    # folders will be directly the `configs`
    ibands = [[] for _ in range(len(siab_settings['orbitals']))]
    for iorb, orb in enumerate(siab_settings['orbitals']):
        if isinstance(orb['nbands_ref'], list):
            ibands[iorb] = flatten([[range(n)]*len(folders[find]) \
                            for find, n in zip(orb['folder'], orb['nbands_ref'])])
        else:
            ibands[iorb] = orb['nbands_ref']
    configs = [folders[indf] for orb in siab_settings['orbitals'] for indf in orb['folder']]
    configs = list(set([folder for f in configs for folder in f]))

    iconfs = [[] for _ in range(len(siab_settings['orbitals']))]
    for iorb, orb in enumerate(siab_settings['orbitals']):
        iconfs[iorb] = [configs.index(folder) for f in orb['folder'] for folder in folders[f]]

    #reduced = siab_settings.get('jY_type', "reduced")
    #orbgen = Spillage(reduced in ["reduced", "nullspace", "svd"])
    orbgen = Spillage()
    # load orb_matix with correct rcut
    fov = None
    for folder in configs:
        if jy:
            orbgen.config_add(os.path.join(folder, f"OUT.{os.path.basename(folder)}"))
            continue
        for fov_, fop_ in _orb_matrices(folder):
            ov, op = map(read_orb_mat, [fov_, fop_])
            assert ov['rcut'] == op['rcut'], "Data violation: rcut of ov and op matrices are different"
            if np.abs(ov['rcut'] - rcut) < 1e-10:
                print(f"ORBGEN: jy_jy, mo_jy and mo_mo matrices loaded from {fov_} and {fop_}", flush = True)
                orbgen.config_add(fov_, fop_, siab_settings.get('spill_coefs', [0.0, 1.0]))
                fov = fov_ if fov is None else fov
    symbol = configs[0].split('-')[0]
    m = [symbol, "monomer"] if not jy else [symbol, "monomer", f"{rcut}au"]
    monomer_dir = "-".join(m)
    monomer_dir = os.path.join(monomer_dir, f"OUT.{monomer_dir}") if jy else monomer_dir
    ov = os.path.join(monomer_dir, fov.replace('\\', '/').split('/')[-1]) if not jy else None

    # infer nzeta if `zeta_notation` specified as `auto`, this cause the `nzeta` to be `auto`
    nbands_ref = [[orb['nbands_ref']] if not isinstance(orb['nbands_ref'], list) else orb['nbands_ref']\
                  for orb in siab_settings['orbitals']]
    nbands_ref = [[nbands_ref[iorb]*len(folders[find]) for find in orb['folder']]
                  for iorb, orb in enumerate(siab_settings['orbitals'])]
    nzeta = [orb['nzeta'] if orb['nzeta'] != "auto" else\
             _nzeta_infer(flatten(nbands_ref[iorb]), 
                          flatten([folders[i] for i in orb['folder']]))\
             for iorb, orb in enumerate(siab_settings['orbitals'])]

    # calculate the firs param of function initgen
    lmax = max([len(nz_orb) for nz_orb in nzeta]) - 1

    # calculate maxial number of zeta for each l
    nzeta_max = [(lambda nzeta: nzeta + (lmax + 1 - len(nzeta))*[-1])(nz_orb) for nz_orb in nzeta]
    nzeta_max = [max([orb[i] for orb in nzeta_max]) for i in range(lmax + 1)]
    init_recipe = {"outdir": monomer_dir, "nzeta": nzeta_max, "diagnosis": True} if jy \
             else {"orb_mat": ov, "nzeta": nzeta_max, "diagnosis": True}
    # coefs_init = initgen(**init_recipe)

    # prepare opt params
    options = {'ftol': 0, 'gtol': 1e-6, 'maxiter': siab_settings.get('max_steps', 2000), 'disp': True, 'maxcor': 20}
    nthreads = siab_settings.get('nthreads_rcut', 1)

    # hierarchy connection between orbitals
    iorbs_ref = [orb['nzeta_from'] for orb in siab_settings['orbitals']]
    
    # optimize orbitals: run optimization for each level hierarchy
    coefs = [None for _ in range(len(siab_settings['orbitals']))]
    for iorb in range(len(siab_settings['orbitals'])):
        _temp = '\n'.join([f'        {configs[iconf]}' for iconf in iconfs[iorb]])
        nzeta_base = None if iorbs_ref[iorb] is None else nzeta[iorbs_ref[iorb]]
        print(f"""ORBGEN: optimization on level {iorb + 1} (with # of zeta functions for each l: {nzeta[iorb]}), 
        based on orbital ({nzeta_base}).
ORBGEN: Orbital fit from structures:\n{_temp}""", flush = True)
        coef_inner = coefs[iorbs_ref[iorb]] if iorbs_ref[iorb] is not None else None
        coef_init = _coef_guess(nzeta[iorb], nzeta_base, init_recipe)
        coefs_shell = orbgen.opt(coef_init, coef_inner, iconfs[iorb], ibands[iorb], options, nthreads)
        coefs[iorb] = merge(coef_inner, coefs_shell, 2) \
            if coef_inner is not None else coefs_shell
        print(f"ORBGEN: End optimization on level {iorb + 1} orbital, merge with previous orbital shell(s).", flush = True)
    return coefs

def _peel(coef, nzeta_lvl_tot):
    from copy import deepcopy
    coef_lvl = [deepcopy(coef)]
    for nzeta_lvl in reversed(nzeta_lvl_tot):
        coef_lvl.append([[coef_lvl[-1][l].pop(0) for _ in range(nzeta)] for l,nzeta in enumerate(nzeta_lvl)])
    return coef_lvl[1:][::-1]

def _coef_guess(nzeta, nzeta0, init_recipe):
    """generate initial guess for jy basis to contract.
    This function is only implemented for jy presently.
    
    Parameters
    ----------
    nzeta: list[int]
        the number of zeta functions for each l
    nzeta0: list[int]
        the number of zeta functions for each l
    init_recipe: dict
        the recipe to generate initial guess
    
    Returns
    -------
    list[list[list[float]]]: the coefficients of the orbitals
    """
    from SIAB.spillage.spillage import initgen_jy, initgen_pw, flatten
    import os

    jy = "orb_mat" not in init_recipe
    initgen = initgen_jy if jy else initgen_pw
    if jy:
        nzeta0 = [0] * len(nzeta) if nzeta0 is None else nzeta0
        ib = _nzeta_analysis(os.path.dirname(init_recipe["outdir"])) # indexed by [ispin][l] -> list of band index
        ib = [[ibsp[l][(2*l+1)*nz0:(2*l+1)*nz] for l, (nz, nz0) in enumerate(zip(nzeta, nzeta0))]
               for ibsp in ib]
        ib = [flatten(ibsp) for ibsp in ib]
        print(f"ORBGEN: initial guess based on isolated atom band-pick, indexes: {ib[0]}", flush=True)
        coef_init = initgen(**(init_recipe|{"ibands": ib[0]}))
    else:
        coef_init = initgen(**init_recipe)
    return _coef_subset(nzeta, nzeta0, coef_init)

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
    import os
    import re
    """
    on the refactor of ABACUS Numerical_Basis class
    
    This function provides a temporary solution for getting correct file name
    of orb_matrix from the folder path. There are discrepancies between the
    resulting orb_matrix files yielded with single bessel_nao_rcut parameter
    and multiple. The former will yield orb_matrix.0.dat and orb_matrix.1.dat,
    while the latter will yield orb_matrix_rcutRderivD.dat, in which R and D
    are the corresponding bessel_nao_rcut and order of derivatives of the
    wavefunctions, presently ranges from 6 to 10 and 0 to 1, respectively.

    Returns:
    tuple of str: the file names of orb_matrix and its derivative (absolute path)
    """
    old = r"orb_matrix.([01]).dat"
    new = r"orb_matrix_rcut(\d+)deriv([01]).dat"

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

def _save_orb(coefs, 
              elem, 
              ecut, 
              rcut, 
              folder,
              jY_type: str = "reduced"):
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
    import numpy as np
    import matplotlib.pyplot as plt
    from SIAB.spillage.plot import plot_chi
    from SIAB.spillage.orbio import write_nao, write_param
    from SIAB.spillage.radial import coeff_normalized2raw, coeff_reduced2raw
    import os

    coeff_converter_map = {"reduced": coeff_reduced2raw, 
                           "normalized": coeff_normalized2raw}
    syms = "SPDFGHIKLMNOQRTUVWXYZ".lower()
    dr = 0.01
    r = np.linspace(0, rcut, int(rcut/dr)+1)

    folder = os.path.abspath(folder)
    os.makedirs(folder, exist_ok=True)

    chi = _build_orb([coefs], rcut, 0.01, jY_type)
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

def _build_orb(coefs, rcut, dr: float = 0.01, jY_type: str = "reduced"):
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
    from SIAB.spillage.radial import build_raw, build_reduced, coeff_normalized2raw
    import numpy as np

    r = np.linspace(0, rcut, int(rcut/dr)+1) # hard code dr to be 0.01? no...
    if jY_type in ["reduced", "nullspace", "svd"]:
        chi = build_reduced(coefs[0], rcut, r, True)
    else:
        coefs = coeff_normalized2raw(coefs, rcut)
        chi = build_raw(coefs[0], rcut, r, 0.0, True, True)
    return chi

def iter(siab_settings: dict, calculation_settings: list, folders: list):
    """Loop over rcut values and yield orbitals"""
    rcuts = calculation_settings[0]["bessel_nao_rcut"]
    rcuts = [rcuts] if not isinstance(rcuts, list) else rcuts
    ecut = calculation_settings[0]["ecutwfc"]
    elem = [f for f in folders if len(f) > 0][0][0].split("-")[0]
    # because element does not really matter when optimizing orbitals, the only thing
    # has element information is the name of folder. So we extract the element from the
    # first folder name. Not elegant, we know.
    jY_type = siab_settings.get("jY_type", "reduced")

    run_map = {"none": "none", "restart": "restart", "bfgs": "opt"}
    run_type = run_map.get(siab_settings.get("optimizer", "none"), "none")
    for rcut in rcuts: # can be parallelized here
        # for jy basis calculation, only matched rcut folders are needed
        if run_type == "opt":
            # REFACTOR: SIAB-v3.0, get folders with matched rcut
            f_ = [[f for f in fgrp if len(f.split("-")) == 3 or \
                   float(f.split("-")[-1].replace("au", "")) == rcut] # jy case 
                  for fgrp in folders]
            jy = [f for f in folders if len(f) > 0][0][0][-2:] == "au"
            # the nzeta can be inferred here if use jy
            if jy:
                pass
            coefs_tot = _coef_opt(rcut, siab_settings, f_, jy)
            # optimize a cascade of orbitals is reasonable because the orbitals always
            # have a hierarchical structure like
            # SZ
            # [SZ] SZP = DZP
            # [DZP] SZP = TZDP
        elif run_type == "restart":
            raise NotImplementedError("restart is not implemented yet")
        else: # run_type == "none", used to generate jY basis
            coefs_tot = [_coef_gen(rcut, ecut, len(orb['nzeta']) - 1) for orb in siab_settings['orbitals']]

        # save orbitals
        for ilev, coefs in enumerate(coefs_tot): # loop over different levels...
            folder = "_".join([elem, f"{rcut}au", f"{ecut}Ry"]) # because the concept of "level" is not clear
            for coefs_it in coefs: # loop over different atom types
                _ = _save_orb(coefs_it, elem, ecut, rcut, folder, jY_type)
    return

def _nzeta_infer(nbands, folders, decimal = False):
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
    
    Returns
    -------
    nzeta: list[int]
        the inferred nzeta for each folder
    """
    import os
    import numpy as np
    from SIAB.spillage.datparse import read_wfc_lcao_txt, read_triu, \
        read_running_scf_log, read_input_script
    from SIAB.spillage.lcao_wfc_analysis import _wll

    assert isinstance(folders, list), f"folders should be a list: {folders}"
    assert all([isinstance(f, str) for f in folders]), f"folders should be a list of strings: {folders}"

    nzeta = np.array([0])
    nbands = [nbands] * len(folders) if isinstance(nbands, int) else nbands

    for folder, nband in zip(folders, nbands):
        # read INPUT and running_*.log
        params = read_input_script(os.path.join(folder, "INPUT"))
        outdir = os.path.abspath(os.path.join(folder, "OUT." + params.get("suffix", "ABACUS")))
        nspin = int(params.get("nspin", 1))
        fwfc = "WFC_NAO_GAMMA" if params.get("gamma_only", False) else "WFC_NAO_K"
        running = read_running_scf_log(os.path.join(outdir, 
                                                    f"running_{params.get('calculation', 'scf')}.log"))
        
        assert nspin == running["nspin"], \
            f"nspin in INPUT and running_scf.log are different: {nspin} and {running['nspin']}"
        
        # if nspin == 2, the "spin-up" kpoints will be listed first, then "spin-down"
        wk = running["wk"]
        for isk in range(nspin*len(wk)): # loop over (ispin, ik)
            w = wk[isk % len(wk)] # spin-up and spin-down share the wk
            wfc, _, _, _ = read_wfc_lcao_txt(os.path.join(outdir, f"{fwfc}{isk+1}.txt"))
            # the complete return list is (wfc.T, e, occ, k)
            ovlp = read_triu(os.path.join(outdir, f"data-{isk}-S"))

            wll = _wll(wfc, ovlp, running["natom"], running["nzeta"])
            
            nz = np.array([0] * wll.shape[1], dtype=float)
            degen = np.array([2*i + 1 for i in range(wll.shape[1])], dtype=float)
            for ib, wb in enumerate(wll):
                if ib >= nband: break
                wlb = np.sum(wb.real, 1)
                nz += wlb / degen
            nz *= w / nspin # average over kpoints and spin

            nzeta = np.resize(nzeta, np.maximum(nzeta.shape, nz.shape)) + nz
    
    # count the number of atoms
    assert len(running["natom"]) == 1, f"multiple atom types are not supported: {running['natom']}"
    nzeta = [nz/running["natom"][0] for nz in nzeta.tolist()]

    return [int(np.ceil(nz/len(folders))) for nz in nzeta] if not decimal else \
           [nz/len(folders) for nz in nzeta]

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
    import os
    import numpy as np
    from SIAB.spillage.datparse import read_wfc_lcao_txt, read_triu, \
        read_running_scf_log, read_input_script
    from SIAB.spillage.lcao_wfc_analysis import _wll

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
        w = wk[isk % len(wk)]
        wfc, _, _, _ = read_wfc_lcao_txt(os.path.join(outdir, f"{fwfc}{isk+1}.txt"))
        # the complete return list is (wfc.T, e, occ, k)
        ovlp = read_triu(os.path.join(outdir, f"data-{isk}-S"))

        wll = _wll(wfc, ovlp, running["natom"], running["nzeta"])

        for ib, wb in enumerate(wll): # loop over bands
            wlb = np.sum(wb.real, 1) # sum over one dimension, get dim 1 x lmax matrix
            for l, wl in enumerate(wlb):
                if wl >= count_thr:
                    out[isk % len(wk)][l].append(ib)
    return out

import unittest
class TestAPI(unittest.TestCase):

    def test_orb_matrices(self):
        import os
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

    def test_nzeta_to_initgen(self):
        import numpy as np
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

    def test_coefs_subset(self):
        import numpy as np
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

    def test_nzeta_infer(self):
        import os
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
        nzeta = _nzeta_infer(4, folders)
        self.assertEqual(nzeta, [1, 1, 0])
        # nbands = 5, should yield 2s1p as [2, 1, 0]
        nzeta = _nzeta_infer(5, folders)
        self.assertEqual(nzeta, [2, 1, 0])
        # nbands = 10, should yield 2s1p1d as [2, 1, 1]
        nzeta = _nzeta_infer(10, folders)
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
        nzeta = _nzeta_infer(4, folders)
        self.assertEqual(nzeta, [1, 1, 0])
        # nbands = 5, should yield
        #   [2/1 + 0,     3/3 + 0,       0]*0.0370 
        # + [1/1 + 0.719, 3/3 + 0,       0.283/5]*0.2222 
        # + [1/1 + 0.392, 3/3 + 0,       0.618/5]*0.4444 
        # + [1/1 + 0.049, 3/3 + 0.116/3, 0.858/5]*0.2963
        # = [1, 1, 0]
        nzeta = _nzeta_infer(5, folders)
        self.assertEqual(nzeta, [1, 1, 0])
        # nbands = 10, should yield
        #   [2/1, 3/3, 5/5] * 0.0370
        # + [1.848, 3.438/3, 4.711/5] * 0.2222
        # + [1.722, 3.769/3, 4.51/5] * 0.4444
        # + [1.607, 4.021/3, 4.37/5] * 0.2963
        # = [1.726, 1.247, 0.906] = [2, 1, 1]
        nzeta = _nzeta_infer(10, folders)
        self.assertEqual(nzeta, [2, 1, 1])

        # test the two folder mixed case, monomer-gamma and monomer-k
        fpath1 = os.path.join(here, "testfiles/Si/jy-7au/monomer-gamma/")
        fpath2 = os.path.join(here, "testfiles/Si/jy-7au/monomer-k/")
        folders = [fpath1, fpath2]
        # nbands = 4, should yield [1, 1, 0]
        nzeta = _nzeta_infer(4, folders)
        self.assertEqual(nzeta, [1, 1, 0])
        # nbands = 5, should yield [2, 1, 0]
        nzeta = _nzeta_infer(5, folders)
        self.assertEqual(nzeta, [2, 1, 0])
        # nbands = 10, should yield [2, 1, 1]
        nzeta = _nzeta_infer(10, folders)
        self.assertEqual(nzeta, [2, 1, 1])

        # test the dimer-1.8-gamma
        fpath_dimer = os.path.join(here, "testfiles/Si/jy-7au/dimer-1.8-gamma/")
        folders = [fpath_dimer]
        nzeta_dimer_nbnd4 = _nzeta_infer(4, folders, True)
        nzeta_dimer_nbnd5 = _nzeta_infer(5, folders, True)
        nzeta_dimer_nbnd10 = _nzeta_infer(10, folders, True)
        # also get the monomer-gamma result
        fpath_mono = os.path.join(here, "testfiles/Si/jy-7au/monomer-gamma/")
        folders = [fpath_mono]
        nzeta_mono_nbnd4 = _nzeta_infer(4, folders, True)
        nzeta_mono_nbnd5 = _nzeta_infer(5, folders, True)
        nzeta_mono_nbnd10 = _nzeta_infer(10, folders, True)
        # the mixed case should return average of the two
        nzeta_mixed_nbnd4 = _nzeta_infer(4, [fpath_dimer, fpath_mono])
        self.assertEqual(nzeta_mixed_nbnd4, 
                         [int(round((a+b)/2, 0)) for a, b in\
                          zip(nzeta_dimer_nbnd4, nzeta_mono_nbnd4)])
        nzeta_mixed_nbnd5 = _nzeta_infer(5, [fpath_dimer, fpath_mono])
        self.assertEqual(nzeta_mixed_nbnd5, 
                         [int(round((a+b)/2, 0)) for a, b in\
                          zip(nzeta_dimer_nbnd5, nzeta_mono_nbnd5)])
        nzeta_mixed_nbnd10 = _nzeta_infer(10, [fpath_dimer, fpath_mono])
        self.assertEqual(nzeta_mixed_nbnd10, 
                         [int(round((a+b)/2, 0)) for a, b in\
                          zip(nzeta_dimer_nbnd10, nzeta_mono_nbnd10)])

    def test_nzeta_analysis(self):
        import os
        here = os.path.dirname(__file__)
        fpath = os.path.join(here, "testfiles/Si/jy-7au/monomer-gamma/")
        out = _nzeta_analysis(fpath)
        print(out)

if __name__ == "__main__":
    unittest.main()

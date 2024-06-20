"""this file defines interface between newly implemented spillage optimization algorithm
with the driver of SIAB"""
def orbgen_of_rcut(rcut: float, siab_settings: dict, folders: list):
    """generate orbitals for one single rcut value"""
    import os
    import numpy as np
    from SIAB.spillage.spillage import Spillage, initgen
    from SIAB.spillage.datparse import read_orb_mat
    from SIAB.spillage.listmanip import merge

    print(f"ORBGEN: Optimizing orbitals for rcut = {rcut} au", flush = True)
    # folders will be directly the `configs`
    folders = list(set([item for sublist in folders for item in sublist]))
    iconfs = [[] for _ in range(len(siab_settings['orbitals']))]
    for iorb, orb in enumerate(siab_settings['orbitals']):
        iconfs[iorb] = [folders.index(f) for f in orb['folder']]
    reduced = siab_settings.get('jY_type', "reduced")
    orbgen = Spillage(reduced in ["reduced", "nullspace", "svd"])

    # load orb_matix with correct rcut
    fov = None
    for folder in folders:
        for fov_, fop_ in orb_matrices(folder):
            ov, op = map(read_orb_mat, [fov_, fop_])
            assert ov['rcut'] == op['rcut'], "Data violation: rcut of ov and op matrices are different"
            if np.abs(ov['rcut'] - rcut) < 1e-10:
                print(f"ORBGEN: jy_jy, mo_jy and mo_mo matrices loaded from {fov_} and {fop_}", flush = True)
                orbgen.add_config(ov, op, siab_settings.get('spill_coefs', [0.0, 1.0]))
                fov = fov_ if fov is None else fov
    symbol = folders[0].split('-')[0]
    monomer_dir = "-".join([symbol, "monomer"]) # weak binding
    ov = read_orb_mat(os.path.join(monomer_dir, fov.replace('\\', '/').split('/')[-1]))

    # calculate the firs param of function initgen
    lmax = max([len(orb['nzeta']) for orb in siab_settings['orbitals']]) - 1

    # calculate maxial number of zeta for each l
    nzeta_max = [(lambda nzeta: nzeta + (lmax + 1 - len(nzeta))*[-1])(orb['nzeta']) for orb in siab_settings['orbitals']]
    nzeta_max = [max([orb[i] for orb in nzeta_max]) for i in range(lmax + 1)]
    coefs_init = initgen(nzeta_max, ov, reduced)

    # prepare opt params
    options = {'ftol': 0, 'gtol': 1e-6, 'maxiter': siab_settings.get('max_steps', 2000), 'disp': True, 'maxcor': 20}
    nthreads = siab_settings.get('nthreads_rcut', 1)

    # run optimization for each level hierarchy
    nzeta = [orb['nzeta'] for orb in siab_settings['orbitals']] # element of this list will always be unique
    iorbs_ref = [orb['nzeta_from'] for orb in siab_settings['orbitals']] # index of reference orbitals
    iorbs_ref = list(map(lambda x: nzeta.index(x) if x is not None else None, iorbs_ref))
    
    # optimize orbitals
    coefs = [None for _ in range(len(siab_settings['orbitals']))]
    for iorb, orb in enumerate(siab_settings['orbitals']):
        print(f"""ORBGEN: optimization on level {iorb + 1} (with # of zeta functions for each l: {orb['nzeta']}), 
        based on orbital ({orb['nzeta_from']})""", flush = True)
        coef_inner = coefs[iorbs_ref[iorb]] if iorbs_ref[iorb] is not None else None
        coefs_shell = orbgen.opt(coefs_subset(orb['nzeta'], orb['nzeta_from'], coefs_init), coef_inner, iconfs[iorb], range(orb['nbands_ref']), options, nthreads)
        coefs[iorb] = merge(coef_inner, coefs_shell, 2) if coef_inner is not None else coefs_shell
        print(f"ORBGEN: End optimization on level {iorb + 1} orbital, merge with previous orbital shell(s).", flush = True)
    return coefs

def iter(siab_settings: dict, calculation_settings: list, folders: list):
    """Loop over rcut values and yield orbitals"""
    rcuts = calculation_settings[0]["bessel_nao_rcut"]
    rcuts = [rcuts] if not isinstance(rcuts, list) else rcuts
    for rcut in rcuts: # can be parallelized here
        coefs_tot = orbgen_of_rcut(rcut, siab_settings, folders)
        # because element does not really matter when optimizing orbitals, the only thing
        # has element information is the name of folder. So we extract the element from the
        # first folder name. Not elegant, we know.
        save_orb(coefs_tot, folders[0][0].split("-")[0], calculation_settings[0]["ecutwfc"],
                 rcut, [orb['nzeta'] for orb in siab_settings['orbitals']],
                 siab_settings.get('jY_type', "reduced"))
    return

def peel(coef, nzeta_lvl_tot):
    from copy import deepcopy
    coef_lvl = [deepcopy(coef)]
    for nzeta_lvl in reversed(nzeta_lvl_tot):
        coef_lvl.append([[coef_lvl[-1][l].pop(0) for _ in range(nzeta)] for l,nzeta in enumerate(nzeta_lvl)])
    return coef_lvl[1:][::-1]

def coefs_subset(nzeta, nzeta0, data):
    """
    Compare `nzeta` and `nzeta0`, get the subset of `data` that `nzeta` has but
    `nzeta0` does not have. Returned nested list has dimension:
    [t][l][z][q]
    t: atom type
    l: angular momentum
    z: zeta
    q: q of j(qr)Y(q)
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

def orb_matrices(folder: str):
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

def save_orb(coefs_tot, elem, ecut, rcut, nzeta, jY_type: str = "reduced"):
    import numpy as np
    import matplotlib.pyplot as plt
    from SIAB.spillage.radial import build_reduced, build_raw, coeff_normalized2raw
    from SIAB.spillage.plot import plot_chi
    from SIAB.spillage.orbio import write_nao, write_param
    import os, uuid
    """
    Plot the orbital and save .orb file
    """
    assert len(nzeta) == len(coefs_tot)
    
    dr = 0.01
    r = np.linspace(0, rcut, int(rcut/dr)+1)

    for i, coefs in enumerate(coefs_tot): # loop over all levels of orbitals
        assert len(coefs) == 1, "multiple elements?"
        if jY_type in ["reduced", "nullspace", "svd"]:
            chi = build_reduced(coefs[0], rcut, r, True) # the first param should be [l][zeta][q]
        else:
            coeff_raw = coeff_normalized2raw(coefs, rcut)
            chi = build_raw(coeff_raw[0], rcut, r, 0.0, True, True)

        syms = "SPDFGHIKLMNOQRTUVWXYZ".lower()
        nz = nzeta[i]
        suffix = "".join([f"{nz[j]}{syms[j]}" for j in range(len(nz))])

        folder, subfolder = f"{elem}_{suffix}", f"{rcut}au_{ecut}Ry"
        os.makedirs(f"{folder}/{subfolder}", exist_ok=True)
        fpng = f"{elem}_gga_{rcut}au_{ecut}Ry_{suffix}.png"
        plot_chi(chi, r, save=fpng)
        os.rename(fpng, os.path.join(f"{folder}/{subfolder}", fpng))
        plt.close()
        forb = fpng.replace(".png", ".orb")
        write_nao(forb, elem, ecut, rcut, len(r), dr, chi)
        fparam = str(uuid.uuid4())
        write_param(fparam, coefs[0], rcut, 0.0, elem)
        os.rename(forb, os.path.join(f"{folder}/{subfolder}", forb))
        os.rename(fparam, os.path.join(f"{folder}/{subfolder}", "ORBITAL_RESULTS.txt"))
        print(f"orbital saved as {forb}")

def jYgen_of_rcut(rcut: float, ecut: float, lmax: int, jY_type: str = "reduced"):
    """generate jY basis instead of zeta (the linearly combined jY). Or say the zeta 
    function now is jY itself, the number of zeta for each l is substituted with nbes_rcut"""
    from SIAB.spillage.radial import build_raw, build_reduced, coeff_normalized2raw
    from SIAB.spillage.spillage import _nbes
    import numpy as np

    if not isinstance(ecut, (int, float)):
        raise ValueError("Expected ecut to be an integer or float")
    if not isinstance(lmax, int):
        raise ValueError("Expected lmax to be an integer")
    if not isinstance(jY_type, str):
        raise ValueError("Expected jY_type to be a string")

    nbes_rcut = [_nbes(l, rcut, ecut) for l in range(lmax + 1)] # calculate the number of "zeta" for each l
    r = np.linspace(0, rcut, int(rcut/0.01)+1) # dr is hard-coded to 0.01
    # then coefs will be, still [it][l][ibes][q], but q now should be identity, which means
    # each [it][l] is a np.eye(nbes_rcut[l])
    coefs = [[np.eye(nbes_rcut[l]).tolist() for l in range(lmax + 1)]]
    if jY_type in ["reduced", "nullspace", "svd"]:
        chi = build_reduced(coefs[0], rcut, r, True)
    else:
        coeff_raw = coeff_normalized2raw(coefs, rcut)
        chi = build_raw(coeff_raw[0], rcut, r, 0.0, True, True)
    return chi
def iter_jY(calculation_settings: list, siab_settings: dict):
    """Loop over rcut values and yield jY basis"""
    import os
    import uuid
    import numpy as np
    import matplotlib.pyplot as plt
    from SIAB.spillage.plot import plot_chi
    from SIAB.spillage.orbio import write_nao, write_param
    rcuts = calculation_settings[0]["bessel_nao_rcut"]
    ecut = calculation_settings[0]["ecutwfc"]
    rcuts = [rcuts] if not isinstance(rcuts, list) else rcuts
    for rcut in rcuts: # can be parallelized here
        r = np.linspace(0, rcut, int(rcut/0.01)+1) # dr is hard-coded to 0.01
        for orb in siab_settings["orbitals"]: # loop over different levels
            lmax = len(orb["nzeta"]) - 1
            chi = jYgen_of_rcut(rcut, ecut, lmax, siab_settings.get("jY_type", "reduced"))
            folder, subfolder = f"jY_lmax{lmax}", f"{rcut}au_{ecut}Ry"
            os.makedirs(f"{folder}/{subfolder}", exist_ok=True)
            fpng = f"jY_lmax{lmax}_{rcut}au_{ecut}Ry.png"
            plot_chi(chi, r, save=fpng)
            os.rename(fpng, os.path.join(f"{folder}/{subfolder}", fpng))
            plt.close()
            print(f"jY basis saved as {fpng}")
            forb = fpng.replace(".png", ".orb")
            write_nao(forb, "jY", ecut, rcut, len(r), 0.01, chi)
            os.rename(forb, os.path.join(f"{folder}/{subfolder}", forb))
            print(f"orbital saved as {forb}")
            fparam = str(uuid.uuid4())
            write_param(fparam, chi, rcut, 0.01, "jY")
            os.rename(fparam, os.path.join(f"{folder}/{subfolder}", "ORBITAL_RESULTS.txt"))
    return

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
            for files in orb_matrices(test_folder):
                pass
        # continue to write orb_matrix.1.dat
        with open(os.path.join(test_folder, "orb_matrix.1.dat"), "w") as f:
            f.write("old version")
        # will not raise an error, but return orb_matrix.0.dat and orb_matrix.1.dat
        for files in orb_matrices(test_folder):
            self.assertEqual(files, 
            (os.path.join(test_folder, "orb_matrix.0.dat"), 
             os.path.join(test_folder, "orb_matrix.1.dat")))
        # write new version
        with open(os.path.join(test_folder, "orb_matrix_rcut6deriv0.dat"), "w") as f:
            f.write("new version")
        # expect an assertion error due to coexistence of old and new files
        with self.assertRaises(AssertionError):
            for files in orb_matrices(test_folder):
                pass
        # continue to write new files
        with open(os.path.join(test_folder, "orb_matrix_rcut6deriv1.dat"), "w") as f:
            f.write("new version")
        # expect an assertion error due to coexistence of old and new files
        with self.assertRaises(AssertionError):
            for files in orb_matrices(test_folder):
                pass
        # remove old files
        os.remove(os.path.join(test_folder, "orb_matrix.0.dat"))
        os.remove(os.path.join(test_folder, "orb_matrix.1.dat"))
        # now will return new files
        for files in orb_matrices(test_folder):
            self.assertEqual(files, 
            (os.path.join(test_folder, "orb_matrix_rcut6deriv0.dat"), 
             os.path.join(test_folder, "orb_matrix_rcut6deriv1.dat")))
        # continue to write new files
        with open(os.path.join(test_folder, "orb_matrix_rcut7deriv0.dat"), "w") as f:
            f.write("new version")
        # expect an assertion error due to odd number of new files
        with self.assertRaises(AssertionError):
            for files in orb_matrices(test_folder):
                pass
        # continue to write new files
        with open(os.path.join(test_folder, "orb_matrix_rcut7deriv1.dat"), "w") as f:
            f.write("new version")
        # now will return new files
        for ifmats, fmats in enumerate(orb_matrices(test_folder)):
            if ifmats == 0:
                self.assertEqual(fmats, 
            (os.path.join(test_folder, "orb_matrix_rcut6deriv0.dat"), 
             os.path.join(test_folder, "orb_matrix_rcut6deriv1.dat")))
            elif ifmats == 1:
                self.assertEqual(fmats, 
            (os.path.join(test_folder, "orb_matrix_rcut7deriv0.dat"), 
             os.path.join(test_folder, "orb_matrix_rcut7deriv1.dat")))
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
        
        subset = coefs_subset(nz1, None, data)
        self.assertEqual(subset, [[[data[0][0]], 
                                   [data[1][0]]]])

        subset = coefs_subset(nz2, None, data)
        self.assertEqual(subset, [[[data[0][0], data[0][1]], 
                                   [data[1][0], data[1][1]],
                                   [data[2][0]]
                                  ]])

        subset = coefs_subset(nz3, None, data)
        self.assertEqual(subset, [[[data[0][0], data[0][1], data[0][2]], 
                                   [data[1][0], data[1][1], data[1][2]],
                                   [data[2][0], data[2][1]]
                                  ]])

        subset = coefs_subset(nz3, nz2, data)
        self.assertEqual(subset, [[[data[0][2]], 
                                   [data[1][2]], 
                                   [data[2][1]]]])
        
        subset = coefs_subset(nz2, nz1, data)
        self.assertEqual(subset, [[[data[0][1]], 
                                   [data[1][1]], 
                                   [data[2][0]]]])
        
        subset = coefs_subset(nz3, nz1, data)
        self.assertEqual(subset, [[[data[0][1], data[0][2]], 
                                   [data[1][1], data[1][2]], 
                                   [data[2][0], data[2][1]]]])
        
        nz2 = [1, 2, 1]
        subset = coefs_subset(nz3, nz2, data)
        self.assertEqual(subset, [[[data[0][1], data[0][2]], 
                                   [data[1][2]], 
                                   [data[2][1]]]])

        subset = coefs_subset(nz3, nz3, data)
        self.assertEqual(subset, [[[], [], []]])
        
        with self.assertRaises(AssertionError):
            subset = coefs_subset(nz1, nz3, data) # nz1 < nz3

        with self.assertRaises(AssertionError):
            subset = coefs_subset(nz2, nz3, data) # nz2 < nz3

if __name__ == "__main__":
    unittest.main()

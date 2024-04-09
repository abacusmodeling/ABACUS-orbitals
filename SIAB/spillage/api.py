import SIAB.spillage.spillage as sss
import SIAB.spillage.datparse as ssd

def run(siab_settings: dict):
    nbes_min = 1000
    # get number of threads of present environment
    nthreads = siab_settings.get("nthreads", 4)
    """run optimization, workflow function"""

    # iterate on orbital levels
    for orb in siab_settings["orbitals"]:
        # first create spillage instance
        spillopt = sss.Spillage()
        folders = orb["folder"]
        # iterate on reference structures
        for mat, dmat in orb_matrices(folders):
            ov = ssd.read_orb_mat(mat)
            op = ssd.read_orb_mat(dmat)
            spillopt.add_config(ov, op)
            nbes_min = min(nbes_min, ov['nbes'])
        # initialize coefficients
        coef_frozen, coef0 = coefs_init(nbes_min)
        spillopt._tab_frozen(coef_frozen)
        spillopt._tab_deriv(coef0)
        # import optimization configuration from siab_settings
        ibands = orb["nbands_ref"]
        options = {'maxiter': siab_settings["max_steps"], 'disp': True, 'maxcor': 20}
        # run optimization
        coef_opt = spillopt.opt(coef0, coef_frozen, 'all', ibands, options, nthreads)

import os
import re
def orb_matrices(folder: str):
    """
    on the refactor of ABACUS Numerical_Basis class
    
    This function provides a temporary solution for getting correct file name
    of orb_matrix from the folder path. There are discrepancies between the
    resulting orb_matrix files yielded with single bessel_nao_rcut parameter
    and multiple. The former will yield orb_matrix.0.dat and orb_matrix.1.dat,
    while the latter will yield orb_matrix_rcutRderivD.dat, in which R and D
    are the corresponding bessel_nao_rcut and order of derivatives of the
    wavefunctions, presently ranges from 6 to 10 and 0 to 1, respectively.
    """
    old = r"orb_matrix.([01]).dat"
    new = r"orb_matrix_rcut(\d+)deriv([01]).dat"
    
    files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    # convert to absolute path
    files = [os.path.join(folder, f) for f in files]
    old_files = [f for f in files if re.match(old, f)]
    new_files = [f for f in files if re.match(new, f)]
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

import numpy as np
def coefs_init(nbes_min: int):
    """initialize coefficients"""
    coef_frozen = [[np.random.randn(1, nbes_min-1).tolist(),
                    np.random.randn(2, nbes_min-1).tolist(),
                    np.random.randn(1, nbes_min-1).tolist()]]

    coef0 = [[np.random.randn(1, nbes_min-1).tolist(),
                np.random.randn(2, nbes_min-1).tolist(),
                np.random.randn(1, nbes_min-1).tolist()]]
    return coef_frozen, coef0

import unittest
class TestAPI(unittest.TestCase):

    def test_orb_matrices(self):
        # test old version
        with open("orb_matrix.0.dat", "w") as f:
            f.write("old version")
        # expect an assertion error
        with self.assertRaises(AssertionError):
            for files in orb_matrices("."):
                pass
        # continue to write orb_matrix.1.dat
        with open("orb_matrix.1.dat", "w") as f:
            f.write("old version")
        # will not raise an error, but return orb_matrix.0.dat and orb_matrix.1.dat
        for files in orb_matrices("."):
            self.assertEqual(files, ("orb_matrix.0.dat", "orb_matrix.1.dat"))
        # write new version
        with open("orb_matrix_rcut6deriv0.dat", "w") as f:
            f.write("new version")
        # expect an assertion error due to coexistence of old and new files
        with self.assertRaises(AssertionError):
            for files in orb_matrices("."):
                pass
        # continue to write new files
        with open("orb_matrix_rcut6deriv1.dat", "w") as f:
            f.write("new version")
        # expect an assertion error due to coexistence of old and new files
        with self.assertRaises(AssertionError):
            for files in orb_matrices("."):
                pass
        # remove old files
        os.remove("orb_matrix.0.dat")
        os.remove("orb_matrix.1.dat")
        # now will return new files
        for files in orb_matrices("."):
            self.assertEqual(files, ("orb_matrix_rcut6deriv0.dat", "orb_matrix_rcut6deriv1.dat"))
        # continue to write new files
        with open("orb_matrix_rcut7deriv0.dat", "w") as f:
            f.write("new version")
        # expect an assertion error due to odd number of new files
        with self.assertRaises(AssertionError):
            for files in orb_matrices("."):
                pass
        # continue to write new files
        with open("orb_matrix_rcut7deriv1.dat", "w") as f:
            f.write("new version")
        # now will return new files
        for ifmats, fmats in enumerate(orb_matrices(".")):
            if ifmats == 0:
                self.assertEqual(fmats, ("orb_matrix_rcut6deriv0.dat", "orb_matrix_rcut6deriv1.dat"))
            elif ifmats == 1:
                self.assertEqual(fmats, ("orb_matrix_rcut7deriv0.dat", "orb_matrix_rcut7deriv1.dat"))
            else:
                self.fail("too many files")
        # remove new files
        os.remove("orb_matrix_rcut6deriv0.dat")
        os.remove("orb_matrix_rcut6deriv1.dat")
        os.remove("orb_matrix_rcut7deriv0.dat")
        os.remove("orb_matrix_rcut7deriv1.dat")

if __name__ == "__main__":
    unittest.main()
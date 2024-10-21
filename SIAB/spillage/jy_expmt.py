'''this is the proving ground for the spillage module'''
from SIAB.spillage.datparse import read_wfc_lcao_txt, read_triu, \
    read_running_scf_log, read_input_script, read_orb_mat
from SIAB.spillage.lcao_wfc_analysis import _wll
from SIAB.spillage.spillage import flatten, initgen_jy
import unittest
import numpy as np
import os

def _grpbnd_lnm(folder, count_thr = 1e-1, itype = 0):
    '''group the band indexes into spin, l, n and m.
    
    Parameters
    ----------
    folder: str
        the folder where the ABACUS run information are stored, will self-adjust if
        it is the outdir provided
    count_thr: float
        the threshold of wll to be considered
    itype: int
        type index
    
    Returns
    -------
    list of list of list of list of int: [ispin][l][n][m] storing the band index(es)
    '''
    params = read_input_script(os.path.join(folder, "INPUT"))
    fwfc = "WFC_NAO_GAMMA" if params.get("gamma_only", False) else "WFC_NAO_K"
    frunning = f"running_{params.get('calculation', 'scf')}.log"

    outdir = os.path.abspath(os.path.join(folder, "OUT." + params.get("suffix", "ABACUS")))\
    if any([not os.path.exists(os.path.join(folder, f)) for f in [fwfc + '1.txt', frunning]]) else folder

    nspin = int(params.get("nspin", 1))

    running = read_running_scf_log(os.path.join(outdir, frunning))
    assert nspin == running["nspin"], \
        f"nspin in INPUT and running_scf.log are different: {nspin} and {running['nspin']}"
    
    lmaxmax = len(running["nzeta"][itype]) - 1
    assert lmaxmax >= 0, f"lmaxmax should be at least 0: {lmaxmax}"
    flat_ = [[[] for _ in range(lmaxmax + 1)] for _ in range(nspin)]

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
                    flat_[isk % len(wk)][l].append(ib)
    
    # reshape the flat_, but first needed padding -1 so that the length can be divided
    # by 2l+1
    pad_ = flat_.copy()
    for i in range(nspin):
        for l in range(lmaxmax + 1):
            f = flat_[i][l]
            len_ = int(np.ceil(len(f) / (2*l + 1)) * (2*l + 1))
            pad_[i][l] = f + [-1] * (len_ - len(f)) if len(f) > 0 else f
    # reshape the pad_
    reshape_ = [[np.array(pad_[i][l]).reshape(-1, 2*l+1).tolist() 
                 for l in range(lmaxmax + 1) ] for i in range(nspin)]
    # remove all -1
    return [[[[j for j in r if j != -1] for r in reshape_[i][l]]
               for l in range(lmaxmax + 1) ] for i in range(nspin)]

def _ibands(ibnd_max, igeom, npert, ibnd_min = None):
    '''calculate the ibands needed in Spillage_*.opt function. This can be run
    at earlier stage when optimizer = bfgs. It is a helper function.
    
    Parameters
    ----------
    ibnd_max: int|list[int]
        the number of bands for all the geometries or for the single geometry
        if it is a list, it should have the same length as folders.
    igeom: int|list[int]
        the index of geometry to be calculated
    npert: list[int]
        the number of perturbation for each geometry
    ibnd_min: int|list[int]|None
        the minimum band index to be calculated
    Returns
    -------
    list[range]: the band index(es) needed
    '''
    ibnd_max = [ibnd_max] if not isinstance(ibnd_max, list) else ibnd_max
    assert all([isinstance(i, int) for i in ibnd_max]), "ibnd_max should be int or list of int"
    igeom = [igeom] if not isinstance(igeom, list) else igeom
    assert all([isinstance(i, int) for i in igeom]), "igeom should be int or list of int"
    ibnd_min = [0] * len(ibnd_max) if ibnd_min is None else ibnd_min
    ibnd_min = [ibnd_min] if not isinstance(ibnd_min, list) else ibnd_min
    assert all([isinstance(i, int) for i in ibnd_min]), "ibnd_min should be int or list of int"
    assert len(ibnd_min) == len(ibnd_max), "ibnd_min and ibnd_max should have the same length"
    assert len(ibnd_max) == len(npert), "ibnd_max and npert should have the same length"

    return flatten([[range(ibmin, ibmax)]*npert[ig] for ig, ibmax, ibmin in zip(igeom, ibnd_max, ibnd_min)])

def _coef_init(outdir, nzeta, izmin = None, diagnosis = False):
    '''from outdir get the initial guess of all coefficients in number of
    nzeta.
    
    Parameters
    ----------
    outdir: str
        the folder where the ABACUS run information are stored, always it
        is the folder of monomer calculation
    nzeta: list[int]
        the number of zeta orbitals for each angular momentum
    izmin: list[int]|None
        from which index of zeta for each l to start genearte nzeta[l] number of
        groups of coefficients
    diagnosis: bool
        whether the diagnosis information (eigval of <jy|ref><ref|jy>) is printed
    
    Returns
    -------
    list[list[list[float]]]: the initial guess of all coefficients, indexed by
    [l][n][q]'''

    ibnd_lnm = _grpbnd_lnm(outdir)
    izmin = [0] * len(nzeta) if izmin is None else izmin
    assert len(izmin) == len(nzeta), "izmin and nzeta should have the same length"
    # ensure there are enough bands for the calculation
    if not all([len(ibnd_lnm[0][l]) >= nz for l, nz in enumerate(nzeta)]):
        raise ValueError("initgen error: requiring more bands than actually calculated")
    coef = [[] for _ in nzeta]
    for l, nz in enumerate(nzeta):
        iz0 = izmin[l]
        for iz in range(nz):
            ib = ibnd_lnm[0][l][iz0 + iz] # how to deal with the spin?...
            nz_ = np.zeros_like(np.array(nzeta))
            nz_[l] = 1
            c = initgen_jy(outdir, nz_, ib, diagnosis=diagnosis)
            coef[l].append(c[l][0])
    return coef

class TestSpillageExperimental(unittest.TestCase):

    def test_grpbnd_lnm(self):
        here = os.path.dirname(__file__)
        fpath = os.path.join(here, "testfiles/Si/jy-7au/monomer-gamma/OUT.ABACUS")
        out = _grpbnd_lnm(fpath)
        self.assertEqual(out, [[[[0], [4], [18]], 
                                [[1, 2, 3], [10, 11, 12], [19, 20, 21]], 
                                [[5, 6, 7, 8, 9], [13, 14, 15, 16, 17], [22, 23, 24]]]])

    def test_ibands(self):
        self.assertEqual(_ibands(25, [0], [1]), [range(0, 25)])
        self.assertEqual(_ibands([25], [0], [1]), [range(0, 25)])
        self.assertEqual(_ibands([25, 25], [0, 1], [1, 1]), [range(0, 25), range(0, 25)])

    def test_coef_init(self):
        from SIAB.spillage.api import _save_orb

        here = os.path.dirname(__file__)
        fpath = os.path.join(here, "testfiles/Si/jy-7au/monomer-gamma/OUT.ABACUS")
        # catch the ValueError
        with self.assertRaises(ValueError):
            _coef_init(fpath, [2, 3, 5])
        out = _coef_init(fpath, [2, 3, 3])
        # the shape should be, 3 * [2, 3, 3] * nq
        self.assertEqual(len(out), 3)
        for l in range(3):
            self.assertEqual(len(out[l]), [2, 3, 3][l])
        
        return # disable the following practical example
        # practical example
        jobdir = '/root/documents/simulation/orbgen/Test1Aluminum-20241011/'
        rcut = 6
        suffix = f'Al-monomer-{rcut}au'
        out = _coef_init(os.path.join(jobdir, suffix, f'OUT.{suffix}'), 
                         [2, 2, 0], diagnosis=True)
        _save_orb(out, 'Al(1)', 100, rcut, os.getcwd())
        out = _coef_init(os.path.join(jobdir, suffix, f'OUT.{suffix}'), 
                         [1, 1, 0], izmin=[1, 1, 0], diagnosis=True)
        _save_orb(out, 'Al(2)', 100, rcut, os.getcwd())

if __name__ == "__main__":
    unittest.main()
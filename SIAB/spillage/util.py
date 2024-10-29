from SIAB.spillage.datparse import read_istate_info, read_input_script, read_kpoints
import os
import numpy as np
import unittest
import ast # for literal_eval

def ptg_spilopt_params_from_dft(calculation_settings, siab_settings, folders):
    """prepare the input for orbital optimization task.
    Because the folder name might not be determined at the beginning of task if perform
    `auto` on bond_length, the exact folder name will be given uniformly after performing
    abacus calculation, therefore it is compulsory to update the folder name in siab_settings
    and cannot be somewhere else."""
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
        orbital["lmax"] = max([calculation_settings[i]["lmaxmax"] for i in orbital["folder"]])
        # the key "folder" has a more reasonable alternative: "abacus_setup"

        # MOVE: let siab_settings know exactly what folders are used for orbital generation
        # in folders, folders are stored collectively according to reference structure
        # and the "abacus_setup" is then expanded to the corresponding folder
        # therefore this setup is to expand the "folder" key to the corresponding folder
        # list
        matmaps_ = orbital["folder"] if siab_settings.get("optimizer", "none") not in ["none", "restart"] else 0
        orbital["folder"] = []
        for i in matmaps_:
            orbital["folder"].extend(folders[i])

    return siab_settings

def neo_spilopt_params_from_dft(calculation_settings, siab_settings, folders):
    '''this function for new method (bfgs) is just a way to make the interface
    unified with the old version of SIAB.
    But there are indeed some tedious work to do:
    
    1. get all the orbital identifier rcut(s), ecut and the element symbol
    2. extract the orbital optimization options (important: refresh the nbands_ref
    if it is specified as str involving `occ` and `all`)
    '''
    rcuts = calculation_settings[0]["bessel_nao_rcut"]
    rcuts = [rcuts] if not isinstance(rcuts, list) else rcuts
    ecut = calculation_settings[0]["ecutwfc"]
    elem = [f for f in folders if len(f) > 0][0][0].split("-")[0]
    # because element does not really matter when optimizing orbitals, the only thing
    # has element information is the name of folder. So we extract the element from the
    # first folder name. Not elegant, we know.
    primitive_type = siab_settings.get("primitive_type", "reduced")

    run_map = {"none": "none", "restart": "restart", "bfgs": "opt"}
    run_type = run_map.get(siab_settings.get("optimizer", "none"), "none")

    # FIXME: it is also possible to let the orb['nbands_ref'] to be dependent on the
    # rcut, but not for now...
    orbparams = siab_settings["orbitals"]
    for orb in orbparams:

        # indexes of folders, it is from the geometries to refer, make it a list
        indf = orb.get("folder", 0)
        if not isinstance(indf, list):
            indf = [indf]
            
        # nbands to ref, make it a list. This means all perts in one geom share
        # the same nbands_ref
        nbnd = orb.get("nbands_ref", 0)
        if not isinstance(nbnd, list):
            nbnd = [nbnd] * len(indf)
        
        # write-back
        orb["folder"] = indf # only one-layer of indexes, means select all perts of one geom
        orb["nbands_ref"] = [[_spil_bnd_autoset(nb, f) for f in folders[i]
                             if f'{rcuts[0]}au' in f] # HERE can introduce the dependence on rcut
                             for nb, i in zip(nbnd, indf)]
        # now the folder is list of indexes igeom
        # now the nbands_ref is indexed by [igeom][ipert]

    shared_option = {'orbparams': orbparams, 
                     'maxiter': siab_settings.get("max_steps", 2000),
                     'nthreads': siab_settings.get("nthreads", 4),
                     'jy': calculation_settings[0].get('basis_type', 'pw') != 'pw',
                     'spill_coefs': siab_settings.get("spill_coefs", None)}

    return rcuts, ecut, elem, primitive_type, run_type, shared_option

def _spil_bnd_autoset(pattern: int|str, 
                      folder: str,
                      occ_thr = 5e-1,
                      merge_sk = 'max'):
    '''set the range of bands to optimize the Spillage
    
    Parameters
    ----------
    pattern : str
        the value of nbands_ref set by user, might be `occ` or `all` or
        simple algebratic expression. Or a simple integer.
    folder: str
        for `occ`, `all` and related expressions, the istate.info file
        is needed to determine the number of bands to optimize.
    occ_thr: float
        the threshold to determine the occupied bands, default is 5e-1
    merge_sk: str
        decide how to merge_sk the bands of different spins and kpoints,
        , can be `max`, `min` or `mean`, default is `max`
    Returns
    -------
    int
        the number of bands to optimize
    '''

    parent = os.path.dirname(folder)
    base = os.path.basename(folder)
    if 'OUT.' not in base:
        param = read_input_script(os.path.join(folder, 'INPUT'))
        folder = 'OUT.' + param.get('suffix', 'ABACUS')
        folder = os.path.join(parent, base, folder)

    # occ indexed by [ispin][ik][ibnd]
    kpts, _, occ = read_istate_info(os.path.join(folder, 'istate.info'))
    kpts_, wk = read_kpoints(os.path.join(folder, 'kpoints'))
    assert np.allclose(kpts, kpts_), f'Inconsistent kpoints in {folder}/ istate.info and kpoints'

    nbnd = [[(len(occ_sk), len(np.where(np.array(occ_sk) >= occ_thr*w)[0])) 
             for occ_sk, w in zip(occ_sp, wk)] for occ_sp in occ]
    nbnd = np.array(nbnd).reshape(-1, 2)
    assert nbnd.shape == (len(kpts)*len(occ), 2), f'Inconsistent shape of nbnd {nbnd.shape}'
    
    # take min, max or mean of the bands over all (ispin, ik) on (nbands, occ_bands)
    if merge_sk == 'max':
        nbnd = nbnd.max(axis=0)
    elif merge_sk == 'min':
        nbnd = nbnd.min(axis=0)
    elif merge_sk == 'mean':
        nbnd = nbnd.mean(axis=0)
    else:
        raise ValueError(f"merge_sk method {merge_sk} is not supported")
    
    nall, nocc = nbnd
    if isinstance(pattern, int):
        if pattern < 0 or pattern > nall:
            raise ValueError(f"nbands_ref {pattern} is out of range (0, {nall})")
        return pattern
    else:
        assert isinstance(pattern, str), f"nbands_ref {pattern} is not a string."
    try:
        return int(ast.literal_eval(pattern.replace('occ', str(nocc)).replace('all', str(nall))))
    except (ValueError, SyntaxError):
        raise ValueError(f"nbands_ref {pattern} is not a valid expression.")

class TestSpillageUtilities(unittest.TestCase):
    def test_spil_bnd_autoset(self):
        here = os.path.dirname(__file__)
        outdir = os.path.join(here, 'testfiles', 'Si', 'jy-7au', 'monomer-k')
        
        # test for simple integer
        out = _spil_bnd_autoset(10, outdir)
        self.assertEqual(out, 10)

        # out of band range
        with self.assertRaises(ValueError):
            _spil_bnd_autoset(10000, outdir)
        
        # occ
        out = _spil_bnd_autoset('occ', outdir)
        self.assertEqual(out, 4)

        # all
        out = _spil_bnd_autoset('all', outdir)
        self.assertEqual(out, 25)

        # simple expression
        out = _spil_bnd_autoset('occ+2', outdir)
        self.assertEqual(out, 6)

        out = _spil_bnd_autoset('all-2', outdir)
        self.assertEqual(out, 23)

if __name__ == "__main__":
    unittest.main()
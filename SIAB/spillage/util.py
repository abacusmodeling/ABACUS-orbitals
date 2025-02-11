# in-built modules
import os
import unittest
import re

# third-party modules
import numpy as np
from scipy.integrate import simps

# local modules
from SIAB.spillage.datparse import read_istate_info, read_input_script, read_kpoints
from SIAB.spillage.radial import _nbes

def _legacy_dft2spillparam(calculation_settings, siab_settings, folders):
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

def literal_eval(expr):
    '''evaluate the expression, but only allow the basic arithmetic operations'''
    allowed = set('+-*/()0123456789')
    if not set(expr) <= allowed:
        raise ValueError(f"Expression {expr} contains invalid characters")
    words = re.findall(r'\d+|\+|\-|\*|\/|\(|\)', expr)
    if not words[0].isdigit():
        raise ValueError(f'Not supported expression {expr}')
    out = int(words[0])
    op_map = {'+': lambda x, y: x + y,
              '-': lambda x, y: x - y,
              '*': lambda x, y: x * y,
              '/': lambda x, y: x / y}
    op = None
    for w in words[1:]:
        if w.isdigit():
            if op is None:
                raise ValueError(f'Not supported expression {expr}')
            out = op_map[op](out, int(w))
        else:
            op = w
    return out

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
        return int(literal_eval(pattern.replace('occ', str(nocc)).replace('all', str(nall))))
    except (ValueError, SyntaxError):
        raise ValueError(f"nbands_ref {pattern} is not a valid expression.")

def _spill_opt_param(raw):
    '''convert the scheme to the spillage module acceptable parameters
    
    Parameters
    ----------
    raw : dict
        the scheme of how to generate the orbitals
    
    Returns
    -------
    dict
        the spillage module acceptable parameters
    '''
    SCIPY_SUPPORTED = ['ftol', 'gtol', 'maxcor']
    TORCH_SUPPORTED = ['lr', 'beta', 'eps', 'weight_decay']

    common = {'maxiter': raw.get('max_steps', 5000), 
              'disp': raw.get('verbose', True)}
    scipy_ = {k.replace('scipy.', ''): v for k, v in raw.items() 
              if k.startswith('scipy.') and k.split('.')[1] in SCIPY_SUPPORTED}
    torch_ = {k.replace('torch.', ''): v for k, v in raw.items() 
              if k.startswith('torch.') and k.split('.')[1] in TORCH_SUPPORTED}
    
    optimizer = raw.get('optimizer', 'scipy.bfgs')
    impl, method = optimizer.split('.')
    if impl not in ['scipy', 'torch']:
        raise ValueError(f"Spillage optimizer implementation framework `{impl}` is not supported")

    specific = scipy_ if impl == 'scipy' else torch_|{'method': method}
    
    report  =  '\nSpillage optimization parameterization summary\n'
    report +=  '---------------------------------------------\n'
    report += f'optimizer: {optimizer}'
    report +=  '\nCommon parameters:\n'
    report +=  "\n".join([f"{k:>10s}: {v}" for k, v in common.items()])
    report +=  '\nSpecific parameters:\n'
    report +=  "\n".join([f"{k:>10s}: {v}" for k, v in specific.items()])
    print(report, flush=True)
    return optimizer, {**common, **specific}

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

    def test_spilparam(self):

        test = {'max_steps': 1000, 'verbose': False, 'optimizer': 'scipy.bfgs',
                'scipy.ftol': 1e-6, 'scipy.gtol': 1e-6, 'scipy.maxcor': 10}
        optimizer, options = _spill_opt_param(test)
        self.assertEqual(optimizer, 'scipy.bfgs')
        self.assertEqual(options['maxiter'], 1000)
        self.assertEqual(options['disp'], False)
        self.assertEqual(options['ftol'], 1e-6)
        self.assertEqual(options['gtol'], 1e-6)
        self.assertEqual(options['maxcor'], 10)

def _padding(arr, n):
    '''pad the array to the length of n'''
    if len(arr) < n:
        return np.pad(arr, (0, n-len(arr)))
    return arr[:n]

def _jyproj(radial, dr, l, ecut, rcut, primitive_type: str = 'reduced'):
    '''
    project any given radial f onto jy basis \sum |jl><jl|,
    
    Parameters
    ----------
    radial: np.ndarray
        the radial function to project
    dr: float
        the grid spacing of the radial function
    l: int
        the angular momentum quantum number
    ecut: float
        the kinetic energy cutoff of the jy basis
    rcut: float
        the cutoff radius of the jy basis
    primitive_type: str
        the type of jy basis, can be `reduced` or `normalized`
    
    Returns
    -------
    np.ndarray
        the coefficients of the projection
    float
        the norm of projection error 
    '''
    # the temporary way to avoid circular import
    from SIAB.spillage.legacy.api import _coef_gen, _coef_griddata 

    coefs = _coef_gen(rcut, ecut, primitive_type, l=l)
    assert len(coefs[0]) == l + 1, 'Unexpected number of angular momentum.'
    chi = np.array(_coef_griddata(coefs, rcut, dr, primitive_type)[-1])
    
    _, nr = chi.shape
    radial = np.array(_padding(radial, nr))
    
    r = np.linspace(0, rcut, int(rcut/dr)+1)
    proj = np.array([simps(c*radial * r**2, r) for c in chi]) # <jl|f>
    return proj, np.linalg.norm(radial - np.dot(proj, chi))

class TestRadialProjection(unittest.TestCase):
    def test_jyproj(self):
        from SIAB.spillage.legacy.api import _coef_griddata
        njy = _nbes(l=1, rcut=7, ecut=20) - 1
        coefs = [[[] for _ in range(1)] + [np.eye(njy).tolist()]]
        radials = _coef_griddata(coefs, 7, 0.01)[-1]
        
        outmat = []
        for radial in radials:
            proj, _ = _jyproj(radial, 0.01, 1, 20, 7)
            outmat.append(proj)
        outmat = np.array(outmat)
        self.assertEqual(outmat.shape, (len(radials), len(radials)))
        self.assertTrue(np.linalg.norm(outmat - np.eye(len(radials))) < 1e-10)

if __name__ == "__main__":
    unittest.main()
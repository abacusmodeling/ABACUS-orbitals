import numpy as np
from scipy.optimize import curve_fit
import unittest
from SIAB.abacus.io import DEFAULT_BOND_LENGTH
from SIAB.data.interface import RCOVALENT
from SIAB.io.convention import dft_folder
from SIAB.io.read_output import read_energy, read_natom
import os
import re

def _morse(r, De, a, re, e0=0.0):
    '''Morse potential function

    Parameters
    ----------
    r : list[float]|np.array
        the bond length
    De : float
        the dissociation energy
    a : float
        the bond length scalling parameter
    re : float
        the equilibrium bond length
    e0 : float
        the zero point energy
    '''
    r = np.array(r)
    return (De * (1.0 - np.exp(-a*(r-re)))**2.0 + e0).tolist()

def blgen(elem: str,
          proto: str,
          n: int = 5,
          dr_l: float = 0.2,
          dr_r: float = 0.5):
    '''generate initial guess for bond lengths

    Parameters
    ----------
    elem : str
        the element symbol
    proto : str
        the prototype name
    n : int
        the number of bond lengths to generate on each side of the equilibrium bond length
    dr_l : float
        the stepsize of bond length decrease, default is 0.2 Angstrom
    dr_r : float
        the stepsize of bond length increase, default is 0.5 Angstrom
    '''
    try:
        return DEFAULT_BOND_LENGTH[proto][elem]
    except KeyError:
        # no predefined bond length, use covalent radius
        bl0 = RCOVALENT[elem]
        return _blgrid_gen(bl0, dr_l, dr_r, n)

def _blgrid_gen(center: float, 
                dr_l: float = 0.2,
                dr_r: float = 0.5, 
                nr: int = 5):
    '''generate initial guess for bond lengths
    
    Parameters
    ----------
    center : float
        the equilibrium bond length
    dr_l : float
        the stepsize of bond length decrease, default is 0.2 Angstrom
    dr_r : float
        the stepsize of bond length increase, default is 0.5 Angstrom
    nr : int
        the number of bond lengths to generate on each side of the equilibrium bond length
    '''
    blmin = center - dr_l*nr
    blmax = center + dr_r*nr
    print(f'Searching bond lengths from {blmin:4.2f} to {blmax:4.2f} Angstrom.', flush=True)
    left = np.linspace(blmin, center, nr+1).tolist()
    right = np.linspace(center, blmax, nr+1, endpoint=True).tolist()
    bl = left + right[1:]
    return [round(b, 2) for b in bl]

def _fit(r: list, e: list):
    '''fitting morse potential, return D_e, a, r_e, e_0 in the equation below:

    V(r) = D_e * (1-exp(-a(r-r_e)))^2 + e_0

    Use scipy.optimize.curve_fit to fit the parameters
    
    Parameters
    ----------
    r : list[float]
        the list of bond lengths to fit Morse potential
    e : list[float]
        the list of energies corresponding with r to fit Morse potential
    
    Returns
    -------
    D_e: float
        the dissociation energy
    a: float
        bond length scalling parameter
    r_e: float
        equilibrium bond length
    e_0: float
        zero point energy
    ''' 
    # precondition the fitting problem, first assert the location of minimum energy point
    # always be sure there are at least two points on the both
    # left and right side of the minimum energy point
    idx_min = e.index(min(e))
    assert idx_min > 1, 'There are fewer than 2 points on the left side of the minimum energy point, which indicates unreasonable bond length sampling.'
    assert idx_min < len(e) - 2, 'There are fewer than 2 points on the right side of the minimum energy point.'
    assert len(e) >= 5, 'There are fewer than 5 points in total.'
    # set threshold to be 10, this will force the point with the energy no higher than 10 eV
    cndt_thr = 10 # eV
    ediff = max(e) - min(e)
    conditioned = ediff < cndt_thr # if true, the fitting problem is relatively balanced
    while not conditioned:
        # remove the highest energy point
        idx_remove = e.index(max(e))
        if idx_remove >= idx_min:
            break # it means all points are evenly distributed around the minimum energy point
        print('MORSE POTENTIAL FITTING: remove the highest energy point %4.2f eV at bond length %4.2f Angstrom.'%(e[idx_remove], r[idx_remove]), flush=True)
        e.pop(idx_remove)
        r.pop(idx_remove)
        # refresh the condition
        ediff = max(e) - min(e)
        conditioned = ediff < cndt_thr or len(e) == 5

    try:
        popt, pcov = curve_fit(f=_morse, 
                               xdata=r, 
                               ydata=e,
                               p0=[e[-1] - min(e), 1.0, 2.7, min(e)])
    except RuntimeError:
        raise ValueError('Fitting failed. This may due to the ill-defined PES caused by low-quality pseudopotential')

    if np.any(np.diag(pcov) > 1e5):
        print('WARNING: fitting parameters are not accurate.', flush=True)

    # MUST SATISFY THE PHYSICAL MEANING
    assert popt[0] > 0 # D_e, dissociation energy MUST be positive
    assert popt[1] > 0 # a, Morse potential parameter MUST be positive
    assert popt[2] > 0 # r_e, equilibrium bond length MUST be positive
    assert popt[3] < 0 # e_0, zero point energy ALWAYS be negative

    print('Morse potential fitting results:', flush=True)
    print(f'{"D_e":6s}: {popt[0]:15.10f} {"eV":10s} (Bond dissociation energy)', flush=True)
    print(f'{"a":6s}: {popt[1]:15.10f} {"":10s} (Morse potential parameter)', flush=True)
    print(f'{"r_e":6s}: {popt[2]:15.10f} {"Angstrom":10s} (Equilibrium bond length)', flush=True)
    print(f'{"e_0":6s}: {popt[3]:15.10f} {"eV":10s} (Zero point energy)', flush=True)
    
    return popt[0], popt[1], popt[2], popt[3]

def _select(bl0: float, 
            e0: float, 
            bl: list, 
            e: list, 
            n: int = 5,
            ethr: float = 1.5):
    '''with the fitted optimal bond length and minimal energy, find `n` (bl, e) pairs that
    form the energy well with depth defined by ethr, whose unit is eV
    
    Parameters
    ----------
    bl0 : float
        the Equilibrium bond length
    e0 : float
        the zero point energy
    bl : list[float]
        the list of bond lengths
    e : list[float]
        the list of energies calculated for each bond length in bl
    ethr : float
        the threshold (the depth of well) used for determining the range of bond lengths
    
    Returns
    -------
    list[float]
        the bond lengths that will be considered, whose corresponding energies define the
        energy well with energy depth ethr
    '''
    if n // 2 == 0:
        n = n + 1
        print(f'Number of bond lengths to select must be odd, reset to {n}.', flush=True)
    if n > len(bl):
        raise ValueError('Number of bond lengths to select is larger than the number of bond lengths provided.')

    idx = np.argsort(bl)
    bl, e = np.array(bl)[idx], np.array(e)[idx]

    if bl0 < bl[0] or bl0 > bl[-1]:
        raise ValueError('Equilibrium bond length is out of the range of bond lengths.')
    
    e = e - e0 # shift the energy to zero point energy
    if any(e < 0):
        raise ValueError('Ill-defined PES: there are points with energies lower than zero point energy.')
    
    # find the inteval of bond lengths that form the energy well
    center = np.argmin(e)
    
    left = 0
    while e[left] - e[center] > ethr:
        left += 1
    if center - left < n // 2:
        raise ValueError('Not enough bond lengths on the left side of the minimum energy point.')
    left = np.linspace(left, center, n//2+1).astype(int)

    right = len(bl) - 1
    while e[right] - e[center] > ethr:
        right -= 1
    if right - center < n // 2:
        raise ValueError('Not enough bond lengths on the right side of the minimum energy point.')
    right = np.linspace(center, right, n//2+1, endpoint=True).astype(int)    
    
    idx = np.concatenate([left, right[1:]])
    return [round(b, 2) for b in bl[idx]]

def _blfilter(bl: list,
              e: list,
              ethr: float = 1.5,
              n: int = 5):
    '''filter the bond lengths that form the energy well with depth defined by ethr, whose unit is eV
    
    Parameters
    ----------
    bl : list[float]
        the list of bond lengths
    e : list[float]
        the list of energies calculated for each bond length in bl
    ethr : float
        the threshold (the depth of well) used for determining the range of bond lengths
    n : int
        the number of bond lengths to select
    
    Returns
    -------
    list[float]
        the bond lengths that will be considered, whose corresponding energies define the
        energy well with energy depth ethr
    '''
    D_e, a, r_e, e_0 = _fit(bl, e)
    selected = _select(r_e, e_0, bl, e, n, ethr)
    return selected

def jobfilter(root: str,
              elem: str,
              proto: str,
              pertkind: str,
              pertmags: list,
              rcut: float,
              n: int = 5,
              ethr: float = 1.5):
    '''filter the jobs' folder names based on the bond length scan
    
    Parameters
    ----------
    root : str
        the root directory of the jobs
    elem : str
        the element symbol
    proto : str
        the prototype name
    pertkind : str
        the kind of perturbation
    pertmags : list[float]
        the list of perturbation magnitudes
    rcut : float
        the cutoff radius of the orbital to generate
    n : int
        the number of bond lengths to select
    ethr : float
        the threshold (the depth of well) used for determining the range of bond lengths
    
    Returns
    -------
    list[float]
        the bond lengths that will be considered, whose corresponding energies define the
        energy well with energy depth ethr
    '''
    # the case that does not need to filter
    if isinstance(pertmags, list) and all([isinstance(p, float) for p in pertmags]):
        return pertmags

    # otherwise...
    magicnum = 10086.00 + np.random.randint(1000, 9999)
    magic = f'{magicnum:.2f}'
    template = dft_folder(elem, proto, magicnum, rcut)
    
    bl, e = [], []
    for c in os.listdir(root):
        m = re.match(re.compile(template.replace(magic, '(\d\.\d\d)')), c)
        if m is not None:
            bl.append(float(m.group(1)))
            nat = read_natom(os.path.join(root, c), None)
            e.append(read_energy(os.path.join(root, c), None)/nat)

    return _blfilter(bl, e, ethr, n) if pertmags == 'scan' else bl

class TestBLSCAN(unittest.TestCase):
    '''test the bond length scan module'''
    def test_blgen(self):
        '''test the bond length generation'''
        bl = _blgrid_gen(2.0)
        self.assertEqual(len(bl), 11)
        self.assertEqual(bl[0], 1.0)
        self.assertEqual(bl[-1], 4.5)
        self.assertEqual(bl[5], 2.0)

    def test_fit(self):
        '''test the Morse potential fitting'''
        r = [1.0, 1.5, 2.0, 2.5, 3.0]
        e = [0.0, -0.5, -1.0, -0.5, 0.0]
        with self.assertRaises(ValueError):
            _fit(r, e)
        
        r = [1.0, 1.5, 2.0, 2.5, 3.0]
        e = _morse(r, 1.0, 1.0, 2.0, -0.5)
        D_e, a, r_e, e_0 = _fit(r, e)
        self.assertAlmostEqual(D_e, 1.0)
        self.assertAlmostEqual(a, 1.0)
        self.assertAlmostEqual(r_e, 2.0)
        self.assertAlmostEqual(e_0, -0.5)

    def test_select(self):
        '''test the bond length selection'''
        bl = np.linspace(1.0, 3.0, 11).tolist()
        e = _morse(bl, 1.0, 1.0, 2.0, -0.5)
        D_e, a, r_e, e_0 = _fit(bl, e)
        selected = _select(r_e, e_0, bl, e, 5, 1.5)
        self.assertEqual(len(selected), 5)
        self.assertListEqual(selected, [1.4, 1.6, 2.0, 2.4, 3.0])

    def test_blfilter(self):
        '''test the bond length filtering'''
        bl = np.linspace(1.0, 3.0, 11).tolist()
        e = _morse(bl, 1.0, 1.0, 2.0, -0.5)
        selected = _blfilter(bl, e, 1.5, 5)
        self.assertEqual(len(selected), 5)
        self.assertListEqual(selected, [1.4, 1.6, 2.0, 2.4, 3.0])

    def test_jobfilter(self):
        # test the case that does not need to filter
        root = os.getcwd()
        elem = 'H'
        proto = 'dimer'
        pertkind = 'stretch'
        pertmags = [0.0, 0.1, 0.2]
        rcut = 3.0
        n = 5
        ethr = 1.5
        selected = jobfilter(root, elem, proto, pertkind, pertmags, rcut, n, ethr)
        self.assertEqual(len(selected), 3)
        self.assertListEqual(selected, pertmags)

        # test the case that needs to filter
        bl = np.linspace(1.0, 3.0, 11).tolist()
        e = _morse(bl, 1.0, 1.0, 2.0, -0.5)
        candidates = []
        for bl_, e_ in zip(bl, e):
            # make dir the dft_folder/OUT.ABACUS
            candidates.append(os.path.join(root, dft_folder(elem, proto, bl_, rcut)))
            os.makedirs(os.path.join(candidates[-1], 'OUT.ABACUS'), exist_ok=True)
            with open(os.path.join(root, 
                                   dft_folder(elem, proto, bl_, rcut),
                                   'OUT.ABACUS',
                                   'running_scf.log'), 'w') as f:
                f.write(f'TOTAL ATOM NUMBER 1\n!FINAL_ETOT_IS {e_} eV\n')
        selected = jobfilter(root, elem, proto, pertkind, 0.0, rcut, n, ethr)
        for c in candidates:
            os.system(f'rm -rf {c}')
        self.assertEqual(len(selected), 5)
        self.assertListEqual(selected, [1.40, 1.60, 2.00, 2.40, 3.00])

        candidates = []
        for bl_, e_ in zip(bl, e):
            # make dir the dft_folder/OUT.ABACUS
            candidates.append(os.path.join(root, dft_folder(elem, proto, bl_, None)))
            os.makedirs(os.path.join(candidates[-1], 'OUT.ABACUS'), exist_ok=True)
            with open(os.path.join(root, 
                                   dft_folder(elem, proto, bl_, None),
                                   'OUT.ABACUS',
                                   'running_scf.log'), 'w') as f:
                f.write(f'TOTAL ATOM NUMBER 1\n!FINAL_ETOT_IS {e_} eV\n')
        selected = jobfilter(root, elem, proto, pertkind, 0.0, None, n, ethr)
        for c in candidates:
            os.system(f'rm -rf {c}')
        self.assertEqual(len(selected), 5)
        self.assertListEqual(selected, [1.40, 1.60, 2.00, 2.40, 3.00])

if __name__ == '__main__':
    unittest.main()
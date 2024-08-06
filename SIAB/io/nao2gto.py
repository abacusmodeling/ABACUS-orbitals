
"""
From Basis Set Exchange (https://www.basissetexchange.org/), we can download the basis set in the Gaussian format.
STO-3G of H (CP2K SZV level)
H     0
S    3   1.00
      0.3425250914D+01       0.1543289673D+00
      0.6239137298D+00       0.5353281423D+00
      0.1688554040D+00       0.4446345422D+00

6-31G of H
H     0
S    3   1.00
      0.1873113696D+02       0.3349460434D-01
      0.2825394365D+01       0.2347269535D+00
      0.6401216923D+00       0.8137573261D+00
S    1   1.00
      0.1612777588D+00       1.0000000

6-31G* of H (CP2K DZVP level)
H     0
S    3   1.00
      0.1873113696D+02       0.3349460434D-01
      0.2825394365D+01       0.2347269535D+00
      0.6401216923D+00       0.8137573261D+00
S    1   1.00
      0.1612777588D+00       1.0000000

6-311G* of H (CP2K TZVP level)
H     0
S    3   1.00
      33.86500               0.0254938
      5.094790               0.190373
      1.158790               0.852161
S    1   1.00
      0.325840               1.000000
S    1   1.00
      0.102741               1.000000
"""
class GTORadials:
    """
    A general introduction to the Gaussian Type Orbitals (GTOs) can be found in the Wikipedia:
    https://en.wikipedia.org/wiki/Gaussian_orbital
    A specific introduction to the GTOs in the Gaussian format can be found here:
    http://sobereva.com/60

    In following code will use the typical notation like:
    c for contraction coefficient, a for exponent, l for angular momentum, r for grid points.
    """
    NumericalRadials = None # list of the radial for each type. 
    # indexed by [it][l][ic][ig] to get (a, c) of primitive GTO, 
    # it: _ of type
    # l: angular momentum
    # ic: _ of contracted GTOs of one angular momentum
    # ig: _ of primitive GTOs of one contracted GTOs
    # instead of what in ABACUS the [it][l][ichi][r]!!!
    symbols = None
    # as it is, the list of symbols for each type

    def __init__(self, fgto: str = None) -> None:
        """construct a GTORadials instance, initialize the value of NumericalRadials
        and symbols. If fgto is provided, read the GTOs from the file."""
        self.NumericalRadials = []
        self.symbols = []
        if fgto is not None:
            self.init_from_file(fgto)

    def init_from_file(self, fgto):
        """write the GTORadials from a file, default behavior is to overwrite the
        existing GTORadials"""
        with open(fgto, "r") as f:
            data = f.read()
        self.symbols, self.NumericalRadials = GTORadials._cgto_parse(data)

    def register_cgto(self, a, c, l, elem = None, mode = 'a'):
        """
        add one CGTO to the GTORadials instance, for a given l
        """
        assert mode in ['a', 'w'], f"Invalid mode: {mode}"
        assert len(c) == len(a), f"Invalid basis: {c}, {a}"
        
        # find correct atom index it
        it = self.symbols.index(elem) if elem in self.symbols else None
        if it is None:
            it = len(self.symbols)
            self.symbols.append(elem)
            self.NumericalRadials.append([])
        if len(self.NumericalRadials[it]) <= l:
            # the case there is not enough l for present type
            self.NumericalRadials[it] += [[] for i in range(l - len(self.NumericalRadials[it]) + 1)]
        if mode == 'w':
            self.NumericalRadials[it][l] = []
        # then append as a new CGTO, convert tuple[list[float], list[float]] to list[tuple[float, float]]
        self.NumericalRadials[it][l].append([(a_, c_) for a_, c_ in zip(a, c)])
      
    def build(self, rgrid, normalize = True):
        """map all the radial functions for each l and cgto onto grid
        
        Args:
            rgrid: numpy array, the grid points
            normalize: bool, whether to normalize the GTOs
        
        Return:
            list of list of numpy arrays, the mapped radial functions, indexed by [it][l][ic][r] to get grid value
        """
        import numpy as np
        ntype = len(self.NumericalRadials)
        assert ntype == len(self.symbols)
        out = [[] for i in range(ntype)] # the output, indexed by [it][l][ic][r] to get grid value
        for it in range(ntype):
            lmax = len(self.NumericalRadials[it]) - 1
            out[it] = [[] for i in range(lmax+1)]
            for l in range(lmax+1):
                for i in range(len(self.NumericalRadials[it][l])): # for each CGTO
                    cgto = np.zeros_like(rgrid)
                    # print(self.NumericalRadials[it][l][i])
                    for a, c in self.NumericalRadials[it][l][i]: # for each primitive GTO
                        cgto += GTORadials._build_gto(a, c, l, rgrid, normalize)
                    out[it][l].append(cgto)
        return out

    def _cgto_parse(data):
        """
        Parse the Contracted Gaussian basis set in the Gaussian format.
        Can be downloaded from Basis Set Exchange: https://www.basissetexchange.org/
        Choose the output format as Gaussian.

        Args:
            basis: list of strings, each string contains the information of a GTO basis function:
        ```plaintext
        H     0
        S    3   1.00
            0.1873113696D+02       0.3349460434D-01
            0.2825394365D+01       0.2347269535D+00
            0.6401216923D+00       0.8137573261D+00
        S    1   1.00
            0.1612777588D+00       1.0000000
        ...
        ****
        C     0
        S    6   1.00
        ...
        ```
        Return:
            out[it][l][ic][ig] = (a, c): the exponential and contraction coefficients of the primitive GTO
            it: the index of the type
            l: the angular momentum
            ic: the index of the contracted GTOs
            ig: the index of the primitive GTOs in the contracted GTOs
            c: coefficient of primitive GTO
            a: exponent of primitive GTO
        """
        import re
        import numpy as np
        spectra = ["S", "P", "D", "F", "G", "H"] # no more...
        data = [d.strip() for d in data.split("****")] # the separator of different elements
        data = [d for d in data if d] # remove empty data
        nelem = len(data)
        out = [[[ # CGTO, because the number is still uncertain, leave as a list
                 ] for j in range(len(spectra))] for i in range(nelem)]
        elems = []
        for d in data: # for each element...
            # wash data
            d = [l.strip() for l in d.split("\n") if not l.startswith("!")] # annotation from Basis Set Exchange
            d = [l for l in d if l]                                         # remove empty lines
            
            elem = None   # should be read, if at last it is still None, abnormal case...
            lmax = 0      # record lmax of the file read
            
            elempat = r"^([A-Z][a-z]?)\s+0$"             # the starting line
            cgtopat = r"^([A-Z]+)\s+(\d+)\s+(\d+\.\d+)$" # the header of one contracted GTOs

            switch = False # switch to read the data
            i = 0          # the line number, for-loop is not used because we read CGTO-by-CGTO instead of line by line
            while i < len(d):
                if re.match(elempat, d[i]):
                    elem = re.match(elempat, d[i]).group(1)
                    switch = True
                elif re.match(cgtopat, d[i]) and switch: # a new CGTO
                    spec_, ngto, _ = re.match(cgtopat, d[i]).groups()
                    l_ = [spectra.index(s_) for s_ in spec_] # the angular momentum of this CGTO, for Pople basis
                                                             # it is possible to be not only one angular momentum
                    lmax = max(lmax, max(l_)) # refresh the maximal angular momentum for this atom type
                    ngto = int(ngto)
                    # then read the coefficients and exponents by ngto lines:
                    ac_ = np.array([re.split(r"\s+", line) for line in d[i+1:i+1+ngto]])
                    a_, c_ = ac_[:, 0], ac_[:, 1:]
                    a_ = [float(a.upper().replace("D", "E")) for a in a_]
                    c_ = [[float(c.upper().replace("D", "E")) for c in ci] for ci in c_] # convert to float
                    for j, l__ in enumerate(l_): # save the GTOs read from the section
                        out[-1][l__].append([(a_[k], c_[k][j]) for k in range(ngto)])
                    i += ngto
                else:
                    print("WARNING! IGNORED LINE:", d[i])
                i += 1
            assert elem is not None, "No symbol found in the file!"
            elems.append(elem)
            # clean the list up to lmax+1
            out[-1] = out[-1][:lmax+1]

        return elems, out

    def _build_gto(a, c, l, r, normalize):
        """build one GTO with coefficients c, exponents a, and grid r"""
        import numpy as np
        g = c * np.exp(-a * r**2) * r**l
        if normalize:
            g /= np.sqrt(np.trapz(g**2, r))
        return g
    
    def __str__(self) -> str:
        """print the GTOs in the Gaussian format. Different CGTO are printed as different section."""
        spectra = ["S", "P", "D", "F", "G", "H"]
        out = ""
        ntype = len(self.symbols)
        for it in range(ntype): # for each type
            out += f"{self.symbols[it]:<2s}     0\n"
            NumericalRadial = self.NumericalRadials[it]
            for l in range(len(NumericalRadial)):
                if len(NumericalRadial[l]) == 0:
                    continue
                ncgto = len(NumericalRadial[l]) # number of contracted GTOs for this l
                for ic in range(ncgto):
                    ngto = len(NumericalRadial[l][ic]) # number of primitive GTOs for this l and ic
                    out += f"{spectra[l]:<2s} {ngto:3d} {1:6.2f}\n"
                    for ig in range(ngto):
                        a, c = NumericalRadial[l][ic][ig]
                        out += f"{a:22.10e} {c:22.10e}\n"
            out += "****\n" if it < ntype - 1 else ""
        return out + "\n"

    def molden(self) -> str:
        """print the GTOs in the Molden format. Different CGTO are printed as different section."""
        spectra = ["s", "p", "d", "f", "g", "h"]
        out = "[GTO]\n"
        ntype = len(self.symbols)
        for it in range(ntype): # for each type
            out += f"{it:>8d}{'0':>8s}\n"
            NumericalRadial = self.NumericalRadials[it]
            for l in range(len(NumericalRadial)):
                for ic in range(len(NumericalRadial[l])):
                    ngto = len(NumericalRadial[l][ic])
                    out += f"{spectra[l]:>25s}{ngto:>8d}{'1.00':>8s}\n"
                    for ig in range(ngto):
                        a, c = NumericalRadial[l][ic][ig]
                        out += f"{a:>62.3f} {c:>12.3f}\n"
            out += "\n"
        return out

def fit_radial_with_gto(nao, ngto, l, r):
    """fit one radial function mapped on grid with GTOs
    
    Args:
        nao: numpy array, the radial function mapped on grid.
        ngto: int, the number of GTOs.
        l: int, the angular momentum.
        r: numpy array, the grid points.
    """
    from scipy.optimize import basinhopping
    import numpy as np
    def f(a_and_c, nao=nao, ngto=ngto, l=l, r=r):
        """calculate the distance between the nao and superposition of GTOs of given
        angular momentum l on user defined grid points r"""
        a, c = a_and_c[:ngto], a_and_c[ngto:]
        assert len(c) == len(a), f"Invalid basis: {c}, {a}"
        gto = np.zeros_like(r)
        for i in range(len(c)):
            gto += GTORadials._build_gto(a[i], c[i], l, r, False)
        dorb = gto - nao
        if l == 0:
            return np.sum(dorb**2)
        else:
            # should exclude the case where the r is almost zero
            while r[0] < 1e-10:
                r = r[1:]
                dorb = dorb[1:]
            return np.sum((dorb/r**l)**2)
    
    init = np.random.rand(ngto + ngto)
    # find optimal c and a values
    
    # bounds for c and a
    bounds = [(0, 5) for i in range(ngto)] + [(-np.inf, np.inf) for i in range(ngto)]
    res = basinhopping(f, init, niter=100, minimizer_kwargs={"method": "L-BFGS-B", "bounds": bounds}, disp=True)
    #res = minimize(f, init, bounds=bounds, method="L-BFGS-B", options={"maxiter": 1000, "disp": True, "ftol": 1e-10})
    a, c = res.x[:ngto], res.x[ngto:]
    err = res.fun
    print(f"""NAO2GTO: Angular momentum {l}, with {ngto} superposition to fit numerical atomic orbitals on given grid, 
this method refers to H. Shang et al. Summary:\nNonlinear fitting error: {err}\nCoefficients and exponents of primitive
Gaussian Type Orbitals (GTOs):\n{"a":>10} {"c":>10}\n---------------------""")
    for i in range(ngto):
        print(f"{a[i]:10.6f} {c[i]:10.6f}")
    return a, c

def convert_nao_to_gto(fnao, fgto = None, ngto: int = 7):
    """convert the numerical atomic orbitals to GTOs. Each chi (or say the zeta function)
    corresponds to a CGTO (contracted GTO), and the GTOs are fitted to the radial functions.
    Which also means during the SCF, the coefficient inside each CGTO is unchanged, while the
    coefficients of CGTO will be optimized."""
    from SIAB.spillage.orbio import read_nao
    import matplotlib.pyplot as plt
    import numpy as np

    gto = GTORadials()
    # read the numerical atomic orbitals
    nao = read_nao(fnao)
    rgrid = np.linspace(0, nao["rcut"], nao["nr"])
    symbol = nao["elem"]
    # fit the radial functions with GTOs
    lmax = len(nao["chi"]) - 1
    for l in range(lmax+1):
        nchi = len(nao["chi"][l])
        for i in range(nchi):
            a, c = fit_radial_with_gto(nao["chi"][l][i], ngto, l, rgrid)
            gto.register_cgto(a, c, l, symbol, 'a')
    
    # draw the fitted GTOs
    out = gto.build(rgrid)
    for it in range(len(out)):
        for l in range(len(out[it])):
            for ic in range(len(out[it][l])):
                plt.plot(rgrid, out[it][l][ic], label=f"element {symbol}, l={l}, ic={ic}")
    plt.legend()
    plt.savefig(fnao.replace(".orb", ".gto.png"))
    plt.close()

    fgto = fnao.replace(".orb", ".gto") if fgto is None else fgto
    with open(fgto, "w") as f:
        f.write(str(gto))

    return gto

import unittest
class TestNAO2GTO(unittest.TestCase):
    def test_cgto_parse(self):
        data = """
Li     0
S    6   1.00
      0.6424189150D+03       0.2142607810D-02
      0.9679851530D+02       0.1620887150D-01
      0.2209112120D+02       0.7731557250D-01
      0.6201070250D+01       0.2457860520D+00
      0.1935117680D+01       0.4701890040D+00
      0.6367357890D+00       0.3454708450D+00
SP   3   1.00
      0.2324918408D+01      -0.3509174574D-01       0.8941508043D-02
      0.6324303556D+00      -0.1912328431D+00       0.1410094640D+00
      0.7905343475D-01       0.1083987795D+01       0.9453636953D+00
SP   1   1.00
      0.3596197175D-01       0.1000000000D+01       0.1000000000D+01
SP   1   1.00
      0.7400000000D-02       0.1000000000D+01       0.1000000000D+01
"""
        symbols, cgtos = GTORadials._cgto_parse(data)
        self.assertEqual(symbols, ["Li"])
        self.assertEqual(len(cgtos), 1) # only one type
        self.assertEqual(len(cgtos[0]), 2) # s and p
        self.assertEqual(len(cgtos[0][0]), 4) # 4 CGTOs for s
        self.assertEqual(len(cgtos[0][0][0]), 6) # 6 primitive GTOs for the first CGTO of s
        self.assertEqual(len(cgtos[0][0][1]), 3) # 3 primitive GTOs for the second CGTO of s
        self.assertEqual(len(cgtos[0][0][2]), 1) # 1 primitive GTOs for the third CGTO of s
        self.assertEqual(len(cgtos[0][0][3]), 1) # 1 primitive GTOs for the fourth CGTO of s
        # thus it is 6-311G basis for Li
        self.assertEqual(len(cgtos[0][1]), 3) # 2 CGTOs for p
        self.assertEqual(len(cgtos[0][1][0]), 3) # 3 primitive GTOs for the first CGTO of p
        self.assertEqual(len(cgtos[0][1][1]), 1) # 1 primitive GTOs for the second CGTO of p
        self.assertEqual(len(cgtos[0][1][2]), 1) # 1 primitive GTOs for the third CGTO of p

    def test_build(self):
        import numpy as np
        import uuid, os
        data = """
Ca     0
S    6   1.00
 202699.                     0.000222964
  30382.5                    0.00172932
   6915.08                   0.00900226
   1959.02                   0.0366699
    640.936                  0.119410
    233.977                  0.291825
S    2   1.00
     92.2892                 0.404415
     37.2545                 0.296313
S    1   1.00
      9.13198                1.000000
S    1   1.00
      3.81779                1.000000
S    1   1.00
      1.04935                1.000000
S    1   1.00
      0.428660               1.000000
S    1   1.00
      0.0628226              1.000000
S    1   1.00
      0.0260162              1.000000
P    3   1.00
   1019.76                   0.00205986
    241.596                  0.01665010
     77.6370                 0.07776460
P    3   1.00
     29.1154                 0.241806
     11.7626                 0.432578
      4.92289                0.367325
P    1   1.00
      1.90645                1.000000
P    1   1.00
      0.73690                1.000000
P    1   1.00
      0.27642                1.000000
P    1   1.00
      0.06027                1.000000
P    1   1.00
      0.01791                1.000000
D    3   1.00
     15.08                   0.0368947
      3.926                  0.1778200
      1.233                  0.4255130
D    1   1.00
      0.260000               1.000000
"""
        fgto = f"test_{uuid.uuid4()}.gto"
        with open(fgto, "w") as f:
            f.write(data)
        gto_obj = GTORadials(fgto)
        os.remove(fgto) # remove the temporary file
        ngrid = 100
        dr = 0.1
        rgrid = np.linspace(0, ngrid*dr, ngrid) # the evenly spaced grid points, the simplest case
        gto = gto_obj.build(rgrid)

        self.assertEqual(len(gto), 1) # only one type
        self.assertEqual(len(gto[0]), 3) # s, p, d
        self.assertEqual(len(gto[0][0]), 8) # 8 CGTOs for s
        ncgto = len(gto[0][0])
        for ic in range(ncgto):
            self.assertEqual(len(gto[0][0][ic]), ngrid)
        self.assertEqual(len(gto[0][1]), 7) # 7 CGTOs for p
        ncgto = len(gto[0][1])
        for ic in range(ncgto):
            self.assertEqual(len(gto[0][1][ic]), ngrid)
        self.assertEqual(len(gto[0][2]), 2) # 2 CGTOs for d
        ncgto = len(gto[0][2])
        for ic in range(ncgto):
            self.assertEqual(len(gto[0][2][ic]), ngrid)
    
    def test_inbuilt_str_method(self):
        import numpy as np
        import uuid, os
        data = """
Ca     0
S    6   1.00
 202699.                     0.000222964
  30382.5                    0.00172932
   6915.08                   0.00900226
   1959.02                   0.0366699
    640.936                  0.119410
    233.977                  0.291825
S    2   1.00
     92.2892                 0.404415
     37.2545                 0.296313
S    1   1.00
      9.13198                1.000000
S    1   1.00
      3.81779                1.000000
S    1   1.00
      1.04935                1.000000
S    1   1.00
      0.428660               1.000000
S    1   1.00
      0.0628226              1.000000
S    1   1.00
      0.0260162              1.000000
P    3   1.00
   1019.76                   0.00205986
    241.596                  0.01665010
     77.6370                 0.07776460
P    3   1.00
     29.1154                 0.241806
     11.7626                 0.432578
      4.92289                0.367325
P    1   1.00
      1.90645                1.000000
P    1   1.00
      0.73690                1.000000
P    1   1.00
      0.27642                1.000000
P    1   1.00
      0.06027                1.000000
P    1   1.00
      0.01791                1.000000
D    3   1.00
     15.08                   0.0368947
      3.926                  0.1778200
      1.233                  0.4255130
D    1   1.00
      0.260000               1.000000
"""
        fgto = f"test_{uuid.uuid4()}.gto"
        with open(fgto, "w") as f:
            f.write(data)
        gto_obj = GTORadials(fgto)
        os.remove(fgto) # remove the temporary file
        # will return
        """Ca     0
S    6   1.00
      2.0269900000e+05       2.2296400000e-04
      3.0382500000e+04       1.7293200000e-03
      6.9150800000e+03       9.0022600000e-03
      1.9590200000e+03       3.6669900000e-02
      6.4093600000e+02       1.1941000000e-01
      2.3397700000e+02       2.9182500000e-01
S    2   1.00
      9.2289200000e+01       4.0441500000e-01
      3.7254500000e+01       2.9631300000e-01
S    1   1.00
      9.1319800000e+00       1.0000000000e+00
S    1   1.00
      3.8177900000e+00       1.0000000000e+00
S    1   1.00
      1.0493500000e+00       1.0000000000e+00
S    1   1.00
      4.2866000000e-01       1.0000000000e+00
S    1   1.00
      6.2822600000e-02       1.0000000000e+00
S    1   1.00
      2.6016200000e-02       1.0000000000e+00
P    3   1.00
      1.0197600000e+03       2.0598600000e-03
      2.4159600000e+02       1.6650100000e-02
      7.7637000000e+01       7.7764600000e-02
P    3   1.00
      2.9115400000e+01       2.4180600000e-01
      1.1762600000e+01       4.3257800000e-01
      4.9228900000e+00       3.6732500000e-01
P    1   1.00
      1.9064500000e+00       1.0000000000e+00
P    1   1.00
      7.3690000000e-01       1.0000000000e+00
P    1   1.00
      2.7642000000e-01       1.0000000000e+00
P    1   1.00
      6.0270000000e-02       1.0000000000e+00
P    1   1.00
      1.7910000000e-02       1.0000000000e+00
D    3   1.00
      1.5080000000e+01       3.6894700000e-02
      3.9260000000e+00       1.7782000000e-01
      1.2330000000e+00       4.2551300000e-01
D    1   1.00
      2.6000000000e-01       1.0000000000e+00
"""
        gto_obj.register_cgto([1, 2, 3], [0.1, 0.2, 0.3], 1, 'Arbitrary', 'a')

    def test_cgto_molden(self):
        import numpy as np
        import uuid, os
        data = """
Li     0
S    6   1.00
      0.6424189150D+03       0.2142607810D-02
      0.9679851530D+02       0.1620887150D-01
      0.2209112120D+02       0.7731557250D-01
      0.6201070250D+01       0.2457860520D+00
      0.1935117680D+01       0.4701890040D+00
      0.6367357890D+00       0.3454708450D+00
SP   3   1.00
      0.2324918408D+01      -0.3509174574D-01       0.8941508043D-02
      0.6324303556D+00      -0.1912328431D+00       0.1410094640D+00
      0.7905343475D-01       0.1083987795D+01       0.9453636953D+00
SP   1   1.00
      0.3596197175D-01       0.1000000000D+01       0.1000000000D+01
SP   1   1.00
      0.7400000000D-02       0.1000000000D+01       0.1000000000D+01
"""
        fgto = f"test_{uuid.uuid4()}.gto"
        with open(fgto, "w") as f:
            f.write(data)
        gto_obj = GTORadials(fgto)
        os.remove(fgto) # remove the temporary file
        out = gto_obj.molden()
        print(out)

    def est_fit_radial_with_gto(self):
        from SIAB.spillage.orbio import read_nao
        import numpy as np

        # read the numerical atomic orbitals
        nao = read_nao("SIAB/spillage/testfiles/In_gga_10au_100Ry_3s3p3d2f.orb")
        rgrid = np.linspace(0, nao["rcut"], nao["nr"])
        l = 0
        ngto = 7

        chi = nao["chi"][l][2]
        a, c = fit_radial_with_gto(chi, ngto, l, rgrid)
        # the fitted GTOs
        gto = np.zeros_like(rgrid)
        for a_, c_ in zip(a, c):
            gto += GTORadials._build_gto(a_, c_, l, rgrid, False)
        
        import matplotlib.pyplot as plt
        plt.plot(rgrid, chi, label="NAO")
        plt.plot(rgrid, gto, label="GTO")
        plt.axhline(0, color="black", linestyle="--")
        plt.legend()
        plt.savefig("nao2gto.png")
        plt.close()

if __name__ == "__main__":
    unittest.main(exit=True)
    convert_nao_to_gto("SIAB/spillage/testfiles/In_gga_10au_100Ry_3s3p3d2f.orb", "test.gto", 7)
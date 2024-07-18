
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

    NumericalRadials = None # list of the radial for each type. 
    # indexed by [it][l][i] to get (c, a), the c is coefficient of primitive GTO, a is the exponent.
    # instead of what in ABACUS the [it][l][ichi][r]!!!
    symbol = None
    def __init__(self, fgto: str = None) -> None:
        if fgto is not None:
            self.init_from_file(fgto)

    def init_from_file(self, fgto):
        with open(fgto, "r") as f:
            data = f.read()
        self.symbol, self.NumericalRadial = GTORadials._read_gto(data)

    def set_from_list(self, c, a, l, mode = 'a', symbol = None):
        """
        set the GTOs from a list of coefficients and exponents
        Args:
            c: list, the coefficients of GTOs
            a: list, the exponents of GTOs
            l: int, the angular momentum
            mode: str, the mode to set the GTOs, 'a' for append, 'w' for
            overwrite.
        
        Return:
            None
        """
        assert mode in ['a', 'w'], f"Invalid mode: {mode}"
        assert len(c) == len(a), f"Invalid basis: {c}, {a}"
        assert isinstance(l, int), f"Invalid angular momentum: {l}"
        if mode == 'w' or self.NumericalRadial is None:
            self.NumericalRadial = [[] * (l+1)]
        if len(self.NumericalRadial) < l+1:
            self.NumericalRadial.extend([[] for i in range(l+1 - len(self.NumericalRadial))])
        self.NumericalRadial[l].append(list(zip(c, a)))
        assert symbol or self.symbol, "No symbol provided!"
        self.symbol = symbol if symbol else self.symbol
        
    def build(self, rgrid, normalize = True):
        ntype = len(self.NumericalRadial)
        out = [[] for i in range(ntype)]

        for it in range(ntype):
            lmax = len(self.NumericalRadial[it]) - 1
            out[it] = [[] for i in range(lmax+1)]
            for l in range(lmax+1):
                for i in range(len(self.NumericalRadial[it][l])):
                    c, a = self.NumericalRadial[it][l][i]
                    out[it][l].append(GTORadials._gto(c, a, l, rgrid, normalize))
        return out

    def map_on_grid(self, rgrid, normalize = True):
        """map all the radial functions for each l, superposition of GTOs
        
        Args:
            rgrid: numpy array, the grid points
            normalize: bool, whether to normalize the GTOs
        
        Return:
            list of numpy arrays, the mapped radial functions, indexed by l
        """
        import numpy as np
        ntype = len(self.NumericalRadial)
        out = [[] for i in range(ntype)]
        for it in range(ntype):
            lmax = len(self.NumericalRadial[it]) - 1
            out[it] = [np.zeros_like(rgrid) for i in range(lmax+1)]
            for l in range(lmax+1):
                for i in range(len(self.NumericalRadial[it][l])):
                    c, a = self.NumericalRadial[it][l][i]
                    out[it][l] += GTORadials._gto(c, a, l, rgrid, normalize)
        return out

    def _read_gto(data):
        """
        Parse the Gaussian basis set in the Gaussian format.

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
        ```
        Return:
            nested list, first index is type and will forever be [0], the second is angular momentum, and the third is the GTOs.
            out[0][l][i]: [coefficients, exponents], the coefficients and exponents of the i-th GTO with angular momentum l.
        """
        import re
        import numpy as np

        data = [l.strip() for l in data.split("\n") if not l.startswith("!")] # annotation from Basis Set Exchange
        data = [l for l in data if l] # remove empty lines
        spectrum = ["s", "p", "d", "f", "g", "h"] # no more...
        out = [[] for i in range(len(spectrum))] # first allocate a larger space, then will delete the last few empty lists
        symbol = None
        i = 0 # the line number
        # the starting line
        startpat = r"^([A-Z][a-z]?)\s+0$"
        switch = False
        # the header of one contracted GTOs
        headerpat = r"^([A-Z]+)\s+(\d+)\s+(\d+\.\d+)$"
        # record lmax
        lmax = 0
        while i < len(data): # read the data line by line
            if re.match(startpat, data[i]):
                symbol = re.match(startpat, data[i]).group(1)
                switch = True
            elif re.match(headerpat, data[i]) and switch:
                spec_, n_, _ = re.match(headerpat, data[i]).groups()
                l_ = [spectrum.index(s_.lower()) for s_ in spec_]
                lmax = max(lmax, max(l_))
                n_ = int(n_)
                # then read the coefficients and exponents by n_ lines:
                ca_ = np.array([re.split(r"\s+", line) for line in data[i+1:i+1+n_]])
                c_, a_ = ca_[:, :-1].T, ca_[:, -1]
                a_ = [float(a.upper().replace("D", "E")) for a in a_]
                for j, l__ in enumerate(l_): # save the GTOs read from the section
                    out[l__].extend([[float(c_[j][k].upper().replace("D", "E")), float(a_[k])] for k in range(n_)])
                i += n_
            else:
                print("WARNING! IGNORED LINE:", data[i])
            i += 1
        return symbol, [out[:lmax+1]]

    def _gto(c, a, l, r, normalize):
        """build one GTO with coefficients c, exponents a, and grid r"""
        import numpy as np
        g = c * np.exp(-a * r**2) * r**l
        if normalize:
            g /= np.sqrt(np.trapz(g**2, r))
        return g
    
    def __str__(self) -> str:
        """print the GTOs in the Gaussian format"""
        spectrum = ["s", "p", "d", "f", "g", "h"]
        out = f"{self.symbol}     0\n"
        for l in range(len(self.NumericalRadial[0])):
            if len(self.NumericalRadial[0][l]) == 0:
                continue
            out += f"{spectrum[l].upper():<2s} {len(self.NumericalRadial[0][l]):3d} {1:6.2f}\n"
            for c, a in self.NumericalRadial[0][l]:
                out += f"{a:22.10e} {c:22.10e}\n"
        return out

def fit_nao_with_gto(nao, ngto, l, r):
    """fit one radial function mapped on grid with GTOs
    
    Args:
        nao: numpy array, the radial function mapped on grid.
        ngto: int, the number of GTOs.
        l: int, the angular momentum.
        r: numpy array, the grid points.
    """
    from scipy.optimize import basinhopping, minimize
    import numpy as np
    def f(c_and_a, nao=nao, ngto=ngto, l=l, r=r):
        """calculate the distance between the nao and superposition of GTOs of given
        angular momentum l on user defined grid points r"""
        c, a = c_and_a[:ngto], c_and_a[ngto:]
        assert len(c) == len(a), f"Invalid basis: {c}, {a}"
        gto = np.zeros_like(r)
        for i in range(len(c)):
            gto += GTORadials._gto(c[i], a[i], l, r, False)
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
    bounds = [(-2, 2) for i in range(ngto)] + [(0, 5) for i in range(ngto)]
    res = basinhopping(f, init, niter=100, minimizer_kwargs={"method": "L-BFGS-B", "bounds": bounds}, disp=True)
    #res = minimize(f, init, bounds=bounds, method="L-BFGS-B", options={"maxiter": 1000, "disp": True, "ftol": 1e-10})
    c, a = res.x[:ngto], res.x[ngto:]
    err = res.fun
    print(f"""NAO2GTO: Angular momentum {l}, with {ngto} superposition to fit numerical atomic orbitals on given grid, 
this method refers to H. Shang et al. Summary:\nNonlinear fitting error: {err}\nCoefficients and exponents of primitive
Gaussian Type Orbitals (GTOs):\n{"c":>10} {"a":>10}\n---------------------""")
    for i in range(ngto):
        print(f"{c[i]:10.6f} {a[i]:10.6f}")
    return c, a

def convert_nao_to_gto(fnao, fgto, ngto: int = 7):
    from SIAB.spillage.orbio import read_nao
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
            c, a = fit_nao_with_gto(nao["chi"][l][i], ngto, l, rgrid)
            gto.set_from_list(c, a, l, 'a', symbol)
    with open(fgto, "w") as f:
        f.write(str(gto))

def write_molden(labels, coords, labels_kinds_map, kinds):
    from SIAB.data.interface import PERIODIC_TABLE_TOINDEX
    info = "Writing Molden format...\n"
    info += "Note: if pseudopotential is used, remember to add a section called [zval]\n"
    info += "like:\n[zval]\nH 1.0\nO 6.0\n...\n"
    print(info, flush=True)
    out = "[Molden Format]\n"
    out += "[Atoms] (Angstrom)\n"
    for i, label in enumerate(labels):
        x, y, z = coords[i]
        out += f"{label} {i+1} {PERIODIC_TABLE_TOINDEX[kinds[labels_kinds_map[i]]]} "
        out += f"{x:20.10f}{y:20.10f}{z:20.10f}\n"
    out += "[GTO]\n"
    return out

import unittest
class TestNAO2GTO(unittest.TestCase):
    def test_read_gto(self):
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
        symbol, basis = GTORadials._read_gto(data)
        self.assertEqual(symbol, "Li")
        self.assertEqual(len(basis), 1) # only one type
        self.assertEqual(len(basis[0]), 2) # s and p
        self.assertEqual(len(basis[0][0]), 6 + 3 + 1 + 1) # 6 s, 3 sp, 1 sp, 1 sp
        self.assertEqual(len(basis[0][1]), 3 + 1 + 1) # 3 sp, 1 sp, 1 sp
    
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
        self.assertEqual(len(gto[0][0]), 14)
        self.assertEqual(len(gto[0][1]), 11)
        self.assertEqual(len(gto[0][2]), 4)
        for i in range(14):
            self.assertEqual(len(gto[0][0][i]), ngrid)
        for i in range(11):
            self.assertEqual(len(gto[0][1][i]), ngrid)
        for i in range(4):
            self.assertEqual(len(gto[0][2][i]), ngrid)

    def test_set_from_list(self):
        gto = GTORadials()
        c = [1, 2, 3]
        a = [0.1, 0.2, 0.3]
        l = 1
        gto.set_from_list(c, a, l, 'w', "H")
        self.assertEqual(len(gto.NumericalRadial), 2)
        self.assertEqual(len(gto.NumericalRadial[1]), 1)
        self.assertEqual(len(gto.NumericalRadial[1][0]), 3)
        for i in range(3):
            self.assertEqual(gto.NumericalRadial[1][0][i], (c[i], a[i]))
        c = [4, 5, 6]
        a = [0.4, 0.5, 0.6]
        gto.set_from_list(c, a, l, 'a')
        self.assertEqual(len(gto.NumericalRadial), 2)
        self.assertEqual(len(gto.NumericalRadial[1]), 2)
        self.assertEqual(len(gto.NumericalRadial[1][0]), 3)
        self.assertEqual(len(gto.NumericalRadial[1][1]), 3)
        for i in range(3):
            self.assertEqual(gto.NumericalRadial[1][1][i], (c[i], a[i]))
        c = [7, 8, 9]
        a = [0.7, 0.8, 0.9]
        l = 0
        gto.set_from_list(c, a, l, 'a')
        self.assertEqual(len(gto.NumericalRadial), 2)
        self.assertEqual(len(gto.NumericalRadial[0]), 1)
        self.assertEqual(len(gto.NumericalRadial[0][0]), 3)
        for i in range(3):
            self.assertEqual(gto.NumericalRadial[0][0][i], (c[i], a[i]))
        c = [4, 5, 6] # are they still there?
        a = [0.4, 0.5, 0.6]
        for i in range(3):
            self.assertEqual(gto.NumericalRadial[1][1][i], (c[i], a[i]))
        # test overwrite
        c = [10, 11, 12]
        a = [1.0, 1.1, 1.2]
        l = 0
        gto.set_from_list(c, a, l, 'w')
        self.assertEqual(len(gto.NumericalRadial), 1)
        self.assertEqual(len(gto.NumericalRadial[0]), 1)
        self.assertEqual(len(gto.NumericalRadial[0][0]), 3)
        for i in range(3):
            self.assertEqual(gto.NumericalRadial[0][0][i], (c[i], a[i]))

    def test_map_on_grid(self):
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
        gto = gto_obj.map_on_grid(np.linspace(0, 100, 100))
        self.assertEqual(len(gto), 1)
        for it in range(len(gto)):
            self.assertEqual(len(gto[it]), 3)
            for l in range(len(gto[it])):
                self.assertEqual(len(gto[it][l]), 100)
    
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
        print(gto_obj)
        # will return
        """
Ca     0
S   14   1.00
      2.2296400000e-04       2.0269900000e+05
      1.7293200000e-03       3.0382500000e+04
      9.0022600000e-03       6.9150800000e+03
      3.6669900000e-02       1.9590200000e+03
      1.1941000000e-01       6.4093600000e+02
      2.9182500000e-01       2.3397700000e+02
      4.0441500000e-01       9.2289200000e+01
      2.9631300000e-01       3.7254500000e+01
      1.0000000000e+00       9.1319800000e+00
      1.0000000000e+00       3.8177900000e+00
      1.0000000000e+00       1.0493500000e+00
      1.0000000000e+00       4.2866000000e-01
      1.0000000000e+00       6.2822600000e-02
      1.0000000000e+00       2.6016200000e-02
P   11   1.00
      2.0598600000e-03       1.0197600000e+03
      1.6650100000e-02       2.4159600000e+02
      7.7764600000e-02       7.7637000000e+01
      2.4180600000e-01       2.9115400000e+01
      4.3257800000e-01       1.1762600000e+01
      3.6732500000e-01       4.9228900000e+00
      1.0000000000e+00       1.9064500000e+00
      1.0000000000e+00       7.3690000000e-01
      1.0000000000e+00       2.7642000000e-01
      1.0000000000e+00       6.0270000000e-02
      1.0000000000e+00       1.7910000000e-02
D    4   1.00
      3.6894700000e-02       1.5080000000e+01
      1.7782000000e-01       3.9260000000e+00
      4.2551300000e-01       1.2330000000e+00
      1.0000000000e+00       2.6000000000e-01
      """

        gto_obj.set_from_list([1, 2, 3], [0.1, 0.2, 0.3], 1, 'a')
        print(gto_obj)

    def est_fit_nao_with_gto(self):
        from SIAB.spillage.orbio import read_nao
        import numpy as np

        # read the numerical atomic orbitals
        nao = read_nao("SIAB/spillage/testfiles/In_gga_10au_100Ry_3s3p3d2f.orb")
        rgrid = np.linspace(0, nao["rcut"], nao["nr"])
        l = 1
        ngto = 7

        chi = nao["chi"][l][1]
        c, a = fit_nao_with_gto(chi, ngto, l, rgrid)
        # the fitted GTOs
        gto = np.zeros_like(rgrid)
        for i in range(len(c)):
            gto += GTORadials._gto(c[i], a[i], l, rgrid, False)
        exit()
        import matplotlib.pyplot as plt
        plt.plot(rgrid, chi, label="NAO")
        plt.plot(rgrid, gto, label="GTO")
        plt.axhline(0, color="black", linestyle="--")
        plt.legend()
        plt.savefig("nao2gto.png")

if __name__ == "__main__":
    unittest.main(exit=True)
    convert_nao_to_gto("SIAB/spillage/testfiles/In_gga_10au_100Ry_3s3p3d2f.orb", "test.gto", 7)
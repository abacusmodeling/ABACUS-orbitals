"""basic"""
import re
def is_numeric_data(data):
    """judge if the data line is full of numbers (including scientific notation) separated by spaces, tabs or newlines"""
    if re.match(r"^\s*[+-]?\d+(\.\d*)?([eE][+-]?\d+)?(\s+[+-]?\d+(\.\d*)?([eE][+-]?\d+)?)*\s*$", data):
        return True
    else:
        return False

def decompose_data(data):
    """to decompose all numbers in one line, but need to judge whether int or float"""
    if re.match(r"^\s*[+-]?\d+(\.\d*)([eE][+-]?\d+)?(\s+[+-]?\d+(\.\d*)([eE][+-]?\d+)?)*\s*$", data):
        return [float(x) for x in data.split()] if " " in data else float(data.strip())
    elif re.match(r"^\s*[+-]?\d+(\s+[+-]?\d+)*\s*$", data):
        return [int(x) for x in data.split()] if " " in data else int(data.strip())
    else:
        raise ValueError("data is not numeric")

def orbconf_fromxzyp(zeta_notation: str, 
                     minimal_basis: list = None,
                     as_list: bool = False):
    """convert zeta notation to orbital configuration
    
    zeta_notation: str, like "DZP", "TZDP", "TZ5P"
    minimal_basis: list, can be [ns, np, nd, nf, ...] or the one directly grep from
    pseudopotential: [["4S"], ["3P", "4P"], ["3D", "4D"], ["4F"]]
    as_list: bool, if True, return the orbital configuration as a list, otherwise
    return a string. 
    For example if as_list is True, then "DZP" with minimal_basis [1, 1] will return
    [2, 2, 1].

    NOTE:
    There is a special case, in minimal_basis there might one (even more) list(s) empty,
    for example the PSlibrary Norm-Conserving 0.3.1 Mn pseudopotential, the minimal_basis
    is [["4S"], [], ["3D"]], 3p electrons are pseudized. Then if zeta_notation is "TZ5P", 
    the result will be [3, 0, 3, 5].
    However, p orbital is directly important when forming covalent bond, say for MnO4-, 
    it is sp3 hybridization. But f may also be the one not to be neglected.
    Therefore, multiplicity of P in this case will be added to all l with 0 electron. Say
    if zeta_notation is "TZ5P", the result will be [3, 5, 3], string as "3s3d5p".
    """
    is_numeric = True
    for layer in minimal_basis:
        if isinstance(layer, list):
            is_numeric = False
            break
    if not is_numeric:
        minimal_basis = [len(layer) for layer in minimal_basis]
    pattern = r"([SDTQ5-9]?Z)(([SDTQ5-9]?P)*)"
    symbols = ["s", "p", "d", "f", "g", "h", "i", "k", "l", "m", "n", "o"]
    multiplier = {"S": 1, "D": 2, "T": 3, "Q": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9}
    _match = re.match(pattern, zeta_notation)
    if _match is None:
        raise ValueError(f"zeta_notation is not valid: {zeta_notation}")
    nzeta = multiplier[_match.group(1)[0]] if len(_match.group(1)) > 1 else 1
    basis = [nzeta*i for i in minimal_basis]
    result = "".join([f"{basis[i]}{symbols[i]}" for i in range(len(minimal_basis)) if basis[i] != 0])
    # Polarization
    if _match.group(2) is not None:
        if len(_match.group(2)) > 1:
            nzeta_p = list(map(int, [multiplier[x] if x != "" else 1 for x in _match.group(2).split("P")[:-1]]))
            # case 1: no 0 in minimal_basis, means each l has at least one electron
            if 0 not in minimal_basis:
                result += "".join([f"{nzeta_p[i]}{symbols[len(minimal_basis) + i]}" for i in range(len(nzeta_p))])
                basis.extend(nzeta_p)
                #result += str(multiplier[_match.group(2)[0]]) + symbols[len(minimal_basis)]
                #basis.append(multiplier[_match.group(2)[0]])
            # case 2: 0 in minimal_basis, means some l has no electron
            # in this case, polarization is added to all l with 0 electron
            else:
                for l in range(len(minimal_basis)):
                    if basis[l] == 0: # if l has no electron
                        result += f"{nzeta_p[0]}{symbols[l]}"
                        basis[l] = nzeta_p[0]
                        nzeta_p.pop(0)
                        if len(nzeta_p) == 0:
                            break
                result += "".join([f"{nzeta_p[i]}{symbols[len(minimal_basis) + i]}" for i in range(len(nzeta_p))])
                basis.extend(nzeta_p)
        else:
            # case 1: no 0 in minimal_basis, means each l has at least one electron
            if 0 not in minimal_basis:
                result += "1" + symbols[len(minimal_basis)]
                basis.append(1)
            # case 2: 0 in minimal_basis, means some l has no electron
            # in this case, polarization is added to all l with 0 electron
            else:
                for i in range(len(minimal_basis)):
                    if basis[i] == 0:
                        result += "1" + symbols[i]
                        basis[i] = 1
    if as_list:
        return basis
    else:
        return result
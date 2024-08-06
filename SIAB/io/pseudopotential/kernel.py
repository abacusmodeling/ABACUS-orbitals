import xml.etree.ElementTree as ET
def iter_tree(root: ET.Element):
    """iterate through the tree, return a dictionary flattened from the tree"""
    return {child.tag: {"attrib": child.attrib, "data": child.text} for child in list(root.iter())}

def preprocess(fname: str):
    """ADC pseudopotential has & symbol at the beginning of line, which is not allowed in xml, replace & with &amp;"""
    with open(fname, "r") as f:
        lines = f.readlines()
    """GBRV pseudopotential does not startswith <UPF version="2.0.1">, but <PP_INFO>, 
    add <UPF version="2.0.1"> to the beginning of the file and </UPF> to the end of the file"""
    if not lines[0].startswith("<UPF version="):
        lines.insert(0, "<UPF version=\"2.0.1\">\n")
        lines.append("</UPF>")

    with open(fname, "w") as f:
        for line in lines:
            """if line starts with &, replace & with &amp;, 
            but if already &amp;, do not replace"""
            if line.strip().startswith("&") and not line.strip().startswith("&amp;"):
                line = line.replace("&", "&amp;")
            
            f.write(line)

            if line.strip() == "</UPF>":
                break

import SIAB.io.pseudopotential.tools.basic as siptb
def postprocess(parsed: dict):

    for section in parsed:
        """first the data"""
        if parsed[section]["data"] is not None:
            parsed[section]["data"] = parsed[section]["data"].strip()
            if siptb.is_numeric_data(parsed[section]["data"]):
                parsed[section]["data"] = siptb.decompose_data(parsed[section]["data"])
        """then the attributes"""
        if parsed[section]["attrib"] is not None:
            for attrib in parsed[section]["attrib"]:
                parsed[section]["attrib"][attrib] = parsed[section]["attrib"][attrib].strip()
                if siptb.is_numeric_data(parsed[section]["attrib"][attrib]):
                    parsed[section]["attrib"][attrib] = siptb.decompose_data(parsed[section]["attrib"][attrib])
                elif parsed[section]["attrib"][attrib] == "T":
                    if attrib == "element": continue # fix the bug of element F
                    parsed[section]["attrib"][attrib] = True
                elif parsed[section]["attrib"][attrib] == "F":
                    if attrib == "element": continue # fix the bug of element F
                    parsed[section]["attrib"][attrib] = False
    return parsed

def upf(fname: str):
    """parse the pseudopotential file, return a dictionary"""
    error_msg = """ERROR: UPF file with non-XML format. Please contact with either developer
of pseudopotential you use or the developer of this package. For the latter choice, 
you can submit issue in Github Repository at:
https://github.com/kirk0830/abacus_orbital_generation
, thanks for understanding, raise TypeError and Quit..."""
    preprocess(fname)
    try:
        tree = ET.parse(fname)
    except ET.ParseError:
        print(error_msg, flush=True)
        raise TypeError("ERROR: Please read the error message above.") from None
    root = tree.getroot()
    parsed = iter_tree(root)
    postprocess(parsed)
    return parsed
    
def vwr(fname: str):
    """roughly parse the vwr pseudopotential file, which is used by
    PWmat, ABACUS, siesta(?) etc.
    the vwr pseudopotential file has format like:
    ```plaintext
    611, 0, 105, 0.5, 1, 0.5, 0.0, 0.0  0   ***nrr,ic,iatom,z,spd_loc,occ_s,occ_p,occ_d,iso  (Ecut=20Ryd)
    1  1    1         1    1    0
    .000000E+00 -.145890E+01 -.107086E+01 -.525677E+00  .698986E+00  .000000E+00  .000000E+00
    .808054E-05 -.145890E+01 -.107086E+01 -.525677E+00  .698986E+00  .294843E-06  .669819E-14
    .163243E-04 -.145890E+01 -.107086E+01 -.525677E+00  .698986E+00  .595643E-06  .273367E-13
    .247346E-04 -.145890E+01 -.107086E+01 -.525677E+00  .698986E+00  .902519E-06  .627606E-13
    .333148E-04 -.145890E+01 -.107086E+01 -.525677E+00  .698986E+00  .121559E-05  .113855E-12
    ...
    ```
    in source code of PEtot, can find the annotation of each term: 
    https://github.com/qsnake/petot/blob/master/DOC_FILES/DOC_vwr.atom
    , I just copy here:
    ```plaintext
    nrr,icor,iatom,z,spd_loc,occ_s,occ_p,occ_d,iso   | annotation1
    iref_s,iref_p,iref_d,  iTB_s, iTB_p, iTB_d       | annotation2
    r, v_s, v_p, v_d, w_s, w_p, w_d, [v_loc], [core], [v_(p+1/2)-v_(p-1/2), v_(d+1/2)-v(d-1/2)]
    .....
    .....
    r, v_s, v_p, v_d, w_s, w_p, w_d, [v_loc], [core], [v_(p+1/2)-v_(p-1/2), v_(d+1/2)-v(d-1/2)]
    # final annotation3
    ---------------------------------------------------

    Explanation:
    (1) nrr: the number of lines (following the first two lines)
    (2) icor: core correction. icor=0, no core correction  (the optional [core] 
            does not exist), icore=1, with core correction ([core] exists).
    (3) iatom: the atomic number (znuc). In a PEtot calculation, this number must
            match the atomic number used in xatom.config file. 
    (4) z: the pseudo core charge (i.e., znuc-zcore). 
    (5) spd_loc: the angular momentum +1 for the local potential. 
                I.e, spd_loc=1,2,3 for s,p,d  (=llocal+1 of atomi.input). 
                if spd_loc=0, use the [v_loc] column as the local potential.
    (6) occ_s,occ_p,occ_d: the occupation number for s,p,d wavefunction to generate
                the initial atomic charge for PEtot (same as the value in atom.input).
    (6) iso: spin information. iso=0, no spin coupling (from "n","s", calculation for
                non relativistic and LSDA in atom.input), the [v_(p+1/2)-v_(p-1/2),..]
                does not exist. iso=1, (from "r", relativistic calculation), 
                the [v_(p+1/2)-v_(p-1/2),..] exists, can be used in Escan for 
                spin-orbit coupling calculations.  
    (7) iref_s,iref_p,iref_d: whether or not to evaluation the s,p,d KB projection.
                If iref_l=0, turns that operator off (i.e, when it is the local 
                potential).  
                These are not used in the current PEtot program!, ie, always set iref_l=1
    (8) iTB_s,iTB_p,iTB_d: whether or not to use its atomic orbital for tight-binding
                wavefunction initialization. These numbers, and the TB initialization
                are not used in the current version of PEtot. 
                iTB_l=0, not use this orbital, iTB_l=1, use this orbital. 

    (9) r, v_s, v_p, v_d, w_s, w_p, w_d, [v_loc], [core], [v_(p+1/2)-v_(p-1/2), v_(d+1/2)-v(d-1/2)]:
                r: raduis, in the unit of Bohr. 
                v_s,v_p,v_d: the s,p,d potentials, in the unit of Hartree. 
                These three potentials are always
                there in the file, even if you only used nval=2: s,p, in 
                atom.input. In that case, v_d just equals the v_local, and 
                iref_d=0. 
                Note that, in spin calculation, v_l=(v_l_up+v_l_down)/2. 
                in relativistic calculation: v_l=(v_(l+1/2)+v_(l-1/2))/2.
                w_s,w_p,w_d: the s,p,d wavefunctions [actually psi(r)/sqrt(4pi)]. 
                If w_d is not calculated in atom.input (i.e., nval=2: s,p), then
                w_d=1. 
                [v_loc], the local potential which is different from v_s,v_p,v_d. 
                        Only exists when spd_loc=0.
                [core], the core charge density, only exist when icor=1
                [v_(p+1/2)-v_(p-1/2), v_(d+1/2)-v(d-1/2)]: the potential difference
                for relavistic calculation. Can be used in Escan for spin-orbit 
                coupling calculation. 
    (10) annotation1: obvious. 
    (11) annotation2: contains the information in atom.input, so we can regenerate
                    it if we want. 
    (12) annotation3: obvious. 
    ```
    """
    import re
    from SIAB.data.interface import PERIODIC_TABLE_TOSYMBOL
    with open(fname, "r") as f:
        lines = [l.strip() for l in f.readlines()]
    lines = [l for l in lines if l]
    header = lines[0]
    words = re.split(r"\s|,", header)
    words = [w for w in words if w] # remove empty string
    assert len(words) >= 9, f"header has less than 9 words: {header}"
    nrr, icor, iatom, z, spd_loc, occ_s, occ_p, occ_d, iso = words[:9]
    assert len(lines) == int(nrr) + 2, f"number of lines does not match nrr: {nrr}"

    header = lines[1]
    #iref_s, iref_p, iref_d, iTB_s, iTB_p, iTB_d = header.split()
    sequence = ["S", "P", "D", "F", "G", "H", "I", "K", "L", "M", "N"]
    occ = [float(occ_s), float(occ_p), float(occ_d)]
    val_conf = [[] if occ[i] == 0 else [sequence[i]] for i in range(3)]
    out = {
        "PP_INFO": {
            "attrib": {
                "val_conf": val_conf
            }, 
            "data": "vwr pseudopotential"
        },
        "PP_HEADER": {
            "attrib": {
                "generated": "vwr", "author": "unknown",
                "date": "unknown", "comment": "unknown",
                "element": PERIODIC_TABLE_TOSYMBOL[int(iatom)], "pseudo_type": "NC",
                "relativistic": "scalar",
                "is_ultrasoft": False, "is_paw": False, "is_coulomb": False,
                "has_so": iso == "0", "has_wfc": False, "has_gipaw": False,
                "core_correction": icor == "1",
                "functional": "PBE",
                "z_valence": float(z),
                "total_psenergy": 0.0,
                "rho_cutoff": 0.0,
                "l_max": int(spd_loc) - 1,
                "l_local": -1,
                "mesh_size": int(nrr),
                "number_of_wfc": 0,
                "number_of_proj": 0
            },
            "data": ""
        }
    }
    return out

import unittest
class TestPspotKernel(unittest.TestCase):
    def test_vwr(self):
        import uuid # for temporary file name
        import os # for file operation
        to_parse = """
18, 0, 73,  5.0, 1,  2.00,   .00,  3.00, 1  |nrr,icor,iatom,z,spd_loc, occ_s,occ_p,occ_d,iso (Ecut=60Ryd)
1 1 1   1 1 0   |<-iref_s,p,d,iTB_s,p,d; inform->|  pg tm2 ca r;rc=2.5 2.5 2.5 ;nc,nv=* 3;n,l,od,ou=6 0  2.0   .0:6 1   .0   .0:5 2  3.0   .0:
 .42710773E-06 -.88690740E+00 -.66564168E+01 -.72258995E+01  .82669769E-01  .10968608E-06  .18474565E-12 -.16998869E+01  .80433646E-01
 .85958781E-06 -.88690740E+00 -.66564168E+01 -.72258995E+01  .82669769E-01  .22075184E-06  .74830710E-12 -.16998869E+01  .80433646E-01
 .12975078E-05 -.88690740E+00 -.66564168E+01 -.72258995E+01  .82669769E-01  .33321464E-06  .17049802E-11 -.16998869E+01  .80433646E-01
 .17409362E-05 -.88690740E+00 -.66564168E+01 -.72258995E+01  .82669769E-01  .44709204E-06  .30694819E-11 -.16998869E+01  .80433646E-01
 .21899422E-05 -.88690740E+00 -.66564168E+01 -.72258995E+01  .82669769E-01  .56240185E-06  .48569620E-11 -.16998869E+01  .80433646E-01
 .26445960E-05 -.88690740E+00 -.66564168E+01 -.72258995E+01  .82669769E-01  .67916207E-06  .70830140E-11 -.16998869E+01  .80433646E-01
 .31049687E-05 -.88690740E+00 -.66564168E+01 -.72258995E+01  .82669769E-01  .79739096E-06  .97636876E-11 -.16998869E+01  .80433646E-01
 .35711321E-05 -.88690740E+00 -.66564168E+01 -.72258995E+01  .82669769E-01  .91710698E-06  .12915501E-10 -.16998869E+01  .80433646E-01
 .40431591E-05 -.88690740E+00 -.66564168E+01 -.72258995E+01  .82669769E-01  .10383288E-05  .16555455E-10 -.16998869E+01  .80433646E-01
 .45211235E-05 -.88690740E+00 -.66564168E+01 -.72258995E+01  .82669769E-01  .11610755E-05  .20701041E-10 -.16998869E+01  .80433646E-01
 .50051000E-05 -.88690740E+00 -.66564168E+01 -.72258995E+01  .82669769E-01  .12853661E-05  .25370263E-10 -.16998869E+01  .80433646E-01
 .54951641E-05 -.88690740E+00 -.66564168E+01 -.72258995E+01  .82669769E-01  .14112201E-05  .30581641E-10 -.16998869E+01  .80433646E-01
 .59913925E-05 -.88690740E+00 -.66564168E+01 -.72258995E+01  .82669769E-01  .15386571E-05  .36354234E-10 -.16998869E+01  .80433646E-01
 .64938627E-05 -.88690740E+00 -.66564168E+01 -.72258995E+01  .82669769E-01  .16676972E-05  .42707648E-10 -.16998869E+01  .80433646E-01
 .70026531E-05 -.88690740E+00 -.66564168E+01 -.72258995E+01  .82669769E-01  .17983603E-05  .49662055E-10 -.16998869E+01  .80433646E-01
 .75178434E-05 -.88690740E+00 -.66564168E+01 -.72258995E+01  .82669769E-01  .19306669E-05  .57238205E-10 -.16998869E+01  .80433646E-01
 .80395140E-05 -.88690740E+00 -.66564168E+01 -.72258995E+01  .82669769E-01  .20646378E-05  .65457444E-10 -.16998869E+01  .80433646E-01
 .85677463E-05 -.88690740E+00 -.66564168E+01 -.72258995E+01  .82669769E-01  .22002939E-05  .74341729E-10 -.16998869E+01  .80433646E-01
""" 
        fvwr = f"{uuid.uuid4()}.vwr"
        with open(fvwr, "w") as f:
            f.write(to_parse)
        parsed = vwr(fvwr)
        os.remove(fvwr)
        print(parsed)

if __name__ == "__main__":
    unittest.main()
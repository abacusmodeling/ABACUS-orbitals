"""a simple module for parsing main information from pseudopotential
With Python-xml parser, provide general parser for UPF format pseudopotential"""

def parse(fname: str):
    """Interface function for parsing pseudopotential file.
    
    Args:
        fname (str): the name of pseudopotential file.
    
    Returns:
        dict: a dictionary contains ALL information that can be parsed from the pseudopotential file.
    """
    import SIAB.io.pseudopotential.components as sipc
    return sipc.parse(fname=fname)


def ppinfo(fname: str):
    """Extract element, valence electronic configuration and z_valence from pseudopotential file.
    
    Args:
        fname (str): the name of pseudopotential file.
    
    Returns:
        dict: a dictionary contains element, valence electronic configuration and z_valence.
    """
    import SIAB.io.pseudopotential.tools.advanced as sipta
    parsed = parse(fname=fname)
    if "val_conf" in parsed["PP_INFO"]["attrib"]:
        # this shortcut may be triggered by the case of vwr format pseudopotential
        return {
            "element": parsed["PP_HEADER"]["attrib"]["element"],
            "val_conf": parsed["PP_INFO"]["attrib"]["val_conf"],
            "z_val": parsed["PP_HEADER"]["attrib"]["z_valence"]
        }
    element, val_conf = sipta.val_conf(parsed=parsed)
    z_val = sipta.z_val(parsed=parsed)
    return { "element": element, "val_conf": val_conf, "z_val": z_val }
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


def parse(fname: str):
    """parse the pseudopotential file, return a dictionary"""
    preprocess(fname)
    tree = ET.parse(fname)
    root = tree.getroot()
    parsed = iter_tree(root)
    postprocess(parsed)
    return parsed

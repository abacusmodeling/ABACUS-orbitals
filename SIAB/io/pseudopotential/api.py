"""a simple module for parsing main information from pseudopotential"""
"""With Python-xml parser, provide general parser for UPF format pseudopotential"""

import SIAB.io.pseudopotential.components as sipc
def parse(fname: str):
    return sipc.parse(fname=fname)

import SIAB.io.pseudopotential.tools.advanced as sipta
def towards_siab(fname: str):
    """towards SIAB generating numerical atomic orbitals, return a dictionary
    contains information like:
    {
        "element": "Fe",
        "valence_electron_configuration": [
            ["1s", "2s", "2p"],
            ["3s", "3p"],
            ["3d"]
        ],
    }"""
    parsed = parse(fname=fname)
    element, valence_electron_configuration = sipta.valence_electron_configuration(parsed=parsed)
    z_valence = sipta.z_valence(parsed=parsed)
    return {
        "element": element,
        "valence_electron_configuration": valence_electron_configuration,
        "z_valence": z_valence
    }
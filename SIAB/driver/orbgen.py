from SIAB.orb.orb import build_orbgraph
from SIAB.spillage.listmanip import flatten
def init(elem,
         rcut,
         ecut,
         siab_settings,
         dft_folders):
    '''a temporary driver for the orbital generation
    '''
    dft_folders = list(set(flatten(dft_folders)))
    primitive_type = siab_settings.get('primitive_type', 'reduced')
    mode = siab_settings.get('fit_basis', 'jy')
    
    graph = build_orbgraph(elem, rcut, ecut, primitive_type, mode)
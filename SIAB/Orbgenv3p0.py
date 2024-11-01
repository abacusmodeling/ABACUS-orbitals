from SIAB.driver.main3p0 import init, rundft, spillage

def main(fn):
    '''main function'''
    glbparam, dftparam, spillparam, compparam = init(fn)

    _ = rundft(glbparam['element'],
               glbparam['bessel_nao_rcut'],
               dftparam, 
               spillparam.get('geoms'), 
               spillparam.get('spill_guess'),
               compparam)
    
    spillage(elem=glbparam['element'],
             ecut=dftparam['ecutwfc'],
             rcuts=glbparam['bessel_nao_rcut'],
             primitive_type=spillparam['primitive_type'],
             scheme=spillparam['orbitals'],
             basis_type=spillparam['fit_basis'])
    
if __name__ == '__main__':
    main('examples/jy-v3.0.json')
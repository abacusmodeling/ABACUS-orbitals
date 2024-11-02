from SIAB.driver.main3p0 import init, rundft, spillage
import argparse
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
             basis_type=spillparam['fit_basis'],
             nthreads_rcut=spillparam.get('nthreads_rcut', 1),
             disp=True)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the ABACUS-ORBGEN workflow')
    # only -i, --input is required
    parser.add_argument('-i', '--input', required=True, help='input json file')
    args = parser.parse_args()
    main(args.input)
    # main('examples/jy-v3.0.json')
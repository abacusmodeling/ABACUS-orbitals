from SIAB.driver.main import init as read
from SIAB.driver.main import rundft, spillage
import argparse
import os

def start() -> str:
    '''User interface, return the input json file specified in -i tag'''

    helpmsg = 'To use ABACUS-ORBGEN v3.0 workflow, you should configure the ABACUS with'\
              ' version higher than 3.7.5. Since v3.0, the workflow has deprecated '\
              'thoroughly the plain text input file as v1.0, 2.0 and 2.0+. Now only '\
              'the json file is supported. An example can be found in folder examples/.'\
              ' For more information on parameter settings, please refer to the '\
              'Github repository:\nhttps://github.com/kirk0830/ABACUS-ORBGEN'

    parser = argparse.ArgumentParser(description=helpmsg)    
    parser.add_argument('-i', '--input', required=True, help='input json file')
    parser.add_argument('-v', '--version', action='version', version='%(prog)s 3.0')
    args = parser.parse_args()

    return args.input

def main():
    '''main function'''

    glbparam, dftparam, spillparam, compparam = read(start())

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
             dft_root=os.getcwd(),
             run_mode=spillparam['fit_basis'],
             nthreads_rcut=spillparam.get('nthreads_rcut', 1),
             disp=True)
    
if __name__ == '__main__':
    '''entry point if run as a script'''
    main()

from SIAB.driver.main3p0 import init, rundft

def main(fn):
    '''main function'''
    glbparam, dftparam, spillparam, compparam = init(fn)
    folders = rundft(glbparam, 
                     dftparam, 
                     spillparam.get('geoms'), 
                     spillparam.get('spill_guess'),
                     compparam)
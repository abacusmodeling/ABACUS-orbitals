import numpy as np

def read_wfc_lcao_txt(fname):
    '''
    Read a wave function coefficient file in txt format.

    '''
    with open(fname, 'r') as f:
        data = f.read()
        data = data.replace('\n', ' ').split()

    nbands = int(data[0])
    nbasis = int(data[4])

    print(f'nbands = {nbands}')
    print(f'nbasis = {nbasis}')

    wfc = np.zeros((nbands, nbasis))
    e = np.zeros(nbands)

    # use the string "(band)" as delimiters
    delim = [i for i, x in enumerate(data) if x == '(band)']


    e = np.array([float(data[i+1]) for i in delim])
    occ = np.array([float(data[i+3]) for i in delim])
    wfc = np.array([[float(c) for c in data[i+5:i+5+nbasis]]
                    for i in delim])



import unittest

class _TestWfc(unittest.TestCase):

    def test_read_wfc_lcao_txt(self):
        read_wfc_lcao_txt('./WFC_NAO_GAMMA1.txt')


if __name__ == '__main__':
    unittest.main()

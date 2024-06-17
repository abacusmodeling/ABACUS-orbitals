from SIAB.spillage.orbio import read_nao
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Arial']})
rc('text', usetex=False)

def plot_chi(chi, r, save=None):
    lmax = len(chi)-1
    nzeta = [len(chi_l) for chi_l in chi]

    fig, ax = plt.subplots(nrows=1, ncols=lmax+1, figsize=((lmax+1)*7, 6), layout='tight', squeeze=False)
    
    for l, chi_l in enumerate(chi):
        for zeta, chi_lz in enumerate(chi_l):
            # adjust the sign so that the largest value is positive
            if chi_lz[np.argmax(np.abs(chi_lz))] < 0:
                chi_lz *= -1
    
            ax[l].plot(r, chi_lz, label='$\zeta=%i$'%(zeta))
    
        ax[0, l].legend(fontsize=16)
        ax[0, l].axhline(0, color='black', linestyle=':')
        ax[0, l].set_title('$l=%d$'%(l), fontsize=20)
        ax[0, l].set_xlim([0, r[-1]])

    if save is not None:
        plt.savefig(save)


def plot_orbfile(orbfile, save=None):
    nao = read_nao(orbfile)
    r = nao['dr'] * np.arange(nao['nr'])
    chi = nao['chi']
    plot_chi(chi, r, save=save)


if __name__ == '__main__':

    #plot_orbfile('/home/zuxin/tmp/nao/v2.0/SG15-Version1p0__AllOrbitals-Version2p0/72_Hf_TZDP/Hf_gga_6au_100Ry_6s3p3d3f2g.orb')
    #plot_orbfile('/home/zuxin/abacus-community/abacus_orbital_generation/Si/Si_2s2p1d/7au_40Ry/Si_gga_40Ry_7au_2s2p1d.orb')
    #plot_orbfile('/home/zuxin/abacus-community/abacus_orbital_generation/Si/Si_3s3p2d/7au_40Ry/Si_gga_40Ry_7au_3s3p2d.orb')
    
    #plt.show()

    pass


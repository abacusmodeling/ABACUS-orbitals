from util import ND_list
import numpy as np
from scipy.special import spherical_jn
from scipy.integrate import simps
from scipy.optimize import fsolve
import functools
import torch

def generate_orbital(info_element, info_radial, C, E):
	""" C[it][il][ie,iu] """
	""" orb[it][il][iu][r] = \suml_{ie} C[it][il][ie,iu] * jn(il,ie*r) """
	orb = dict()
	for it in info_element:
		Nr = int(info_radial["Rcut"][it]/info_radial["dr"][it])+1
		orb[it] = ND_list(info_element[it].Nl)
		for il in range(info_element[it].Nl):
			orb[it][il] = ND_list(info_element[it].Nu[il])
			for iu in range(info_element[it].Nu[il]):
				orb[it][il][iu] = np.zeros(Nr)
				for ir in range(Nr):
					r = ir * info_radial["dr"][it]
					for ie in range(info_element[it].Ne):
						orb[it][il][iu][ir] += C[it][il][ie,iu].item() * spherical_jn(il,E[it][il,ie].item()*r)
	return orb

def generate_jn(info_element, E):
	""" orb[it][il][ie][r] = jn(il,ie*r) """
	orb = dict()
	for it in info_element:
		Nr = int(info_element[it].Rcut/info_element[it].dr)+1
		orb[it] = ND_list(info_element[it].Nl)
		for il in range(info_element[it].Nl):
			orb[it][il] = ND_list(info_element[it].Ne)
			for ie in range(info_element[it].Ne):
				orb[it][il][ie] = np.zeros(Nr)
				for ir in range(Nr):
					r = ir * info_element[it].dr
					orb[it][il][ie][ir] = spherical_jn(il, E[it][il,ie].item()*r)
	return orb

def generate_jn_smooth(info_element, E):
	""" orb[it][il][ie][r] = jn(il,ie*r) - jn(il,(ie+1)*r) * jn'(il,ie*Rcut) / jn'(il,(ie+1)*Rcut) """
	orb = dict()
	for it in info_element:
		Nr = int(info_element[it].Rcut/info_element[it].dr)+1
		orb[it] = ND_list(info_element[it].Nl)
		for il in range(info_element[it].Nl):
			orb[it][il] = ND_list(info_element[it].Ne-1)
			for ie in range(info_element[it].Ne-1):
				mix_coef = ( spherical_jn(il, E[it][il,ie  ].item()*info_element[it].Rcut, True) * E[it][il,ie  ].item() )			\
					     / ( spherical_jn(il, E[it][il,ie+1].item()*info_element[it].Rcut, True) * E[it][il,ie+1].item() )
				orb[it][il][ie] = np.zeros(Nr)
				for ir in range(Nr):
					r = ir * info_element[it].dr
					orb[it][il][ie][ir] = spherical_jn(il, E[it][il,ie].item()*r) - mix_coef * spherical_jn(il, E[it][il,ie+1].item()*r)
	return orb



def smooth_orbital(orb_t, Rcut, dr, smearing_sigma):
	""" orb_t[il][iu,r] """
	for orb_tl in orb_t:
		for orb_tlu in orb_tl:
			for ir in range(orb_tlu.shape[0]):
				assert orb_tlu.shape[0] == int(Rcut/dr)+1
				r = ir * dr
				orb_tlu[ir] *= 1-np.exp( -(r-Rcut)**2/(2*smearing_sigma**2) )

def inner_product(orb1, orb2, dr):
	assert orb1.shape == orb2.shape
	r = np.array(range(orb1.shape[0]))*dr
	return simps( orb1 * orb2 * r * r, dx=dr )

def normalize(orb, dr, C=None, flag_norm_orb=False, flag_norm_C=False):
	""" C[it][il][ie,iu] """
	""" orb[it][il][iu][r] = \suml_{ie} C[it][il][ie,iu] * jn(il,ie*r) """
	for it,orb_t in orb.items():
		for il,orb_tl in enumerate(orb_t):
			for iu,orb_tlu in enumerate(orb_tl):
				norm = np.sqrt(inner_product(orb_tlu,orb_tlu,dr[it]))
				if flag_norm_orb:  orb_tlu[:]           = orb_tlu              / norm
				if flag_norm_C:    C[it][il].data[:,iu] = C[it][il].data[:,iu] / norm

def orth(orb_t, dr):
	""" |n'> = 1/Z ( |n> - \sum_{i=0}^{n-1} |i><i|n> ) """
	""" orb_t[il][iu,r] """
	for il,orb_tl in enumerate(orb_t):
		for iu1,orb_tlu1 in enumerate(orb_tl):
			for iu2 in range(iu1):
				orb_tlu1[:] -= orb_tl[iu2] * inner_product(orb_tlu1,orb_tl[iu2],dr)
			orb_tlu1[:] = orb_tlu1 / np.sqrt(inner_product(orb_tlu1,orb_tlu1,dr))



def find_eigenvalue(Nl, Ne):
	""" E[il,ie] """
	E = np.zeros((Nl,Ne+Nl+1))
	for ie in range(1,Ne+Nl+1):
		E[0,ie] = ie*np.pi
	for il in range(1,Nl):
		jl = functools.partial(spherical_jn,il)
		for ie in range(1,Ne+Nl+1-il):
			E[il,ie] = fsolve( jl, (E[il-1,ie]+E[il-1,ie+1])/2 )
	return E[:,1:Ne+1]

def set_E(info_element, Rcut):
	""" E[it][il,ie] """
	eigenvalue = { it:find_eigenvalue(info_element[it].Nl,info_element[it].Ne) for it in info_element }
	E = dict()
	for it in info_element:
		E[it] = torch.from_numpy(( eigenvalue[it]/Rcut[it] ).astype("float64"))
	return E
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# present packages
import SIAB.spillage.pytorch_swat.IO.read_QSV as sspsirqsv
import SIAB.spillage.pytorch_swat.IO.func_C as sspsifc
import SIAB.spillage.pytorch_swat.IO.read_json as sspsirj
import SIAB.spillage.pytorch_swat.IO.print_orbital as sspsipo
import SIAB.spillage.pytorch_swat.IO.change_info as sspsicinfo
import SIAB.spillage.pytorch_swat.IO.cal_weight as sspsicw
from SIAB.spillage.pytorch_swat.opt_orbital import Opt_Orbital
from SIAB.spillage.pytorch_swat.opt_orbital_wavefunc import Opt_Orbital_Wavefunc
import SIAB.spillage.pytorch_swat.orbital as sspso
import SIAB.spillage.pytorch_swat.util as sspsu
# released/official packages
import torch
import torch_optimizer 
import pprint
import numpy as np
import time

def main(params: dict = None):
	seed = int(1000*time.time())%(2**32)
	np.random.seed(seed)
	print("seed:",seed)
	time_start = time.time()
	if params is None:
		file_list, info_true, weight_info, C_init_info, V_info = sspsirj.read_json("INPUT")
	else:
		file_list = params["file_list"]
		weight_info = params["weight"]
		C_init_info = params["C_init_info"]
		V_info = params["V_info"]
		info_true = sspsirj.Info()
		for key, value in params["info"].items():
			info_true.__dict__[key] = value
		info_true.Nl = {it:len(Nu) for it, Nu in info_true.Nu.items()}
	# weight is the one has dimension [nSTRU*nkpt][nband], is calculated by multiplication of two weight data matrix,
	# the first is STRU_weight, has dimension [nSTRU][nband], because the band itself doesnot have weight, the weight is from STRU
	# the second is kpt_weight, has dimension [nSTRU][nkpt], for each STRU, there will be a kpoint list and kpt itself has weight
	# every element of weight is calculated by multiplying the corresponding element of STRU_weight and kpt_weight,
	# get physical meaning of the weight per STRU per (k, b)-pair.
	# More exactly, the weight should have the structure like:
	#				band1	band2	band3	...
	# STRU1-kpt1    w11		w12		w13		...
	# STRU1-kpt2	...
	# ...			
	# STRU1-kptN	...
	# STRU2-kpt1	...
	# STRU2-kpt2	...
	# ...
	# STRU2-kptN	...
	# ...
	weight = sspsicw.cal_weight(weight_info, V_info["same_band"], file_list["origin"])

	info_kst = sspsirqsv.read_file_head(info_true,file_list["origin"])

	info_stru, info_element, info_opt = sspsicinfo.change_info(info_kst,weight)
	info_max = sspsicinfo.get_info_max(info_stru, info_element)

	print("info_kst:", info_kst, sep="\n", end="\n"*2, flush=True)
	print("info_stru:", pprint.pformat(info_stru), sep="\n", end="\n"*2, flush=True)
	print("info_element:", pprint.pformat(info_element,width=40), sep="\n", end="\n"*2, flush=True)
	print("info_opt:", pprint.pformat(info_opt,width=40), sep="\n", end="\n"*2, flush=True)
	print("info_max:", pprint.pformat(info_max), sep="\n", end="\n"*2, flush=True)
	# Q is the overlap matrix between psi and jY, or that of the gradient of psi and jY
	# Q = <psi|jY> or <grad(psi)|grad(jY)>
	# S is the overlap matrix between jY and jY, or that of the gradient of jY and jY
	# S = <jY|jY> or <grad(jY)|grad(jY)>
	# V is the overlap matrix between psi and psi, or that of the gradient of psi and psi
	# V = <psi|psi> or <grad(psi)|grad(psi)>
	QI,SI,VI_origin = sspsirqsv.read_QSV(info_stru, info_element, file_list["origin"], V_info)
	if "linear" in file_list.keys():
		QI_linear, SI_linear, VI_linear = list(zip(*( sspsirqsv.read_QSV(info_stru, info_element, file, V_info) for file in file_list["linear"] )))

	# C is intialized here! carefully treat it!
	if C_init_info["init_from_file"]:
		C, C_read_index = sspsifc.read_C_init( C_init_info["C_init_file"], info_element )
	else:
		C = sspsifc.random_C_init(info_element)
	E = sspso.set_E(info_element)
	sspso.normalize(
		sspso.generate_orbital(info_element,C,E),
		{it:info_element[it].dr for it in info_element},
		C, flag_norm_C=True)

	#opt = torch.optim.Adam(sum( ([c.real,c.imag] for c in sum(C,[])), []), lr=info_opt.lr, eps=1e-8)
	#opt = torch.optim.Adam( sum(C.values(),[]), lr=info_opt.lr, eps=1e-20, weight_decay=info_opt.weight_decay)
	#opt = radam.RAdam( sum(C.values(),[]), lr=info_opt.lr, eps=1e-20 )
	opt = torch_optimizer.SWATS( sum(C.values(),[]), lr=info_opt.lr, eps=1e-20 )


	with open("Spillage.dat","w") as S_file:

		print( "\nSee \"Spillage.dat\" for detail status: " , flush=True )
		if info_opt.cal_T:
			print( '%5s'%"istep", "%20s"%"Spillage", "%20s"%"T.item()", "%20s"%"Loss", flush=True )
		else:
			print( '%5s'%"istep", "%20s"%"Spillage", flush=True )
		# initialize the loss_old to be infinity, so that the optimization will start
		loss_old = np.inf
		# arbitrarily set the max step to 30000 and if input defines it, use the input value
		maxSteps = 30000
		if type(info_opt.max_steps) == int :
			if info_opt.max_steps > 0 :
				maxSteps = info_opt.max_steps
		# optimization loop starts here
		for istep in range(maxSteps):
			# START: hack function here to change definition of Spillage
			Spillage = 0
			for ist in range(len(info_stru)): # for each structure...
				# initialize a new Opt_Orbital_Wavefunc object for STRU
				opt_orb_wave = Opt_Orbital_Wavefunc(info_stru[ist], info_element, V_info)
				# 
				V_origin = opt_orb_wave.cal_V_origin(C, QI[ist], SI[ist])
				# linear corresponds to the mode of calculating the derivative of the wavefunction
				if "linear" in file_list.keys():
					V_linear = [ opt_orb_wave.cal_V_linear(C, QI_linear[i][ist], SI_linear[i][ist])
						for i in range(len(file_list["linear"]))]
						
				def cal_Spillage(V_delta):
					Spillage = (V_delta * weight[ist][:info_stru[ist].Nb_true]).sum()
					return Spillage

				def cal_delta(VI, V):
					return ((VI[ist]-V)/sspsu.update0(VI[ist])).abs()		# abs or **2?
				# central expression appears to be here, one can modifiy the mixing coefficients between the two terms
				# psi and dpsi here.
				coeff_psi = 2  # 2 is the default value
				coeff_dpsi = 1
				cstpsi = 1
				fpsi_dpsi = 1
				# because we dont have the second order derivatives of quantities, the further mixing with ddpsi is not
				# possible.
				# coeff_ddpsi = 0
				# CALCULATE THE SPILLAGE
				s0 = cal_Spillage(cal_delta(VI_origin,V_origin))
				Spillage += coeff_psi*s0
				
				if "linear" in file_list.keys():
					for i in range(len(file_list["linear"])):
						s1 = cal_Spillage(cal_delta(VI_linear[i],V_linear[i]))
						Spillage += coeff_dpsi*s1
						# Spillage += coeff_dpsi*s1*(cstpsi + s0/fpsi_dpsi)

			# END: hack function here to change definition of Spillage
			if info_opt.cal_T:
				T = Opt_Orbital.cal_T(C,E)
				if not "TSrate" in vars():	TSrate = torch.abs(0.002*Spillage/T).data[0]
				Loss = Spillage + TSrate*T
			else:
				Loss = Spillage

			if info_opt.cal_T:
				print_content = [istep, Spillage.item(), T.item(), Loss.item()]
			else:
				print_content = [istep, Spillage.item()]
			print(*print_content, sep="\t", file=S_file, flush=True)
			if not istep%100:
				print(*print_content, sep="\t", flush=True)

			flag_finish = 0
			if Loss.item() < loss_old:
				loss_old = Loss.item()
				C_old = sspsifc.copy_C(C,info_element)
				flag_finish = 0
			else:
				flag_finish += 1
				if flag_finish > 50:
					break

			opt.zero_grad()
			Loss.backward()		
			if C_init_info["init_from_file"] and not C_init_info["opt_C_read"]:
				for it,il,iu in C_read_index:
					C[it][il].grad[:,iu] = 0
			opt.step()
			#orbital.normalize(
			#	orbital.generate_orbital(info_element,C,E),
			#	{it:info_element[it].dr for it in info_element},
			#	C, flag_norm_C=True)

	orb = sspso.generate_orbital(info_element,C_old,E)
	if info_opt.cal_smooth:
		sspso.smooth_orbital(
			orb,
			{it:info_element[it].Rcut for it in info_element}, {it:info_element[it].dr for it in info_element},
			0.1)
	sspso.orth(
		orb,
		{it:info_element[it].dr for it in info_element})
	sspsipo.print_orbital(orb,info_element)
	sspsipo.plot_orbital(
		orb,
		{it:info_element[it].Rcut for it in info_element},
		{it:info_element[it].dr for it in info_element})

	sspsifc.write_C("ORBITAL_RESULTS.txt",C_old,Spillage)

	print("Time (PyTorch):     %s\n"%(time.time()-time_start), flush=True )


if __name__=="__main__":
	import sys
	np.set_printoptions(threshold=sys.maxsize, linewidth=10000)
	print( sys.version, flush=True ) 
	main()

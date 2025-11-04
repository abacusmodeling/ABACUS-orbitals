#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import IO.read_QSV
import IO.print_QSV
import IO.func_C
import IO.read_json
import IO.print_orbital
import IO.cal_weight
import IO.change_info
import orbital
import optimize
from opt_orbital import Opt_Orbital
from opt_orbital_spillage import Opt_Orbital_Spillage

import numpy as np
import torch
import time
import pprint

def main():
	seed = int(1000*time.time())%(2**32)
	np.random.seed(seed)
	print("seed:",seed)
	time_start = time.time()

	file_list, info_true, info_weight, info_optimize, info_C_init, info_V, info_radial = IO.read_json.read_json("INPUT")

	weight = IO.cal_weight.cal_weight(info_weight, info_V["same_band"], file_list["origin"])

	info_kst = IO.read_QSV.read_file_head(info_true, file_list["origin"])

	info_stru, info_element = IO.change_info.change_info(info_kst, weight, info_V["same_band"])
	#info_max = IO.change_info.get_info_max(info_stru, info_element)

	print("info_kst:", pprint.pformat(info_kst), sep="\n", end="\n"*2)
	print("info_element:", pprint.pformat(info_element,width=40), sep="\n", end="\n"*2)
	print("info_optimize:", pprint.pformat(info_optimize,width=40), sep="\n", end="\n"*2)
	print("info_radial:", pprint.pformat(info_radial,width=40), sep="\n", end="\n"*2)
	print("info_stru:", pprint.pformat(info_stru), sep="\n", end="\n"*2)
	#print("info_max:", pprint.pformat(info_max), sep="\n", end="\n"*2)

	QI,SI,VI_origin = IO.read_QSV.read_QSV(info_stru, info_element, file_list["origin"], info_V)
	if "linear" in file_list.keys():
		QI_linear, SI_linear, VI_linear = list(zip(*( IO.read_QSV.read_QSV(info_stru, info_element, file, info_V) for file in file_list["linear"] )))

	if info_C_init["init_from_file"]:
		C, C_read_index = IO.func_C.read_C_init( info_C_init["C_init_file"], info_element )
	else:
		C = IO.func_C.random_C_init(info_element)
	E = orbital.set_E(info_element, info_radial["Rcut"])
	orbital.normalize(
		orbital.generate_orbital(info_element, info_radial, C, E),
		info_radial["dr"],
		C, flag_norm_C=True)

	with open("Spillage.dat","w") as S_file:

		for info_opt in info_optimize:

			print( '\nSee "Spillage.dat" for detail status:'  )
			print( "istep", "Spillage", sep="\t" )

			opt = optimize.get_optim( info_opt, sum(C.values(),[]))

			spillage = Opt_Orbital_Spillage(info_stru, info_element, info_V, info_opt["norm"], file_list)
			spillage.set_QSVI(QI, SI, VI_origin)
			if "linear" in file_list.keys():
				spillage.set_QSVI_linear(QI_linear, SI_linear, VI_linear)

			data_transmit = dict()

			def closure():
				nonlocal data_transmit

				Spillage = spillage.cal_Spillage(C)

				if info_opt["cal_T"]:
					T = Opt_Orbital.cal_T(C,E)
					if not "TSrate" in vars():	TSrate = torch.abs(0.002*Spillage/T).data[0]
					Loss = Spillage + TSrate*T
				else:
					Loss = Spillage

				if info_opt["cal_T"]:
					print_content = [data_transmit["istep_big"], data_transmit["istep_small"], data_transmit["istep_all"], Spillage.item(), T.item(), Loss.item()]
				else:
					print_content = [data_transmit["istep_big"], data_transmit["istep_small"], data_transmit["istep_all"], Spillage.item()]
				print(*print_content, sep="\t", file=S_file)
				data_transmit["istep_small"] += 1
				data_transmit["istep_all"] += 1

				data_transmit.update({"Loss":Loss.item(), "Spillage":Spillage.item()})
				if info_opt["optimizer"] != "LBFGS":
					if Loss < data_transmit["loss_saved"]:
						data_transmit["loss_saved"] = Loss
						data_transmit["flag_finish"] = 0
						data_transmit["C"] = IO.func_C.copy_C(C,info_element)
					else:
						data_transmit["flag_finish"] += 1
				else:
					data_transmit["C"] = IO.func_C.copy_C(C,info_element)

				opt.zero_grad()
				Loss.backward()
				if info_C_init["init_from_file"] and not info_C_init["opt_C_read"]:
					for it,il,iu in C_read_index:
						C[it][il].grad[:,iu] = 0

				return Loss

			data_transmit["istep_all"] = 0
			if info_opt["optimizer"] != "LBFGS":
				data_transmit["loss_saved"] = np.inf
				data_transmit["flag_finish"] = 0

			for data_transmit["istep_big"] in range(info_opt["max_steps"]):

				data_transmit["istep_small"] = 0

				if info_opt["optimizer"] != "LBFGS":
					if data_transmit["flag_finish"] > 50:
						break

				opt.step(closure)

				if (info_opt["optimizer"]=="LBFGS") or (data_transmit["istep_big"]%10==0):
					print(data_transmit["istep_big"], data_transmit["Spillage"], sep="\t")

				if info_opt["optimizer"] == "LBFGS":
					if data_transmit["istep_small"]==1:
						break

				#orbital.normalize(
				#	orbital.generate_orbital(info_element, info_radial, C, E),
				#	{it:info_element[it].dr for it in info_element},
				#	C, flag_norm_C=True)

	orb = orbital.generate_orbital(info_element, info_radial, data_transmit["C"], E)
	for it in info_element:
		if info_radial["smearing_sigma"][it]:
			orbital.smooth_orbital(
				orb[it],
				info_radial["Rcut"][it],
				info_radial["dr"][it],
				info_radial["smearing_sigma"][it])
		orbital.orth(
			orb[it],
			info_radial["dr"][it])
	IO.print_orbital.print_orbital(
		orb,
		info_radial)
	IO.print_orbital.plot_orbital(
		orb,
		info_radial["Rcut"],
		info_radial["dr"])

	IO.func_C.write_C("ORBITAL_RESULTS.txt", data_transmit["C"], data_transmit["Spillage"])

	print("Time (PyTorch):     %s\n"%(time.time()-time_start) )


if __name__=="__main__":
	import sys
	np.set_printoptions(threshold=sys.maxsize, linewidth=10000)
	print( sys.version )
	main()

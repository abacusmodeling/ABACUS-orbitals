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
from opt_orbital_converge import Opt_Orbital_Converge

import numpy as np
import time
import pprint
import copy
import functools
import operator

def main():
	seed = int(1000*time.time())%(2**32)
	np.random.seed(seed)
	print("seed:",seed)
	time_start = time.time()

	file_list, info_true, info_weight, info_optimize, info_C_init, info_V, info_radial = IO.read_json.read_json("INPUT")
	info_each_thr = {"Loss": info_true.Loss_thr, "orb_num": info_true.orb_num}

	weight = IO.cal_weight.cal_weight(info_weight, info_V["same_band"], file_list["origin"])

	info_kst = IO.read_QSV.read_file_head(info_true, file_list["origin"])

	info_stru, info_element = IO.change_info.change_info(info_kst, weight, info_V["same_band"])
	#info_max = IO.change_info.get_info_max(info_stru, info_element)

	print("info_kst:", pprint.pformat(info_kst), sep="\n", end="\n"*2)
	print("info_element:", pprint.pformat(info_element,width=40), sep="\n", end="\n"*2)
	print("info_each_thr:", pprint.pformat(info_each_thr), sep="\n", end="\n"*2)
	print("info_optimize:", pprint.pformat(info_optimize,width=40), sep="\n", end="\n"*2)
	print("info_radial:", pprint.pformat(info_radial,width=40), sep="\n", end="\n"*2)
	print("info_stru:", pprint.pformat(info_stru), sep="\n", end="\n"*2)
	#print("info_max:", pprint.pformat(info_max), sep="\n", end="\n"*2)

	QI,SI,VI_origin = IO.read_QSV.read_QSV(info_stru, info_element, file_list["origin"], info_V)
	if "linear" in file_list.keys():
		QI_linear, SI_linear, VI_linear = list(zip(*( IO.read_QSV.read_QSV(info_stru, info_element, file, info_V) for file in file_list["linear"] )))

	E = orbital.set_E(info_element, info_radial["Rcut"])

	opt_orb_conv = Opt_Orbital_Converge()
	opt_orb_conv.set_info(file_list, info_optimize, info_stru, info_C_init, info_V)
	opt_orb_conv.set_QSVI(QI, SI, VI_origin)
	if "linear" in file_list.keys():
		opt_orb_conv.set_QSVI_linear(QI_linear, SI_linear, VI_linear)
	opt_orb_conv.set_E(E)

	t_l_saved = []
	with open("Spillage_concise.dat","w") as file_concise, open("Spillage_detail.dat","w") as file_detail:

		for it in info_element:
			info_element[it]["Nu"][0] = 1
			t_l_saved.append((it,0))
		C_test = IO.func_C.random_C_init(info_element)
		opt_orb_conv.set_info_element(info_element)
		data_transmit = opt_orb_conv.cal_converge(C_test, (file_concise,file_detail))
		print("orbital\t", "Loss", sep="\t")
		print(*t_l_saved, data_transmit["Loss"], sep="\t")

		istep=0

		while data_transmit["Loss"]>info_each_thr["Loss"] and functools.reduce(operator.add, [2*il+1 for it,il in t_l_saved]) < info_each_thr["orb_num"]:

			istep+=1
			delta_Loss_max = 0
			for it in info_element:
				for il in range(info_element[it]["Nl"]):

					info_element_test = copy.deepcopy(info_element)
					info_element_test[it]["Nu"][il] += 1
					C_test, C_read_index = IO.func_C.cover_C(data_transmit["C"], info_element_test)
					opt_orb_conv.set_C_read_index(C_read_index)

					opt_orb_conv.set_info_element(info_element_test)
					data_transmit_try = opt_orb_conv.cal_converge(C_test, (file_concise,file_detail))

					delta_Loss_try = (data_transmit["Loss"] - data_transmit_try["Loss"]) / (2*il+1)
					print("\t\t", f"- {2*il+1} * {delta_Loss_try}", sep="\t")
					if delta_Loss_try > delta_Loss_max:
						delta_Loss_max = delta_Loss_try
						t_l_max = it,il
						data_transmit_try_max = data_transmit_try
			t_l_saved.append(t_l_max)
			data_transmit = data_transmit_try_max
			info_element[t_l_max[0]]["Nu"][t_l_max[1]] += 1
			print(t_l_saved[-1], data_transmit["Loss"], sep="\t")

	print("\nFinal orbitals:")
	print(*t_l_saved, sep="\t")
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

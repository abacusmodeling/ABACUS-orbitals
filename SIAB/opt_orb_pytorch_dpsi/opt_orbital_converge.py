from opt_orbital_spillage import Opt_Orbital_Spillage
import IO.func_C
from opt_orbital import Opt_Orbital
import optimize

import torch
import numpy as np


class Opt_Orbital_Converge:
	def set_info(self, file_list, info_optimize, info_stru, info_C_init, info_V):
		self.file_list = file_list
		self.info_optimize = info_optimize
		self.info_stru = info_stru
		self.info_C_init = info_C_init
		self.info_V = info_V

	def set_info_element(self, info_element):
		self.info_element = info_element

	def set_QSVI(self, QI, SI, VI_origin):
		self.QI = QI
		self.SI = SI
		self.VI_origin = VI_origin

	def set_QSVI_linear(self, QI_linear, SI_linear, VI_linear):
		self.QI_linear = QI_linear
		self.SI_linear = SI_linear
		self.VI_linear = VI_linear

	def set_C_read_index(self, C_read_index):
		self.C_read_index = C_read_index

	def set_E(self, E):
		self.E = E

	def cal_converge(self, C, files):
		data_transmit = dict()
		for info_opt in self.info_optimize:

			print( 'See "Spillage.dat" for detail status:', file=files[0], flush=True )
			print( "istep", "Spillage", sep="\t", file=files[0], flush=True )

			opt = optimize.get_optim( info_opt, sum(C.values(),[]))

			spillage = Opt_Orbital_Spillage(self.info_stru, self.info_element, self.info_V, info_opt["norm"], self.file_list)
			spillage.set_QSVI(self.QI, self.SI, self.VI_origin)
			if "linear" in self.file_list.keys():
				spillage.set_QSVI_linear(self.QI_linear, self.SI_linear, self.VI_linear)

			def closure():
				nonlocal data_transmit

				Spillage = spillage.cal_Spillage(C)

				if info_opt["cal_T"]:
					T = Opt_Orbital.cal_T(C, self.E)
					if not "TSrate" in vars():	TSrate = torch.abs(0.002*Spillage/T).data[0]
					Loss = Spillage + TSrate*T
				else:
					Loss = Spillage

				if info_opt["cal_T"]:
					print_content = [data_transmit["istep_big"], data_transmit["istep_small"], data_transmit["istep_all"], Spillage.item(), T.item(), Loss.item()]
				else:
					print_content = [data_transmit["istep_big"], data_transmit["istep_small"], data_transmit["istep_all"], Spillage.item()]
				print(*print_content, sep="\t", file=files[1])
				data_transmit["istep_small"] += 1
				data_transmit["istep_all"] += 1

				data_transmit.update({"Loss":Loss.item(), "Spillage":Spillage.item()})
				if info_opt["optimizer"] != "LBFGS":
					if Loss < data_transmit["loss_saved"]:
						data_transmit["loss_saved"] = Loss
						data_transmit["flag_finish"] = 0
						data_transmit["C"] = IO.func_C.copy_C(C,self.info_element)
					else:
						data_transmit["flag_finish"] += 1
				else:
					data_transmit["C"] = IO.func_C.copy_C(C,self.info_element)

				opt.zero_grad()
				Loss.backward()
				if hasattr(self, "C_read_index"):
					for it,il,iu in self.C_read_index:
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

				if (info_opt["optimizer"]=="LBFGS") or (data_transmit["istep_big"]%100==0):
					print(data_transmit["istep_big"], data_transmit["Spillage"], sep="\t", file=files[0], flush=True)

				if info_opt["optimizer"] == "LBFGS":
					if data_transmit["istep_small"]==1:
						break
		return data_transmit
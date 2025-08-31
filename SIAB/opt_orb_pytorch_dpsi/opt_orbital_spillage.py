#=======================
# AUTHOR : Peize Lin
# DATE :   2025-08-31
#=======================

import opt_orbital_wavefunc
import util

class Opt_Orbital_Spillage:

	def __init__(self, info_stru, info_element, V_info, file_list):
		self.info_stru = info_stru
		self.info_element = info_element
		self.V_info = V_info
		self.file_list = file_list

	def set_QSVI(self, QI, SI, VI_origin):
		self.QI = QI
		self.SI = SI
		self.VI_origin = VI_origin

	def set_QSVI_linear(self, QI_linear, SI_linear, VI_linear):
		self.QI_linear = QI_linear
		self.SI_linear = SI_linear
		self.VI_linear = VI_linear

	def cal_Spillage(self, C):
		Spillage = 0

		for ist in range(len(self.info_stru)):
			opt_orb_wave = opt_orbital_wavefunc.Opt_Orbital_Wavefunc(self.info_stru[ist], self.info_element, self.V_info)

			V_origin = opt_orb_wave.cal_V_origin(C, self.QI[ist], self.SI[ist])

			if "linear" in self.file_list.keys():
				V_linear = [ opt_orb_wave.cal_V_linear(C, self.QI_linear[i][ist], self.SI_linear[i][ist])
					for i in range(len(self.file_list["linear"]))]

			def cal_Spillage(V_delta):
				Spillage = (V_delta * self.info_stru[ist].weight[:self.info_stru[ist].Nb_true]).sum()
				return Spillage

			def cal_delta(VI, V):
				return ((VI[ist]-V)/util.update0(VI[ist])).abs()		# abs or **2?

			Spillage += 2*cal_Spillage(cal_delta(self.VI_origin,V_origin))
			if "linear" in self.file_list.keys():
				for i in range(len(self.file_list["linear"])):
					Spillage += cal_Spillage(cal_delta(self.VI_linear[i],V_linear[i]))

		return Spillage
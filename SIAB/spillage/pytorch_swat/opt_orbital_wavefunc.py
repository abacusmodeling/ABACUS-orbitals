#=======================
# AUTHOR : Peize Lin
# DATE :   2022-10-03
#=======================

from SIAB.spillage.pytorch_swat.opt_orbital import Opt_Orbital

class Opt_Orbital_Wavefunc:

	def __init__(self, info_stru, info_element, V_info):
		self.info_stru = info_stru
		self.info_element = info_element
		self.V_info = V_info

	def  cal_V_origin(self, C, QI, SI):
		"""
			C[it][il][ie,iu]
			QI[it][il][ib*ia*im,ie]
			SI[it1,it2][il1][il2][ia1,im1,ie1,ia2,im2,ie2]
			self.coef[ib,it*il*ia*im*iu]
			self.V[ib1,ib2]
			V_origin[ib]  or  V_origin[ib1,ib2]
		"""
		Q = Opt_Orbital.change_index_Q(
			Opt_Orbital.cal_Q( QI, C, self.info_stru, self.info_element ),
			self.info_stru)
		S = Opt_Orbital.change_index_S(
			Opt_Orbital.cal_S( SI, C,self.info_stru, self.info_element ),
			self.info_stru, self.info_element)
		self.coef = Opt_Orbital.cal_coef(Q, S)
		self.V = Opt_Orbital.cal_V(self.coef, Q)
		V_origin = Opt_Orbital.cal_V_origin(self.V, self.V_info)
		return V_origin

	# attention: must cal_V_origin() firstly
	def cal_V_linear(self, C, QI_linear, SI_linear):
		"""
			C[it][il][ie,iu]
			QI_linear[it][il][ib*ia*im,ie]
			SI_linear[it1,it2][il1][il2][ia1,im1,ie1,ia2,im2,ie2]
			V_linear[ib]  or  V_linear[ib1,ib2]
		"""
		Q_linear = Opt_Orbital.change_index_Q(
			Opt_Orbital.cal_Q( QI_linear, C, self.info_stru, self.info_element ),
			self.info_stru)
		S_linear = Opt_Orbital.change_index_S(
			Opt_Orbital.cal_S( SI_linear, C, self.info_stru, self.info_element ),
			self.info_stru, self.info_element)
		V_linear = Opt_Orbital.cal_V_linear( self.coef, Q_linear, S_linear, self.V, self.V_info )
		return V_linear

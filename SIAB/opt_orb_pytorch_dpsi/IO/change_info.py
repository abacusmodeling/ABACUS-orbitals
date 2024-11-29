import addict
import util
import itertools

def change_info(info_old, weight_old, flag_same_band):
	info_stru = [None] * info_old.Nst
	for ist in range(len(info_stru)):
		info_stru[ist] = addict.Dict()
		info_stru[ist].Na = info_old.Na[ist]
		info_stru[ist].Nl = info_old.Nl[ist]
	for ist,weight in enumerate(weight_old):
		info_stru[ist].weight = weight
		info_stru[ist].Nb = weight.shape[0]
		def get_Nb_true():
			if flag_same_band:
				for ib in range(weight.shape[0], 0, -1):
					if weight[ib-1]>0:
						return ib
			else:
				for ib in range(weight.shape[0], 0, -1):
					for ib2 in range(0, ib):
						if weight[ib-1,ib2]>0 or weight[ib2,ib-1]>0:
							return ib
			return 0
		info_stru[ist].Nb_true = get_Nb_true()

	info_element = addict.Dict()
	for it_index,it in enumerate(info_old.Nt_all):
		info_element[it].index = it_index
	for it,Nu in info_old.Nu.items():
		info_element[it].Nu = Nu
		info_element[it].Nl = len(Nu)
	for it,Rcut in info_old.Rcut.items():
		info_element[it].Rcut = Rcut
	for it,dr in info_old.dr.items():
		info_element[it].dr = dr
	for it,Ecut in info_old.Ecut.items():
		info_element[it].Ecut = Ecut
	for it,Ne in info_old.Ne.items():
		info_element[it].Ne = Ne

	info_opt = addict.Dict()
	info_opt.lr = info_old.lr
	info_opt.cal_T = info_old.cal_T
	info_opt.cal_smooth = info_old.cal_smooth
	info_opt.max_steps = info_old.max_steps

	return info_stru, info_element, info_opt

	"""

	info_kst:
	Nt_all  ['C', 'O']
	Nu      {'C': [2, 2, 1], 'O': [3, 2, 1]}
	Rcut    {'C': 6, 'O': 6}
	dr      {'C': 0.01, 'O': 0.01}
	Ecut    {'C': 100, 'O': 100}
	lr      0.01
	cal_T   False
	cal_smooth      False
	max_steps       200
	Nl      [2, 2, 2]
	Nst     3
	Nt      [['C'], ['C'], ['C', 'O']]
	Na      [{'C': 1}, {'C': 1}, {'C': 1, 'O': 2}]
	Nb      [6, 6, 10]
	Ne      {'C': 19, 'O': 19}

	info_stru =
	[{'Na': {'C': 1},
	  'Nb': 6,
	  'Nb_true': 4,
	  'Nl': 2,
	  'weight': tensor([0.0333, 0.0111, 0.0111, 0.0111, 0.0000, 0.0000])},
	 {'Na': {'C': 1},
	  'Nb': 6,
	  'Nb_true': 2,
	  'Nl': 2,
	  'weight': tensor([0.0667, 0.0667, 0.0000, 0.0000, 0.0000, 0.0000])},
	 {'Na': {'C': 1, 'O': 2},
	  'Nb': 10,
	  'Nb_true': 8,
	  'Nl': 2,
	  'weight': tensor([0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.0000, 0.0000])}]

	info_element =
	{'C': {
		'Ecut': 200,
    	'Ne': 19,
    	'Nl': 3,
    	'Nu': [2, 2, 1],
    	'Rcut': 6,
    	'dr': 0.01,
    	'index': 0},
	 'O': {
		'Ecut': 200,
		'Ne': 19,
		'Nl': 3,
		'Nu': [3, 2, 1],
		'Rcut': 6,
		'dr': 0.01,
		'index': 1}}

	info_opt =
	{'cal_T': False,
	 'cal_smooth': False,
	 'lr': 0.01,
	 'max_steps': 200}
	"""


def get_info_max(info_stru, info_element):
	info_max = [None] * len(info_stru)
	for ist in range(len(info_stru)):
		Nt = info_stru[ist].Na.keys()
		info_max[ist] = addict.Dict()
		info_max[ist].Nt = len(Nt)
		info_max[ist].Na = max((info_stru[ist].Na[it] for it in Nt))
		info_max[ist].Nl = max([info_element[it].Nl for it in Nt])
		info_max[ist].Nm = max((util.Nm(info_element[it].Nl-1) for it in Nt))
		info_max[ist].Nu = max(itertools.chain.from_iterable([info_element[it].Nu for it in Nt]))
		info_max[ist].Ne = max((info_element[it].Ne for it in Nt))
		info_max[ist].Nb = info_stru[ist].Nb
	return info_max

	"""
	[{'Na': 2, 'Nb': 6, 'Ne': 19, 'Nl': 3, 'Nm': 5, 'Nt': 1, 'Nu': 2},
	 {'Na': 2, 'Nb': 6, 'Ne': 19, 'Nl': 3, 'Nm': 5, 'Nt': 1, 'Nu': 2}]
	"""	

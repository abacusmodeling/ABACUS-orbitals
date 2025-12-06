import util
import json
import numpy as np

def read_json(file_name):

	with open(file_name,"r") as file:
		input = file.read()
	input = json.loads(input)

	input_default =	{
		"element":{
				"Loss_thr": 0,
				"orb_num": np.inf
		},
		"optimize": [
			{
				"optimizer": "Adam",
				"kwargs": {},
				"cal_T": False,
				"norm": "element",
				"max_steps": 30000
			}
		],
		"C_init_info": {
			"init_from_file": False,
		},
		"V_info": {
			"init_from_file": True,
			"same_band": True
		},
		"radial": {
			"Ecut": 0.0,
			"smearing_sigma": 0.0
		}
	}
	util.set_dict_default(input, input_default)
	util.set_dict_default_elements(input["radial"], input["element"]["Nt_all"])

	info_element = util.Info()
	for info_attr,info_value in input["element"].items():
		info_element.__dict__[info_attr] = info_value
	if "Nu" in info_element.__dict__.keys():
		info_element.Nl = { it:len(Nu) for it,Nu in info_element.Nu.items() }
	elif "Nl" in info_element.__dict__.keys():
		info_element.Nu = { it:[0]*Nl for it,Nl in info_element.Nl.items() }

	return input["file_list"], info_element, input["weight"], input["optimize"], input["C_init_info"], input["V_info"], input["radial"]

	""" file_name
	{
		"file_list": {
			"origin": [
				"~/C_bulk/orb_matrix/test.0.dat",
				"~/CO2/orb_matrix/test.0.dat"
			],
			"linear": [
				[
					"~/C_bulk/orb_matrix/test.1.dat",
					"~/CO2/orb_matrix/test.1.dat"
				],
				[
					"~/C_bulk/orb_matrix/test.2.dat",
					"~/CO2/orb_matrix/test.2.dat"
				],
			]
		},
		"element": {
			"Nt_all": [ "C", "O" ],
			"Nu":   { "C":[2,2,1], "O":[3,2,1] },		# only for main.py
			"Nl":	{ "C": 3, "O": 3 },					# only for main_each.py
			"Loss_thr":		1E-10,						# only for main_each.py
			"orb_num":		13							# only for main_each.py
		},
		"weight":
		{
			"stru":	[1, 2.3],
			"bands_range":	[10, 15],					# "bands_range" and "bands_file" only once
			"bands_file":
			[
				"~/C_bulk/OUT.ABACUS/istate.info",
				"~/CO2/OUT.ABACUS/istate.info"
			]
		},
		"optimize":[
			{
				"optimizer": "Adam",
				"kwargs": {
					"lr": 0.01
				},
				"max_steps": 30000,
				"cal_T": false,
				"norm": "element" / "max" / "max_ist" / "one",
			}
		],
		"C_init_info": {								# only for main.py
			"init_from_file": false,
			"C_init_file": "~/CO/ORBITAL_RESULTS.txt",
			"opt_C_read": false
		},
		"V_info": {
			"init_from_file": true,
			"same_band": true
		},
		"radial": {
			"Rcut":           { "C": 6,    "O": 7    },
			"dr":             { "C": 0.01, "O": 0.01 },
			"Ecut":           { "C": 100,  "O": 100  },
			"smearing_sigma": { "C": 0.0,  "O": 0.0  };
		}
	}
	"""

	""" info_kst
		Nt_all		['C', 'O']
		Nu			{'C': [2, 2, 1], 'O': [3, 2, 1]}
		Nl			[3, 3, 3]
		Nst			3
		Nt			[['C'], ['C'], ['C', 'O']]
		Na			[{'C': 1}, {'C': 1}, {'C': 1, 'O': 2}]
		Nb			[6, 6, 10]
		Ne			{'C': 19, 'O': 19}
	"""
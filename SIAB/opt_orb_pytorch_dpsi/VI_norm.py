import util

class VI_Norm_Element:

	def set_norm_VI_origin(self, VI_origin):
		self.norm_VI_origin = []
		for VI in VI_origin:
			self.norm_VI_origin.append(util.update0(VI))

	def set_norm_VI_linear(self, VI_linear):
		self.norm_VI_linear = util.ND_list(len(VI_linear), element="list()")
		for i_type in range(len(VI_linear)):
			for VI in VI_linear[i_type]:
				self.norm_VI_linear[i_type].append(util.update0(VI))

	def get_norm_VI_origin(self, ist):
		return self.norm_VI_origin[ist]

	def get_norm_VI_linear(self, i_type, ist):
		return self.norm_VI_linear[i_type][ist]


class VI_Norm_Max:

	def set_norm_VI_origin(self, VI_origin):
		self.norm_VI_origin = 0
		for VI in VI_origin:
			self.norm_VI_origin = max(self.norm_VI_origin, VI.abs().max().item())

	def set_norm_VI_linear(self, VI_linear):
		self.norm_VI_linear = util.ND_list(len(VI_linear), element=0)
		for i_type in range(len(VI_linear)):
			for VI in VI_linear[i_type]:
				self.norm_VI_linear[i_type] = max(self.norm_VI_linear[i_type], VI.abs().max().item())

	def get_norm_VI_origin(self, ist):
		return self.norm_VI_origin

	def get_norm_VI_linear(self, i_type, ist):
		return self.norm_VI_linear[i_type]


class VI_Norm_Max_ist:

	def set_norm_VI_origin(self, VI_origin):
		self.norm_VI_origin = []
		for VI in VI_origin:
			self.norm_VI_origin.append(VI.abs().max().item())

	def set_norm_VI_linear(self, VI_linear):
		self.norm_VI_linear = util.ND_list(len(VI_linear), element="list()")
		for i_type in range(len(VI_linear)):
			for VI in VI_linear[i_type]:
				self.norm_VI_linear[i_type].append(VI.abs().max().item())

	def get_norm_VI_origin(self, ist):
		return self.norm_VI_origin[ist]

	def get_norm_VI_linear(self, i_type, ist):
		return self.norm_VI_linear[i_type][ist]


class VI_Norm_One:

	def set_norm_VI_origin(self, VI_origin):
		pass

	def set_norm_VI_linear(self, VI_linear):
		pass

	def get_norm_VI_origin(self, ist):
		return 1

	def get_norm_VI_linear(self, i_type, ist):
		return 1


def get_VI_norm_type(VI_norm_type):
	if VI_norm_type == "element":
		return VI_Norm_Element()
	elif VI_norm_type == "max":
		return VI_Norm_Max()
	elif VI_norm_type == "max_ist":
		return VI_Norm_Max_ist()
	elif VI_norm_type == "one":
		return VI_Norm_One()
	else:
		raise KeyError(VI_norm_type)
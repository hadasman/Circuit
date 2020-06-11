import pandas as pd
from math import log
import _pickle as cPickle
import pdb
import numpy as np
class Connectivity():

	def __init__(self, pyr_type, PV_rule, SOM_rule, **kwargs):
		'''
		NOTICE! In kwargs there must appear the variable names exactly as written here, together with their filename:
			- cell_type_to_gids
			- thalamic_locs
			- cell_details
			- cortex_connectivity
		'''

		expected_kwargs = ['cell_type_to_gids', 'thalamic_locs', 'cell_details', 'cortex_connectivity', 'thal_connections']
		assert len(expected_kwargs)==len(kwargs), 'Expected {} kwargs but got {}'.format(len(expected_kwargs), len(kwargs))
		assert all([i in kwargs for i in expected_kwargs]), 'At least 1 incorrect kwarg name!'

		self.pyr_type = pyr_type
		self.PV_rule  = PV_rule
		self.SOM_rule = SOM_rule

		cell_type_to_gids = cPickle.load(open(kwargs['cell_type_to_gids'],'rb')) 
		
		# Get all GIDs corresponding to given cell types
		self.pyr_GIDs = self.get_GIDs(cell_type_to_gids, self.pyr_type)
		self.PV_GIDs  = self.get_GIDs(cell_type_to_gids, self.PV_rule)
		self.SOM_GIDs = self.get_GIDs(cell_type_to_gids, self.SOM_rule)

		# Load more user-given files
		for var_name in kwargs:
			filename = kwargs[var_name]
			exec('self.' + var_name + ' = pd.read_pickle(\'%s\')'%filename)

	def get_GIDs(self, type_to_gid, rule):
		'''
		Get all GIDs in dataset, corresponding to given types.
			- type_to_gid: dictionary: {cell_type: [gids]}
			- rule: Can be array or string.
					* If array: [[types], 'include'/'exclude'] - list with cell types list as first element, 
					  and the rule as second element. If the rule is 'include' function returns GIDs 
					  corresponding to the types in the array; if rule is 'exclude' function returns 
					  all GIDs except those in the list. types elements can be the whole name, for specificity 
					  (i.e. 'L4_PC'); or part of it, for generality (i.e. 'PC').
					* If string: returns GIDs corresponding to the cell type represented by the the string.
		'''

		if type(rule) == str:
			GIDs = type_to_gid[rule]

		elif type(rule) == list:
			rule_types = rule[0]
			included_types = []

			if rule[1] == 'include':
				for t in type_to_gid:
					if any([i in t for i in rule_types]):
						included_types.append(t)

			elif rule[1] == 'exclude':
				for t in type_to_gid:
					if all([i not in t for i in rule_types]):
						included_types.append(t)

			GIDs = [j for i in [type_to_gid[t] for t in included_types] for j in i]

		else:
			raise Exception('Invalid input in get_GIDs function')

		return GIDs

	def choose_GID_between_freqs(self, potential_gids, freq1, freq2, min_freq=4000, df_dx=3.5):
		## (pyr_gids, PV_gids, freq1, freq2, min_freq=4000, df_dx=3.5, filenames=filenames)
		# df/dx is n units [octave/mm]
		
		min_freq_loc = min(self.thalamic_locs.x)

		def get_AxonLoc(axon_freq, min_freq_loc, min_freq=min_freq, df_dx=df_dx):
			dOctave 	=  log(axon_freq / min_freq, 2)# In octaves
			d_mm 		= dOctave / df_dx # In millimeters
			d_microne 	= d_mm * 1000 # In micro-meter
			freq_loc 	= min_freq_loc + d_microne

			return freq_loc

		def choose_GID(potential_gids, freq1_loc, freq2_loc, cell_details):

			mid_loc = np.mean([freq1_loc, freq2_loc])

			dists = [abs(cell_details.loc[gid].x - mid_loc) for gid in potential_gids]
			sorted_idx = [dists.index(i) for i in sorted(dists)]

			OK, i = 0, 0
			while not OK:
				chosen_gid = potential_gids[sorted_idx[i]]
				if chosen_gid in list(self.thal_connections.post_gid.values):
					OK = 1
				else:
					i += 1

			return chosen_gid

		freq1_loc = get_AxonLoc(freq1, min_freq_loc)
		freq2_loc = get_AxonLoc(freq2, min_freq_loc)
		
		chosen_gid = choose_GID(potential_gids, freq1_loc, freq2_loc, self.cell_details)

		return chosen_gid

	def choose_GID_by_post(self, potential_gids, post_gid):

		# post_gid_connections = self.cortex_connectivity[post_gid]

		inputs_to_post = [pre_gid for pre_gid in self.cortex_connectivity if post_gid in self.cortex_connectivity[pre_gid]]
		chosen_gids, chosen_n_contacts = [], {}

		for gid in inputs_to_post:
			if int(gid) in potential_gids:
				chosen_gids.append(int(gid))
				chosen_n_contacts[int(gid)] = self.cortex_connectivity[gid][post_gid]

		return chosen_gids, chosen_n_contacts

	def find_PresynGIDs(self, post_cell):

		connecting_gids = []
		for con in self.thal_connections.iterrows():
			if con[1].post_gid == post_cell:
				connecting_gids.append([con[1].pre_gid, con[1].contacts]) # [presynaptic gid, no. of contacts]

		return connecting_gids


			# Find thalamic GIDs connecting the the pyramidal cell
			# for con in thal_connections.iterrows():
			# 	if con[1].post_gid == cell_gid:
			# 		connecting_gids.append([con[1].pre_gid, con[1].contacts]) # [presynaptic gid, no. of contacts]

			# Get tha thalamic activation timings and no. of contacts on the pyramidal cell

	def find_numConnections():
		pass
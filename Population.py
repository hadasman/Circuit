import numpy as np
import os, pdb
import _pickle as cPickle
import pandas as pd
os.chdir('../MIT_spines')
from Cell import Cell
from neuron import h, gui
os.chdir('../Circuit')

from tkinter import messagebox

class Population():

	def __init__(self, population_name, morph_path, template_path, template_name, verbose=True):

		self.cells 				= {}
		self.population_name 	= population_name
		self.morph_path 		= morph_path
		self.template_path 		= template_path
		self.template_name 		= template_name
		self.verbose 			= verbose
		self.inputs 			= {}
		self.outputs 			= {}
		self.name_to_gid 		= {}
		self.loadTemplate()
		
		exec("self.template = h.%s"%template_name)

	def loadTemplate(self):
		"""
		Load template file. This should be done only once for each template type! No need to reload
			when instantiating new cells from the same template.

		This includes loading all .hoc files needed for template loading. The function expects certain file names,
			therefore if file is not found a messagebox pops up to warn user.
		"""

		# Change directory to template folder

		original_path = os.getcwd()
		os.chdir(self.template_path)

		h.load_file("stdrun.hoc")
		print('- Loading constants')		
		h.load_file('import3d.hoc')

		constants_loaded = h.load_file('constants.hoc')
		morphology_loaded = h.load_file('morphology_%s.hoc'%self.template_name)
		biophysics_loaded = h.load_file('biophysics_%s.hoc'%self.template_name)

		error = 'Can\'t find hoc file! Did you create it and call it by the correct name?'
		if not constants_loaded:
			_ = messagebox.showinfo(message=error, title='constants.hoc')
		if not morphology_loaded:
			_ = messagebox.showinfo(message=error, title='morphology.hoc')
		if not biophysics_loaded:
			_ = messagebox.showinfo(message=error, title='biophysics.hoc')


		if self.verbose:
			print('\n- Making %s template from .hoc file'%self.template_name)

		# h.load_file('%s.hoc'%self.template_name)
		h.load_file('template_%s.hoc'%self.template_name)

		# Return to original dir
		os.chdir(original_path)

	def addCell(self, add_synapses=False):

		# Change directory to template folder
		original_path = os.getcwd()
		os.chdir('%s'%self.template_path)


		if self.verbose:
			print('\n- Creating new instantiation of %s template'%self.template_name)

		idx = len(self.cells)
		cell_name = self.population_name + str(idx) 

		self.cells[cell_name] = {}

		self.cells[cell_name]['cell'] = self.template(add_synapses)

		self.cells[cell_name]['soma_v'] = h.Vector()
		self.cells[cell_name]['soma_v'].record(self.cells[cell_name]['cell'].soma[0](0.5)._ref_v)

		self.cells[cell_name]['axons'] 				= [i for i in self.cells[cell_name]['cell'].axon]
		self.cells[cell_name]['terminals'] 			= [i for i in self.cells[cell_name]['axons'] if len(i.children())==0]
		self.cells[cell_name]['apical_dendrites']  	= [i for i in self.cells[cell_name]['cell'].apic]
		self.cells[cell_name]['basal_dendrites']   	= [i for i in self.cells[cell_name]['cell'].dend]

		# Return to original dir
		os.chdir(original_path)

	def connectCells(self, pre_cell_name, post_branches, n_syns, dist, weight=0.5, delay=0, threshold=0):
		def synLocations(L, n_syns, dist):
			"""
			Returns synapse location according to given distribution, in arbitrary units (fraction [0-1]).
			Arbitrary units (relative to branch length) is the way NEURON receives synapse locations.
			"""
			ok = 0
			density = n_syns / L

			while not ok:
				if dist == 'uniform':
					ok = 1

					locs = sorted(np.arange(0, L, 1/density))
					locs = [i/L for i in locs]

					assert len(locs)==n_syns, ['Sanity check warning: unexpected locs length!', pdb.set_trace()]

				elif dist == 'random':
					ok = 1
									
					locs = np.random.rand(n_syns)
					locs = sorted(i/L for i in locs)
				
				elif dist == 'one':
					ok = 1

					locs = [0.5] * n_syns

				else:
					dist = input('Which synapse distribution for %s population? (uniform/random/one) '%self.population_name)

			return locs

		def addInhSynapses(branch, output_dict, delay=delay, weight=weight, threshold=threshold, Dep=0, Fac=0):
			'''locs and preysnaptics should be arrays ordered identically, for synapse location on postsynaptic branch (locs) and
			presynaptic partner (presynaptics)
			Inputs:
				- branch: postsynaptic branch on which to place the synapses
				- locs: location of each synapse on this branch.
				- presyn_seg: segment on the presynaptic section (a place on the axon, i.e. axon[0](0.2)), 
								from which the referenve to membrane voltage will be taken as input to branch. 
								Ordered with respect to <locs> array.
				- delay: synaptic delay between presynaptic threshold-crossing and postsynaptic activation. 
				- weight: weight of the synapse (in units of nS).
			'''

			locs = output_dict['locs']
			presyn_seg = [output_dict['stim_axon_segs'][np.random.randint(len(output_dict['stim_axon_segs']))] for i in range(len(locs))]
			synapses, netcons, conductances, currents = [], [], [], []

			for i in range(len(locs)):
				synapses.append(h.ProbUDFsyn2_lark(locs[i], sec = branch))
				conductances.append(h.Vector().record(synapses[-1]._ref_g))
				currents.append(h.Vector().record(synapses[-1]._ref_i))
				netcons.append(h.NetCon(presyn_seg[i]._ref_v, synapses[-1], sec=presyn_seg[i].sec))

				synapses[-1].tau_r 	= 0.18
				synapses[-1].tau_d 	= 5
				synapses[-1].e 		= -80
				synapses[-1].Dep 	= Dep # Put depression on PV -> pyr synapses
				synapses[-1].Fac 	= Fac
				synapses[-1].Use 	= 0.25
				synapses[-1].u0 	= 0
				synapses[-1].gmax 	= 0.001 # don't touch - weight conversion factor to (us) (later multiplied by the conductance in nS)

				netcons[-1].weight[0] = weight # In units nS
				netcons[-1].delay 	  = delay
				netcons[-1].threshold = threshold

			output_dict['synapses'] = synapses
			output_dict['netcons'] = netcons
			output_dict['syn_g'] = conductances
			output_dict['syn_i'] = currents

			return output_dict

		post_names = [i.name().split('.')[1] for i in post_branches]
		
		# Connect to axon side 1
		pre_branches = [axon(1) for axon in self.cells[pre_cell_name]['terminals']]
		self.outputs[pre_cell_name] = {post_branch: {'locs': [], 
										'stim_axon_segs': [], 
										'synapses': [], 
										'netcons': [],
										'syn_g': []}  for post_branch in post_names}

		for post_branch in self.outputs[pre_cell_name]:
			post_branch_obj = [i for i in post_branches if post_branch in i.name()][0]
			self.outputs[pre_cell_name][post_branch]['locs'] = synLocations(post_branch_obj.L, n_syns, dist)
			self.outputs[pre_cell_name][post_branch]['stim_axon_segs']    = pre_branches
			# Think about weight, delay (synaptic), threshold and noise (stochastic firing)
			self.outputs[pre_cell_name][post_branch] = addInhSynapses(post_branch_obj, 
																		self.outputs[pre_cell_name][post_branch],
																		Dep=0)

	def moveCell(self, cell, dx, dy, dz):
		'''
		Moves cell morphology.
		When changing 3d coordinates of reconstructed cells, only change the soma location! 
		The rest will follow because it's connected: when a section is moved all of its children apparently move too.
		'''

		soma = cell.soma[0]
		for i in range(soma.n3d()):
			h.pt3dchange(i, soma.x3d(i)+dx, soma.y3d(i)+dy, soma.z3d(i)+dz, soma.diam, sec=soma)

	def addInput(self, cell_name, cell_gid, thalamic_activations_filename = '', thalamic_connections_filename='', weight=0.4, axon_gids=None):

		def checkValidFilename(filename, cond='', which_standard=None):	
			ok = 0	
			original_path = os.getcwd()

			path = filename.rpartition('/')[0]
			file = filename.rpartition('/')[-1]
			os.chdir(path)
			
			while not ok:

				# This bloc is inside the while loop because condition should change according to new filename
				if cond == 'exist':
					condition = 'not \'%s\''%filename		
				elif cond == 'in_dir':
					condition = '\'%s\' not in os.listdir()'%file
				else:
					print('Filename validity check not provided with condition, exiting...')
					return filename

				if eval(condition):
					filename = input('Oops, file not %s!\nEnter valid thalamic input file name with ext. (current dir: %s): '%(cond, os.getcwd()))
				else:
					ok = 1

			os.chdir(original_path)
			return filename
		
		def get_activations(filename, axon_gids):
			temp_data = [i.strip().split() for i in open(filename).readlines()]
			activations = [] 
			for i in range(len(temp_data)): 
				if temp_data[i][0].replace('.', '').isdigit():
					if not axon_gids: # If specific GIDs not given by user
						activations.append([float(temp_data[i][0]), int(float(temp_data[i][1]))]) 
					else: 			  # Take only activations for user given GIDs
						if int(float(temp_data[i][1])) in axon_gids:
							activations.append([float(temp_data[i][0]), int(float(temp_data[i][1]))]) 
			return activations

		def getInputs(cell_gid, thalamic_activations, thalamic_connections_filename):
			thal_connections = pd.read_pickle(thalamic_connections_filename)
			connecting_gids = []
			
			# Find thalamic GIDs connecting the the pyramidal cell
			for con in thal_connections.iterrows():
				if con[1].post_gid == cell_gid:
					connecting_gids.append([con[1].pre_gid, con[1].contacts]) # [presynaptic gid, no. of contacts]

			# Get tha thalamic activation timings and no. of contacts on the pyramidal cell
			cell_inputs = {i[0]: {'contacts': i[1], 'times': []} for i in connecting_gids}
			for i in thalamic_activations:
				if i[1] in cell_inputs.keys():
					cell_inputs[i[1]]['times'].append(i[0])

			return cell_inputs

		def synLocationsAll(branches, n_syns):

			branch_locs = {}
			n_branches = len(branches)
				
			# Select branch and location at random
			for i in range(n_syns):
				rand_branch_idx  = np.random.randint(n_branches)
				rand_branch 	 = branches[rand_branch_idx]
				rand_branch_name = rand_branch.name().split('].')[-1]
				rand_loc 		 = np.random.rand()
				
				if rand_branch_name in branch_locs.keys():
					branch_locs[rand_branch_name]['locs'].append(rand_loc)
				else:
					branch_locs[rand_branch_name] 				= {}
					branch_locs[rand_branch_name]['locs'] 	    = [rand_loc]
					branch_locs[rand_branch_name]['branch_obj'] = rand_branch

			return branch_locs

		def addExcSynapses(contact_locs_dict, stim_times, weight=weight, Dep=0, Fac=0):
			# contact_locs_dict = {branch_name: {'locs': [locs], 'branch_obj': branch_obj}}			

			synapses, netcons, g_AMPA, g_NMDA, i_AMPA, i_NMDA = [], [], [], [], [], []

			# Create synapse
			for branch_name in contact_locs_dict: 
				branch = contact_locs_dict[branch_name]['branch_obj']
				
				for loc in contact_locs_dict[branch_name]['locs']: 
					synapses.append(h.ProbAMPANMDA_EMS(loc, sec=branch))
					synapses[-1].Dep = Dep
					synapses[-1].Fac = Fac
					g_AMPA.append(h.Vector().record(synapses[-1]._ref_g_AMPA))
					g_NMDA.append(h.Vector().record(synapses[-1]._ref_g_NMDA))
					i_AMPA.append(h.Vector().record(synapses[-1]._ref_i_AMPA))
					i_NMDA.append(h.Vector().record(synapses[-1]._ref_i_NMDA))

			# NetCon: "The source may be a NULLObject. In this case events can only occur by calling event() from hoc"
			for syn in synapses:
				netcons.append(h.NetCon(None, syn))
				netcons[-1].weight[0] = weight

			return synapses, netcons, g_AMPA, g_NMDA, i_AMPA, i_NMDA

		# Check filenames are valid
		thalamic_connections_filename = checkValidFilename(thalamic_connections_filename, cond='exist')
		thalamic_connections_filename = checkValidFilename(thalamic_connections_filename, cond='in_dir')

		thalamic_activations_filename = checkValidFilename(thalamic_activations_filename, cond='exist')
		thalamic_activations_filename = checkValidFilename(thalamic_activations_filename, cond='in_dir')

		# Open thalamic activations file and process data
		thalamic_activations = get_activations(thalamic_activations_filename, axon_gids)

		# For the given cell, find: presynaptic thalamic GIDs, no. of contacts each one makes and their activation timings
		print('\n- Finding presynaptic contacts for cell {}'.format(cell_name))
		cell_inputs = getInputs(cell_gid, thalamic_activations, thalamic_connections_filename)
		
		# Add thalamic activation as synapses to cell
		cell_obj 	 = self.cells[cell_name]['cell']
		branches 	 = self.cells[cell_name]['basal_dendrites']
		# branches 	 = self.cells[cell_name]['basal_dendrites'] + self.cells[cell_name]['apical_dendrites']
		# branches = [self.cells[cell_name]['cell'].soma[0]]

		# CHECK THIS!
		print('\n- Adding thalamo-cortical synapses (on BASAL dendrites)')
		self.inputs[cell_name] = {}	
		for thal_gid in cell_inputs:
			n_syns 		= cell_inputs[thal_gid]['contacts']
			stim_times 	= cell_inputs[thal_gid]['times']

			contact_locs_dict = synLocationsAll(branches, n_syns)

			temp_synapses, temp_netcons, temp_g_AMPA, temp_g_NMDA, temp_i_AMPA, temp_i_NMDA = addExcSynapses(contact_locs_dict, stim_times,  																							 
																						   Dep=0, Fac=0)				
			self.inputs[cell_name]['thalamic_gid_{}'.format(thal_gid)] = {}
			self.inputs[cell_name]['thalamic_gid_{}'.format(thal_gid)]['synapses']   = temp_synapses
			self.inputs[cell_name]['thalamic_gid_{}'.format(thal_gid)]['netcons']    = temp_netcons
			self.inputs[cell_name]['thalamic_gid_{}'.format(thal_gid)]['stim_times'] = stim_times
			self.inputs[cell_name]['thalamic_gid_{}'.format(thal_gid)]['locations']  = contact_locs_dict
			self.inputs[cell_name]['thalamic_gid_{}'.format(thal_gid)]['g_AMPA']   = temp_g_AMPA
			self.inputs[cell_name]['thalamic_gid_{}'.format(thal_gid)]['g_NMDA']   = temp_g_NMDA
			self.inputs[cell_name]['thalamic_gid_{}'.format(thal_gid)]['i_AMPA']   = temp_i_AMPA
			self.inputs[cell_name]['thalamic_gid_{}'.format(thal_gid)]['i_NMDA']   = temp_i_NMDA














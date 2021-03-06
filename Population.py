import numpy as np
import os, pdb
import _pickle as cPickle
import pandas as pd
# Make sure deleteing the netx 2 lines doesn't make a difference! 
# Then you can move all my modules to "my_modules" folder and import them using from my_modules.Population import Population
# os.chdir('../MIT_spines')
# from Cell import Cell
from neuron import h, gui
os.chdir('../Circuit')

class Population():

	def __init__(self, population_name, morph_path, template_path, template_name, verbose=False, recording_dt=h.dt):

		self.cells 				= {}
		self.population_name 	= population_name
		self.morph_path 		= morph_path
		self.template_path 		= template_path
		self.template_name 		= template_name
		self.verbose 			= verbose
		self.inputs 			= {}
		self.cell_inputs		= {}
		self.outputs 			= {}
		self.name_to_gid 		= {}
		self.recording_dt 		= recording_dt
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
		if self.verbose: print('- Loading constants')		
		h.load_file('import3d.hoc')

		constants_loaded = h.load_file('constants.hoc')
		morphology_loaded = h.load_file('morphology_%s.hoc'%self.template_name)

		biophysics_loaded = h.load_file('biophysics_%s.hoc'%self.template_name)

		error = 'Can\'t find hoc file! Did you create it and call it by the correct name?'
		if not constants_loaded:
			print('WARNING: {} hoc file not loaded! Did you create it and call it by the correct name?'.format(constants))
		if not morphology_loaded:
			print('WARNING: {} hoc file not loaded! Did you create it and call it by the correct name?'.format(morphology))
		if not biophysics_loaded:
			# pdb.set_trace()
			print('WARNING: {} hoc file not loaded! Did you create it and call it by the correct name?'.format(biophysics))


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
		self.cells[cell_name]['soma_v'].record(self.cells[cell_name]['cell'].soma[0](0.5)._ref_v, self.recording_dt)

		self.cells[cell_name]['soma'] 				= self.cells[cell_name]['cell'].soma[0]
		self.cells[cell_name]['axons'] 				= [i for i in self.cells[cell_name]['cell'].axon]
		self.cells[cell_name]['terminals'] 			= [i for i in self.cells[cell_name]['axons'] if len(i.children())==0]
		if len(self.cells[cell_name]['cell'].apic) > 1:
			self.cells[cell_name]['apical_dendrites'] = [i for i in self.cells[cell_name]['cell'].apic]
		else:
			self.cells[cell_name]['apical_dendrites'] = []
		if len(self.cells[cell_name]['cell'].dend) > 1:
			self.cells[cell_name]['basal_dendrites'] = [i for i in self.cells[cell_name]['cell'].dend]
		else:
			self.cells[cell_name]['basal_dendrites'] = []

		# Return to original dir
		os.chdir(original_path)

	def connectCells(self, post_cell_name, pre_pop, pre_cell_name, post_branches, syn_specs, input_source='voltage', weight=0.5, delay=0, threshold=0, record_syns=False, std_params={'Dep':0, 'Fac':0, 'Use':0.25}):
		# ** self if post_pop ** 
		def get_synLocations(post_branches, syn_specs):

			def calc_synLocations(post_branches, n_syns, dist):
				"""
				Returns synapse location according to given distribution, in arbitrary units (fraction [0-1]).
				Arbitrary units (relative to branch length) is the way NEURON receives synapse locations.
				"""

				assert dist in ['uniform', 'random', 'one'], 'Which synapse distribution for %s population? (uniform/random/one) '%self.population_name
				
				n_branches = len(post_branches)
				branch_locs = {}
				
				if dist == 'uniform':
					raise Exception('uniform', '{} dist is under construction!'.format(dist))
					# density = n_syns / L
					# locs = sorted(np.arange(0, L, 1/density))
					# locs = [i/L for i in locs]

					# assert len(locs)==n_syns, ['Sanity check warning: unexpected locs length!', pdb.set_trace()]

				elif dist == 'random':
					
					for i in range(n_syns):

						# Randomly choose branch
						rand_branch_idx  = np.random.randint(n_branches)
						rand_branch 	 = post_branches[rand_branch_idx]
						rand_branch_name = rand_branch.name().split('].')[-1]
						
						# Randomly choose location
						rand_loc = np.random.rand()

						if rand_branch_name in branch_locs.keys():
							branch_locs[rand_branch_name]['locs'].append(rand_loc)
						else:
							branch_locs[rand_branch_name] 				= {}
							branch_locs[rand_branch_name]['locs'] 		= [rand_loc]
							branch_locs[rand_branch_name]['branch_obj'] = rand_branch								

					for key in branch_locs:
						branch_locs[key]['locs'] = sorted(branch_locs[key]['locs'])
				
				elif dist == 'one':
					single_branch_idx 	= np.random.randint(n_branches)
					single_branch 	  	= post_branches[single_branch_idx]
					single_branch_name 	= single_branch.name().split('].')[-1]
					
					branch_locs[single_branch_name] = {'branch_obj': single_branch, 'locs': [0.5]*n_syns}

				return branch_locs

			def load_synLocations(filename, post_branches):

				syn_locs = cPickle.load(open(filename, 'rb'))
				syn_locs = syn_locs[post_cell_name][pre_cell_name]

				syn_locs_dict = {bname: {'locs': [], 'branch_obj': []} for bname in syn_locs}

				for bname in syn_locs:
					syn_locs_dict[bname]['locs'] = syn_locs[bname]
					syn_locs_dict[bname]['branch_obj'] = [i for i in post_branches if bname in i.name()]

					assert len(syn_locs_dict[bname]['branch_obj']) == 1, 'More than 1 branch with the same name! (in Population/addInput/load_synLocations())'
					syn_locs_dict[bname]['branch_obj'] = syn_locs_dict[bname]['branch_obj'][0]

				return syn_locs_dict

			if type(syn_specs) != str:
				if type(syn_specs) == dict:
					dist   = syn_specs[pre_cell_name][0]
					n_syns = syn_specs[pre_cell_name][1]
				else:
					dist   = syn_specs[0] 
					n_syns = syn_specs[1]
				
				locs_dict = calc_synLocations(post_branches, n_syns, dist)
				assert np.sum([len(locs_dict[b]['locs']) for b in locs_dict])==n_syns, 'length of locs is not n_syns'

			elif syn_specs.endswith('.p'):
				filename = syn_specs
				locs_dict = load_synLocations(filename, post_branches)

			else:
				raise Exception('Unrecognized synapse specs! (in Population/connectCells/get_synLocations())')

			return locs_dict

		def addInhSynapses(input_source, locs, branch, output_dict, pre_branches, std_params, delay=delay, weight=weight, threshold=threshold):
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

			presyn_seg = [pre_branches[np.random.randint(len(pre_branches))] for i in range(len(locs))]
			
			synapses, netcons, conductances, currents = [], [], [], []

			for i in range(len(locs)):
				synapses.append(h.ProbUDFsyn2_lark(locs[i], sec = branch))
				conductances.append(h.Vector().record(synapses[-1]._ref_g, self.recording_dt))
				currents.append(h.Vector().record(synapses[-1]._ref_i, self.recording_dt))

				synapses[-1].tau_r 	= 0.18
				synapses[-1].tau_d 	= 5
				synapses[-1].e 		= -80
				synapses[-1].Dep 	= std_params['Dep'] # Put depression on PV -> pyr synapses
				synapses[-1].Fac 	= std_params['Fac']
				synapses[-1].Use 	= std_params['Use']
				synapses[-1].u0 	= 0
				synapses[-1].gmax 	= 0.001 # don't touch - weight conversion factor to (us) (later multiplied by the conductance in nS)


				if input_source == 'voltage':
					netcons.append(h.NetCon(presyn_seg[i]._ref_v, synapses[-1], sec=presyn_seg[i].sec))
					netcons[-1].delay 	  = delay
					netcons[-1].threshold = threshold					

				elif input_source == 'spike_times':
					netcons.append(h.NetCon(None, synapses[-1]))

				netcons[-1].weight[0] = weight # In units nS										
			
			output_dict['synapses'].append(synapses)
			output_dict['netcons'].append(netcons)

			if record_syns:
				output_dict['g_GABA'].append(conductances)
				output_dict['i_GABA'].append(currents)

			return output_dict
		
		# Connect to axon side 1
		pre_branches = [axon(1) for axon in pre_pop.cells[pre_cell_name]['terminals']]

		if post_cell_name not in list(self.cell_inputs.keys()):
			self.cell_inputs[post_cell_name] = {}

		self.cell_inputs[post_cell_name][pre_cell_name] = {'locs': [], 'synapses': [], 'netcons': [], 'g_GABA': [], 'i_GABA': []}

		locs_dict = get_synLocations(post_branches, syn_specs)

		for post_branch in locs_dict:

			post_branch_obj = locs_dict[post_branch]['branch_obj']
			locs            = locs_dict[post_branch]['locs']
			
			self.cell_inputs[post_cell_name][pre_cell_name] = addInhSynapses(input_source, locs, post_branch_obj, self.cell_inputs[post_cell_name][pre_cell_name], pre_branches, std_params)
			self.cell_inputs[post_cell_name][pre_cell_name]['locs'] = self.cell_inputs[post_cell_name][pre_cell_name]['locs'] + [[post_branch, l] for l in locs]

		flatten = lambda L: [j for i in L for j in i]
		self.cell_inputs[post_cell_name][pre_cell_name]['synapses'] = flatten(self.cell_inputs[post_cell_name][pre_cell_name]['synapses'])
		self.cell_inputs[post_cell_name][pre_cell_name]['netcons'] = flatten(self.cell_inputs[post_cell_name][pre_cell_name]['netcons'])

		if record_syns:
			self.cell_inputs[post_cell_name][pre_cell_name]['g_GABA'] = flatten(self.cell_inputs[post_cell_name][pre_cell_name]['g_GABA'])
			self.cell_inputs[post_cell_name][pre_cell_name]['i_GABA'] = flatten(self.cell_inputs[post_cell_name][pre_cell_name]['i_GABA'])

	def addInput(self, cell_name, thalamic_activations_filename = '', connecting_gids=[], weight=0.4, axon_gids=None, where_synapses=[], record_syns=False, delay=0, std_params={'Dep':0, 'Fac':0, 'Use':1}):

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
		
		def getInputs(thalamic_activations, connecting_gids):
			
			# Get tha thalamic activation timings and no. of contacts on the pyramidal cell
			cell_inputs = {i[0]: {'contacts': i[1], 'times': []} for i in connecting_gids}

			# After saving all activations to .p dict
			for a in thalamic_activations:
				if a in cell_inputs.keys():
					cell_inputs[a]['times'] = thalamic_activations[a]

			return cell_inputs

		def calc_synLocationsAll(branches, n_syns):

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

		def load_synLocations(syn_locs, sections):

			syn_locs_dict = {bname: {'locs': [], 'branch_obj': []} for bname in syn_locs}

			for bname in syn_locs:	
				branch = [i for i in sections if bname in i.name()]
				assert len(branch) == 1, 'Ambiguous branch name (in Population/addInput/load_synLocations())'
				branch = branch[0]

				syn_locs_dict[bname]['locs'] = syn_locs[bname]
				syn_locs_dict[bname]['branch_obj'] = branch

			return syn_locs_dict
				
		def addExcSynapses(contact_locs_dict, stim_times, std_params, weight=weight, delay=0):
			# contact_locs_dict = {branch_name: {'locs': [locs], 'branch_obj': branch_obj}}			

			synapses, netcons, g_AMPA, g_NMDA, i_AMPA, i_NMDA = [], [], [], [], [], []

			# Create synapse
			for branch_name in contact_locs_dict: 
				branch = contact_locs_dict[branch_name]['branch_obj']
				
				for loc in contact_locs_dict[branch_name]['locs']: 
					synapses.append(h.ProbAMPANMDA_EMS(loc, sec=branch))
					if np.random.rand()>-1:
						synapses[-1].Dep = std_params['Dep']
						synapses[-1].Fac = std_params['Fac']
						synapses[-1].Use = std_params['Use']
					else:
						synapses[-1].Dep = 0
						synapses[-1].Fac = 0
						synapses[-1].Use = 1
						pdb.set_trace()
					g_AMPA.append(h.Vector().record(synapses[-1]._ref_g_AMPA, self.recording_dt))
					g_NMDA.append(h.Vector().record(synapses[-1]._ref_g_NMDA, self.recording_dt))
					i_AMPA.append(h.Vector().record(synapses[-1]._ref_i_AMPA, self.recording_dt))
					i_NMDA.append(h.Vector().record(synapses[-1]._ref_i_NMDA, self.recording_dt))

			# NetCon: "The source may be a NULLObject. In this case events can only occur by calling event() from hoc"
			for syn in synapses:
				netcons.append(h.NetCon(None, syn))
				netcons[-1].weight[0] = weight
				netcons[-1].delay = delay

			return synapses, netcons, g_AMPA, g_NMDA, i_AMPA, i_NMDA

		# Check filenames are valid
		thalamic_activations_filename = checkValidFilename(thalamic_activations_filename, cond='exist')
		thalamic_activations_filename = checkValidFilename(thalamic_activations_filename, cond='in_dir')

		thalamic_activations = cPickle.load(open(thalamic_activations_filename, 'rb'))
		if 'ITI' in thalamic_activations:
			thalamic_activations = thalamic_activations['activations'] # This should be in the new version of thalamic activations, the previous one was a dict with only axons as keys
		
		
		cell_inputs = getInputs(thalamic_activations, connecting_gids)
		
		# Add thalamic activation as synapses to cell
		cell_obj 	 = self.cells[cell_name]['cell']

		branches = []
		if '.p' in where_synapses: # if arg is a filename
			syn_locs = cPickle.load(open(where_synapses, 'rb'))
			sections = self.cells[cell_name]['basal_dendrites'] + self.cells[cell_name]['apical_dendrites'] + [self.cells[cell_name]['soma']] + self.cells[cell_name]['axons']

			assert sorted([int(i.split('_')[-1]) for i in syn_locs[cell_name].keys()]) == sorted(cell_inputs.keys()), [print('Incompatible thalamic synapse locations and activations!'), pdb.set_trace()]
		else:
			for w in where_synapses:
				if w=='soma':
					branches.append(self.cells[cell_name][w])
				else:
					branches = branches + self.cells[cell_name][w] # example: where_synapses = basal_dendrites/apical_dendrites/soma

		self.inputs[cell_name] = {}	
		for thal_gid in cell_inputs:

			n_syns 		= cell_inputs[thal_gid]['contacts']
			stim_times 	= cell_inputs[thal_gid]['times']

			if '.p' in where_synapses:
				contact_locs_dict = load_synLocations(syn_locs[cell_name]['thalamic_gid_{}'.format(thal_gid)], sections)
			else:
				contact_locs_dict = calc_synLocationsAll(branches, n_syns)

			temp_synapses, temp_netcons, temp_g_AMPA, temp_g_NMDA, temp_i_AMPA, temp_i_NMDA = addExcSynapses(contact_locs_dict, stim_times,  																							 
																						   std_params, weight=weight)				

			self.inputs[cell_name]['thalamic_gid_{}'.format(thal_gid)] = {}
			self.inputs[cell_name]['thalamic_gid_{}'.format(thal_gid)]['synapses']   = temp_synapses
			self.inputs[cell_name]['thalamic_gid_{}'.format(thal_gid)]['netcons']    = temp_netcons
			self.inputs[cell_name]['thalamic_gid_{}'.format(thal_gid)]['stim_times'] = stim_times
			self.inputs[cell_name]['thalamic_gid_{}'.format(thal_gid)]['locations']  = contact_locs_dict

			if record_syns:
				self.inputs[cell_name]['thalamic_gid_{}'.format(thal_gid)]['g_AMPA']   = temp_g_AMPA
				self.inputs[cell_name]['thalamic_gid_{}'.format(thal_gid)]['g_NMDA']   = temp_g_NMDA
				self.inputs[cell_name]['thalamic_gid_{}'.format(thal_gid)]['i_AMPA']   = temp_i_AMPA
				self.inputs[cell_name]['thalamic_gid_{}'.format(thal_gid)]['i_NMDA']   = temp_i_NMDA

			else:

				self.inputs[cell_name]['thalamic_gid_{}'.format(thal_gid)]['g_AMPA']   = []
				self.inputs[cell_name]['thalamic_gid_{}'.format(thal_gid)]['g_NMDA']   = []
				self.inputs[cell_name]['thalamic_gid_{}'.format(thal_gid)]['i_AMPA']   = []
				self.inputs[cell_name]['thalamic_gid_{}'.format(thal_gid)]['i_NMDA']   = []

	def moveCell(self, cell, dx, dy, dz):
		'''
		Moves cell morphology.
		When changing 3d coordinates of reconstructed cells, only change the soma location! 
		The rest will follow because it's connected: when a section is moved all of its children apparently move too.
		'''

		soma = cell.soma[0]
		for i in range(soma.n3d()):
			h.pt3dchange(i, soma.x3d(i)+dx, soma.y3d(i)+dy, soma.z3d(i)+dz, soma.diam, sec=soma)

	def dumpSomaVs(self, path, thalamic_activations_filename, dump_type='', inh_inputs=None, job_id=None):
		'''
		Inputs:
			- t: hoc vector of recorded time (h._ref_t)
			- filename: file name (including path), in which to save the spike times
		
		Returns:
		Dictionary: {cell_name:{
								'soma_v': hoc vector with recorded soma voltages of the cell named <cell_name>
								'gid': assigned gid (from BB database) of cell named <cell_name>
								'spiks_times_func_as_string': function to extract spike times from 'soma_v', 
															  according to threshold given as argument to this function. 
															  This is a lambda function represented as a STRING, because 
															  lambda functions cannot be pickled. After loading this 
															  variable, call eval() on the string to use the function.
								}
					}
		'''

		def chooseFilename(job_id):

			cell_name = self.population_name
			input_name = thalamic_activations_filename.split('/')[-1]
			
			if job_id: # choose name by unique job id ('S' + running number)
				filename = str(job_id) + '_sim_' + cell_name + '_tstop_' + str(int(h.tstop)) + '_' + input_name   

			else: # choose name by running number
				filename = '0_sim_' + cell_name + '_tstop_' + str(int(h.tstop)) + '_' + input_name   
				if filename[-2:] != '.p':
					filename = filename + '.p'
				
				n_iters = 0
				OK = 0
				while not OK:
					if filename in os.listdir(path):
						n_iters += 1
						filename = str(n_iters) + filename[1:]
					else:
						OK = 1

			return filename

		if dump_type == 'as_dict':
			soma_vs = {}

			# Save soma voltages
			soma_vs = {i: {'soma_v': self.cells[i]['soma_v'],
									'gid': self.name_to_gid[i]} 
					  			 for i in self.cells}

			# Save total conductances
			soma_vs = {i: {'soma_v': self.cells[i]['soma_v'],
									'gid': self.name_to_gid[i],
									'inputs': 
										{'stim_times': [self.inputs[i][axon]['stim_times'] for axon in self.inputs[i]],
										 'g_AMPA': np.sum([np.sum(self.inputs[i][axon]['g_AMPA'], axis=0) for axon in self.inputs[i]], axis=0),
										 'g_NMDA': np.sum([np.sum(self.inputs[i][axon]['g_NMDA'], axis=0) for axon in self.inputs[i]], axis=0),
										 'i_AMPA': np.sum([np.sum(self.inputs[i][axon]['i_AMPA'], axis=0) for axon in self.inputs[i]], axis=0),
										 'i_NMDA': np.sum([np.sum(self.inputs[i][axon]['i_NMDA'], axis=0) for axon in self.inputs[i]], axis=0)}} 
					  			for i in self.cells}
			
			# flatten = lambda L: [j for i in L for j in i]
			
			if inh_inputs:
				for i in self.cells:
					soma_vs[i]['inputs']['g_GABA'] = {}
					soma_vs[i]['inputs']['i_GABA'] = {}
					for inh_cell in inh_inputs[i]:

						# g_GABA = np.sum(inh_inputs[i][inh_cell]['g_GABA'], axis=0)
						# i_GABA = np.sum(inh_inputs[i][inh_cell]['i_GABA'], axis=0)


						soma_vs[i]['inputs']['g_GABA'][inh_cell] = inh_inputs[i][inh_cell]['g_GABA']
						soma_vs[i]['inputs']['i_GABA'][inh_cell] = inh_inputs[i][inh_cell]['i_GABA']

					# g_GABA = np.sum(flatten([inh_inputs[i][inh_cell]['g_GABA'] for inh_cell in inh_inputs[i]]), axis=0)
					# i_GABA = np.sum(flatten([inh_inputs[i][inh_cell]['i_GABA'] for inh_cell in inh_inputs[i]]), axis=0)

					# soma_vs[i]['inputs']['g_GABA'] = g_GABA
					# soma_vs[i]['inputs']['i_GABA'] = i_GABA

			filename = chooseFilename()
			cPickle.dump(soma_vs, open(path + '/' + filename, 'wb'))
		
		elif dump_type == 'as_obj':
			print('NOTICE: To save as Population object, deleting all HOC objects from {} population!'.format(self.population_name))
			sure_ = input('Are you sure [y/n]? ')
			if sure_=='y':
				# Delete all hoc objexts (except for Vectors), in order to pickle
				del self.template
				for cell in self.cells: 
					del self.cells[cell]['cell']
					del self.cells[cell]['basal_dendrites'] 
					del self.cells[cell]['apical_dendrites'] 
					del self.cells[cell]['soma'] 
					del self.cells[cell]['terminals'] 
					del self.cells[cell]['axons'] 

				for cell in self.inputs:
					for axon in self.inputs[cell]:
						del self.inputs[cell][axon]['synapses']
						del self.inputs[cell][axon]['netcons']

						for dend in self.inputs[cell][axon]['locations']:
							del self.inputs[cell][axon]['locations'][dend]['branch_obj']

				filename = chooseFilename()
				cPickle.dump(self, open(path + '/' + filename, 'wb'))
		print('Dictionary with {} soma voltages saved to {}'.format(self.population_name, filename))







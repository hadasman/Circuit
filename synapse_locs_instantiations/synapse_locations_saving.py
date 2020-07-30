import pandas as pd
import numpy as np
import _pickle as cPickle
import matplotlib.pyplot as plt
plt.ion()

import pdb, os, sys

from neuron import gui, h
from tqdm import tqdm
from copy import copy

from Population import Population
from Stimulus import Stimulus
from Connectivity import Connectivity
from Parameter_Initialization import * # Initialize parameters before anything else!


def get_GIDs(upload_from, chosen_GID_filenames, connecting_gids_filename):

	chosen_GIDs = {}

	for f in chosen_GID_filenames:
		chosen_GIDs[f]   = cPickle.load(open('{}/{}'.format(upload_from, chosen_GID_filenames[f]), 'rb'))
	
	if syn_type == 'thalamic':
		thalamic_GIDs  = cPickle.load(open('{}/{}'.format(upload_from, connecting_gids_filename), 'rb'))
		return chosen_GIDs, thalamic_GIDs
	elif syn_type == 'cortical':
		chosen_PV_n_contacts  = cPickle.load(open('{}/{}'.format(upload_from, connecting_gids_filename), 'rb'))
		return chosen_GIDs, chosen_PV_n_contacts

def CreatePopulations(n_pyr=0, n_PV=0, n_SOM=0):
	Pyr_pop, PV_pop, SOM_pop = [None]*3

	if n_pyr > 0:
		print('\n==================== Creating pyramidal cell (n = {}) ===================='.format(n_pyr))
		Pyr_pop = Population('Pyr', pyr_morph_path, pyr_template_path, pyr_template_name, verbose=False)
		Pyr_pop.addCell()
		Pyr_pop.name_to_gid['Pyr0'] = chosen_GIDs['Pyr']

	if n_PV > 0:
		print('\n==================== Creating PV population (n = {}) ===================='.format(n_PV))
		PV_pop = Population('PV', PV_morph_path, PV_template_path, PV_template_name, verbose=False)
		for i in tqdm(range(n_PV)):
			PV_cell_name = 'PV%i'%i
			PV_pop.addCell()
			PV_pop.name_to_gid[PV_cell_name] = chosen_GIDs['PV'][i]
			PV_pop.moveCell(PV_pop.cells[PV_cell_name]['cell'], (i*350)-(100*(n_PV+1)), -500, 0) # Morphology Visualization

	if n_SOM > 0:
		print('\n==================== Creating SOM population (n = {}) ===================='.format(n_SOM))
		SOM_pop = Population('SOM', SOM_morph_path, SOM_template_path, SOM_template_name, verbose=False)
		SOM_pop.addCell()	
		SOM_pop.name_to_gid['SOM0'] = chosen_GIDs['SOM']
		SOM_pop.moveCell(SOM_pop.cells['SOM0']['cell'], 0, -1000, 0) # Morphology Visualization

	return Pyr_pop, PV_pop, SOM_pop

def set_ThalamicInputs(Pyr_pop, PV_pop, SOM_pop, thalamic_GIDs):

	if Pyr_pop:

		where_Pyr_syns = ['basal_dendrites']		
		where_Pyr_syns_str = '{}'.format(str(where_Pyr_syns).split('[')[1].split(']')[0].replace(',', ' and').replace('\'',''))

		print('\n==================== Connecting thalamic inputs to Pyramidal cell (on {}, standard: {}Hz, input weight: {}uS) ===================='.format(where_Pyr_syns_str, activated_standard_freq, Pyr_input_weight)); sys.stdout.flush()
		Pyr_pop.addInput('Pyr0', record_syns=record_thalamic_syns, where_synapses=where_Pyr_syns, weight=Pyr_input_weight, thalamic_activations_filename=activated_filename, connecting_gids=thalamic_GIDs)

	elif PV_pop:
		where_PV_syns = ['basal_dendrites', 'apical_dendrites']
		where_PV_syns_str = '{}'.format(str(where_PV_syns).split('[')[1].split(']')[0].replace(',', ' and').replace('\'',''))

		print('\n==================== Connecting thalamic inputs to PV cells (on {}, standard: {}Hz, input weight: {}uS) ===================='.format(where_PV_syns_str, activated_standard_freq, PV_input_weight)); sys.stdout.flush()

		if PV_to_Pyr_source == 'voltage':
			for i, PV_cell_name in enumerate(tqdm(PV_pop.cells)):
				PV_pop.addInput(PV_cell_name, record_syns=record_thalamic_syns, where_synapses=where_PV_syns, weight=PV_input_weight, thalamic_activations_filename=activated_filename, connecting_gids=thalamic_GIDs[PV_pop.name_to_gid[PV_cell_name]])	

		elif PV_to_Pyr_source == 'spike_times':
			print('Loading PV spike times from file ({})'.format(filenames['PV_spike_times']))
			PV_spike_times = cPickle.load(open(filenames['PV_spike_times'], 'rb'))

			for PV_cell_name in PV_pop.cells:
				PV_pop.cells[PV_cell_name]['soma_v'] = PV_spike_times['cells'][PV_cell_name]['soma_v']

	elif SOM_pop:
		SOM_input_source = 'thalamic_input'

		where_SOM_syns = ['basal_dendrites']
		where_SOM_syns_str = '{}'.format(str(where_SOM_syns).split('[')[1].split(']')[0].replace(',', ' and').replace('\'',''))
		
		print('\n==================== Connecting thalamic inputs to SOM cell (on {}, standard: {}Hz, input weight: {}uS) ===================='.format(where_SOM_syns_str, activated_standard_freq, SOM_input_weight)); sys.stdout.flush()

		if SOM_input_source == 'thalamic_input':
			SOM_pop.addInput('SOM0', record_syns=record_thalamic_syns, where_synapses=where_SOM_syns, weight=SOM_input_weight, thalamic_activations_filename=activated_filename, connecting_gids=thalamic_GIDs) 
		
		elif SOM_input_source == 'Pyr':
			SOM_pop.connectCells('SOM0', Pyr_pop, 'Pyr0', 'voltage', [SOM_pop.cells['SOM0']['soma']], n_Pyr_to_SOM_syns, 'random', weight=SOM_input_weight, delay=SOM_input_delay)

activated_filename = filenames['thalamic_activations_6666']
activated_standard_freq = freq1*(str(freq1) in activated_filename) + freq2*(str(freq2) in activated_filename)
alternative_freq = freq1*(str(freq1) not in activated_filename) + freq2*(str(freq2) not in activated_filename)

upload_from = 'GIDs_instantiations/pyr_72851_between_6666_9600' # upload_from = False
filenames = {'PV': 'chosen_PV.p', 'SOM': 'chosen_SOM_high_input.p', 'Pyr': 'chosen_pyr.p'}

if len(sys.argv) > 1:
	syn_type = sys.argv[1]
	print('Creating and saving {} synapses'.format(syn_type))
else:
	syn_type = 'cortical' #'thalamic'

if syn_type == 'thalamic':
	# ===============================================  Choose GIDs  ===============================================
	print('\n========== Choosing GIDs from BlueBrain Database (pyramidal between {}Hz and {}Hz) =========='.format(freq1, freq2))

	which_cell 						= 'SOM'
	chosen_GID_filename 			= filenames['SOM']
	connecting_gids_filename 		= {which_cell: 'connecting_gids_to_SOM_high_input.p'}
	chosen_GIDs, thalamic_GIDs = get_GIDs(upload_from, chosen_GID_filename, connecting_gids_filename)

	# ===============================================  Create Cell Populations  ===============================================
	n_pyr = 1 * (which_cell == 'Pyr')
	n_PV  = 1 * (which_cell == 'PV')
	n_SOM = 1 * (which_cell == 'SOM')
	Pyr_pop, PV_pop, SOM_pop = CreatePopulations(n_pyr=n_pyr, n_PV=n_PV, n_SOM=n_SOM)

	set_ThalamicInputs(Pyr_pop, PV_pop, SOM_pop, thalamic_GIDs)

	assert len([i for i in [Pyr_pop, PV_pop, SOM_pop] if i])==0, 'Too many populations! Only 1 allowed'
	pop = [i for i in [Pyr_pop, PV_pop, SOM_pop] if i][0]
	locs_dict = {cell: {} for cell in pop.cells}

	for cell in pop.cells:
		
		for axon in pop.inputs[cell]:
			locs_dict[cell][axon] = {}

			for branch in pop.inputs[cell][axon]['locations']:
				locs_dict[cell][axon][branch] = pop.inputs[cell][axon]['locations'][branch]['locs']

	saving_filename = input('Choose a filename for thalamic synapses to {} (without \'.p\') '.format(which_cell))

elif syn_type == 'cortical':
	where_post_syns = None
	# ===============================================  Choose GIDs  ===============================================
	print('\n========== Choosing GIDs from BlueBrain Database (pyramidal between {}Hz and {}Hz) =========='.format(freq1, freq2))

	pre_cell  = 'PV'
	post_cell = 'Pyr'

	chosen_GID_filenames = {pre_cell: filenames[pre_cell], post_cell: filenames[post_cell]}
	chosen_GIDs, chosen_PV_n_contacts = get_GIDs(upload_from, chosen_GID_filenames, 'chosen_PV_n_contacts.p')

	n_pyr = ('Pyr' in [pre_cell, post_cell]) * 1
	n_PV  = ('PV' in [pre_cell, post_cell]) * 18
	n_SOM = ('SOM' in [pre_cell, post_cell]) * 1

	Pyr_pop, PV_pop, SOM_pop = CreatePopulations(n_pyr=n_pyr, n_PV=n_PV, n_SOM=n_SOM)

	where_PV_to_Pyr = ['soma']
	where_SOM_to_PV = ['soma', 'basal_dendrites']
	where_SOM_to_Pyr = ['apical_dendrites']


	# PV ==> Pyr (either from PV spike times or PV soma voltage)
	if PV_pop and Pyr_pop:
		where_PV_to_Pyr = ['soma']
		print('Connecting PV population to Pyramidal cell (connection weight: {}uS)'.format(PV_to_Pyr_weight))	

		for Pyr_cell in Pyr_pop.cells:
			# PV_to_Pyr_post_secs = [Pyr_pop.cells['Pyr0']['soma']]
			PV_to_Pyr_post_secs = []
			for i in where_PV_to_Pyr:
				if hasattr(Pyr_pop.cells[Pyr_cell][i], '__len__'):
					PV_to_Pyr_post_secs = PV_to_Pyr_post_secs + Pyr_pop.cells[Pyr_cell][i]
				else:
					PV_to_Pyr_post_secs = PV_to_Pyr_post_secs + [Pyr_pop.cells[Pyr_cell][i]]
			
			for PV_cell_name in tqdm(PV_pop.cells):
				temp_PV_gid = PV_pop.name_to_gid[PV_cell_name]
				temp_n_syns = chosen_PV_n_contacts[temp_PV_gid]
				Pyr_pop.connectCells(Pyr_cell, PV_pop, PV_cell_name, PV_to_Pyr_post_secs, temp_n_syns, 'random', record_syns=record_PV_syns, input_source=PV_to_Pyr_source, weight=PV_to_Pyr_weight, delay=PV_output_delay, threshold=spike_threshold) # Adds self.connections to Population				

		print('\n***WARNING: Assuming isopotential soma and perisomatic PV connections: all PV synapses are placed on soma(0.5)')
		if where_post_syns:
			raise Exception('About to override locations!')
		else:
			where_post_syns = where_PV_to_Pyr

	# SOM ==> PV
	if SOM_pop and PV_pop:
		where_SOM_to_PV = ['soma', 'basal_dendrites']

		print('Connecting SOM population to PV Population (connection weight: {}uS)'.format(SOM_to_PV_weight))
		for PV_cell in PV_pop.cells:
			# SOM_to_PV_post_secs = [PV_pop.cells[PV_cell]['soma']] + PV_pop.cells[PV_cell]['basal_dendrites']
			SOM_to_PV_post_secs = []
			for i in where_SOM_to_PV:
				if hasattr(PV_pop.cells[PV_cell][i], '__len__'):
					SOM_to_PV_post_secs = SOM_to_PV_post_secs + PV_pop.cells[PV_cell][i]
				else:
					SOM_to_PV_post_secs = SOM_to_PV_post_secs + [PV_pop.cells[PV_cell][i]]
			PV_pop.connectCells(PV_cell, SOM_pop, 'SOM0', SOM_to_PV_post_secs, n_syns_SOM_to_PV, 'random', record_syns=record_SOM_syns, input_source='voltage', weight=SOM_to_PV_weight, delay=SOM_output_delay, threshold=spike_threshold)

		if where_post_syns:
			raise Exception('About to override locations!')
		else:
			where_post_syns = where_SOM_to_PV


	# SOM ==> Pyr
	if SOM_pop and Pyr_pop:
		where_SOM_to_Pyr = ['apical_dendrites']
		print('Connecting SOM population to Pyramidal Population (connection weight: {}uS)'.format(SOM_to_Pyr_weight))
		for Pyr_cell in Pyr_pop.cells:
			SOM_to_Pyr_post_secs = []

			for i in where_SOM_to_Pyr:
				if hasattr(Pyr_pop.cells[Pyr_cell][i], '__len__'):
					SOM_to_Pyr_post_secs = SOM_to_Pyr_post_secs + Pyr_pop.cells[Pyr_cell][i]
				else:
					SOM_to_Pyr_post_secs = SOM_to_Pyr_post_secs + [Pyr_pop.cells[Pyr_cell][i]]

			Pyr_pop.connectCells(Pyr_cell, SOM_pop, 'SOM0', SOM_to_Pyr_post_secs, n_syns_SOM_to_Pyr, 'random', record_syns=record_SOM_syns, input_source='voltage', weight=SOM_to_Pyr_weight, delay=SOM_output_delay, threshold=spike_threshold)

		if where_post_syns:
			raise Exception('About to override locations!')
		else:
			where_post_syns = where_SOM_to_Pyr


	assert len([i for i in [Pyr_pop, PV_pop, SOM_pop] if i])==2, 'Ambiguous number of populations!'
	pre_pop = [j for j in [i for i in [Pyr_pop, PV_pop, SOM_pop] if i] if j.population_name==pre_cell][0]
	post_pop = [j for j in [i for i in [Pyr_pop, PV_pop, SOM_pop] if i] if j.population_name==post_cell][0]


	locs_dict = {post: {pre: {} for pre in pre_pop.cells} for post in post_pop.cells}

	for post in post_pop.cells:
		for pre in pre_pop.cells:
			
			all_branches = list(set([i[0] for i in post_pop.cell_inputs[post][pre]['locs']]))
			locs_dict[post][pre]  = {bname: [] for bname in all_branches}

			for B in post_pop.cell_inputs[post][pre]['locs']:
				branch_name = B[0]
				loc = B[1]
				locs_dict[post][pre][branch_name].append(loc)
		
	saving_filename = input('Choose a filename for {} to {} synapses on {} (without \'.p\') '.format(pre_cell, post_cell, where_post_syns))

cPickle.dump(locs_dict, open('synapse_locs_instantiations/{}.p'.format(saving_filename), 'wb'))






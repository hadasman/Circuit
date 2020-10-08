# import matplotlib
# matplotlib.use("TkAgg")
import pandas as pd
import numpy as np
import _pickle as cPickle
import matplotlib.pyplot as plt
import time as time_module
plt.ion()

import pdb, os, sys, multiprocessing

from neuron import gui, h
from math import log
from tqdm import tqdm
from scipy.stats import ttest_ind

from Population import Population
from Stimulus import Stimulus
from Connectivity import Connectivity
from Parameter_Initialization import * # Initialize parameters before anything else!
from plotting_functions import plotThalamicResponses, Wehr_Zador, PlotSomas, plotFRs

job_id = 'S' + str(len(os.listdir('local_simulations')))
os.makedirs('local_simulations/{}'.format(job_id)) 

assert os.getcwd().split('/')[-1] == 'Circuit', 'Wrong directory'

if len(sys.argv)>1:
	tstop = float(sys.argv[1])
	print('Setting tstop to {}ms'.format(tstop))

# ============================================  Define Functions & Constants  ============================================
'''
Plan:
	- V - now: 5000ms, with PV connections, 9600Hz => 6 spikes out of 9
	- V - 5000ms without PV connections, 6666Hz (maybe also 9600Hz) => 7 out of 9 [same synapses+PV: 5] (6666)
	- V - 10000ms without+with PV connections (for same synapses), 6666Hz (maybe also 9600Hz) => 13 (with PV) & 20 (without PV) out of 25
'''
print('Injecting {}nA to Pyramidal for Stabilization'.format(Pyr_IClamp_amp))


def get_GIDs(upload_from):

	if upload_from:
		chosen_GIDs = {}
		chosen_GIDs['pyr'] 			  = cPickle.load(open('{}/chosen_pyr.p'.format(upload_from), 'rb'))
		chosen_GIDs['PV'] 			  = cPickle.load(open('{}/chosen_PV.p'.format(upload_from), 'rb'))
		chosen_GIDs['SOM'] 			  = cPickle.load(open('{}/chosen_SOM_high_input.p'.format(upload_from), 'rb'))
		chosen_PV_n_contacts  		  = cPickle.load(open('{}/chosen_PV_n_contacts.p'.format(upload_from), 'rb'))

		thalamic_GIDs = {}
		thalamic_GIDs['to_pyr'] = cPickle.load(open('{}/connecting_gids_to_pyr.p'.format(upload_from), 'rb'))
		thalamic_GIDs['to_PV']  = cPickle.load(open('{}/connecting_gids_to_PV.p'.format(upload_from), 'rb'))
		thalamic_GIDs['to_SOM'] = cPickle.load(open('{}/connecting_gids_to_SOM_high_input.p'.format(upload_from), 'rb'))
	else:
		print('\nGetting GIDs from connectivity. This is gonna take a while...')

		def choose_GIDs(connectivity, freq1, freq2):

			chosen_GIDs = {}

			# Pyramidal
			chosen_GIDs['pyr'] = connectivity.choose_GID_between_freqs(connectivity.pyr_GIDs, freq1, freq2)

			# PV
			chosen_GIDs['PV'], chosen_PV_n_contacts = connectivity.choose_GID_by_post(connectivity.PV_GIDs, chosen_GIDs['pyr'])
			
			# SOM
			if SOM_from == 'freq':
				chosen_GIDs['SOM'] = connectivity.choose_GID_between_freqs(connectivity.SOM_GIDs, freq1, freq2)
			
			elif SOM_from == 'post':
				print('Getting SOM GIDs connecting to PV and Pyramidal separately')
				chosen_GIDs['SOM'], chosen_SOM_n_contacts = {}, {}

				chosen_GIDs['SOM']['to_pyr'], chosen_SOM_n_contacts['to_pyr'] = connectivity.choose_GID_by_post(connectivity.SOM_GIDs, chosen_GIDs['pyr'])
				
				chosen_GIDs['SOM']['to_PV'], chosen_SOM_n_contacts['to_PV'] = {}, {}
				for PV_gid in chosen_GIDs['PV']:
					chosen_GIDs['SOM']['to_PV'][PV_gid], chosen_SOM_n_contacts['to_PV'][PV_gid] = connectivity.choose_GID_by_post(connectivity.SOM_GIDs, PV_gid)

			return chosen_GIDs, chosen_PV_n_contacts, chosen_SOM_n_contacts

		connectivity = Connectivity(pyr_type, [not_PVs, 'exclude'], [SOM_types, 'include'], 
									cell_type_to_gids=filenames['cell_type_gids'],
									thalamic_locs=filenames['thalamic_locs'], 
									cell_details=filenames['cell_details'],
									cortex_connectivity=filenames['cortex_connectivity'],
									thal_connections=filenames['thalamic_connections'])


		chosen_GIDs, chosen_PV_n_contacts, chosen_SOM_n_contacts = choose_GIDs(connectivity, freq1, freq2)
		pdb.set_trace()
		# Find thalamic GIDs connecting the the pyramidal cell

		thalamic_GIDs = {}
		def worker_job(data):
			gid 	  = data[0]
			cell_type = data[1]

			return connectivity.find_PresynGIDs(gid), gid, cell_type

		pool = multiprocessing.Pool()
		thalamic_GIDs['to_PV'], thalamic_GIDs['to_pyr'], thalamic_GIDs['to_SOM'] = {}, {}, {}

		if SOM_from == 'freq':
			all_gids = [[chosen_GIDs['pyr'], 'pyr']] + \
			           [[i, 'PV'] for i in chosen_GIDs['PV']] + \
			           [[chosen_GIDs['SOM'], 'SOM']]
		elif SOM_from == 'post':
			all_gids = [[chosen_GIDs['pyr'], 'pyr']] + \
					   [[i, 'PV'] for i in chosen_GIDs['PV']] + \
					   [[j, 'SOM'] for i in chosen_GIDs['SOM']['to_PV'].values() for j in i] + \
					   [[i,'SOM'] for i in chosen_GIDs['SOM']['to_pyr']]
		
		for result in pool.imap(worker_job, all_gids):

			if result[2] == 'pyr':
				thalamic_GIDs['to_'+result[2]] = result[0]
			else:
				thalamic_GIDs['to_'+result[2]][result[1]] = result[0]
	
	return chosen_GIDs, chosen_PV_n_contacts, thalamic_GIDs

def get_thalamic_input_filename(thalamic_params):
	fname_check = lambda fname, fchecks: all([f in fname for f in fchecks])

	if thalamic_params['type'] == 'pairs':	
		path_ = 'thalamocortical_Oren/CreateThalamicInputs/test_times/tone_pairs'	
		activated_filename = [i for i in os.listdir(path_) if fname_check(i, [thalamic_params['freq'], '.p', 'ITI_%s'%thalamic_params['ITI'], 'IPI_%s'%thalamic_params['IPI']])]

	elif thalamic_params['type'] == 'single':
		path_ = 'thalamocortical_Oren/CreateThalamicInputs/test_times/single_tone'
		activated_filename = [i for i in os.listdir(path_) if fname_check(i, [thalamic_params['freq'], '.p', 'IPI_%s'%thalamic_params['IPI']])]

	assert len(activated_filename) == 1, ['Nonspecific thalamic times filename!', pdb.set_trace()] 
	activated_filename = '{}/{}'.format(path_, activated_filename[0])
	
	filenames['thalamic_activations'] = activated_filename

	print('Activating input file: \"{}\"'.format(activated_filename))

	return activated_filename

def CreatePopulations(n_pyr=0, n_PV=0, n_SOM=0):
	Pyr_pop, PV_pop, SOM_pop = [None]*3

	if n_pyr > 0:
		print('\n==================== Creating pyramidal cell (n = {}) ===================='.format(n_pyr))
		Pyr_pop = Population('Pyr', pyr_morph_path, pyr_template_path, pyr_template_name, verbose=False, recording_dt=recording_dt)
		Pyr_pop.addCell()
		Pyr_pop.name_to_gid['Pyr0'] = chosen_GIDs['pyr']

	if n_PV > 0:
		print('\n==================== Creating PV population (n = {}) ===================='.format(n_PV))
		PV_pop = Population('PV', PV_morph_path, PV_template_path, PV_template_name, verbose=False, recording_dt=recording_dt)
		for i in tqdm(range(n_PV)):
			PV_cell_name = 'PV%i'%i
			PV_pop.addCell()
			PV_pop.name_to_gid[PV_cell_name] = chosen_GIDs['PV'][i]
			PV_pop.moveCell(PV_pop.cells[PV_cell_name]['cell'], (i*350)-(100*(n_PV+1)), -500, 0) # Morphology Visualization

	if n_SOM > 0:
		print('\n==================== Creating SOM population (n = {}) ===================='.format(n_SOM))
		SOM_pop = Population('SOM', SOM_morph_path, SOM_template_path, SOM_template_name, verbose=False, recording_dt=recording_dt)
		SOM_pop.addCell()	
		SOM_pop.name_to_gid['SOM0'] = chosen_GIDs['SOM']
		SOM_pop.moveCell(SOM_pop.cells['SOM0']['cell'], 0, -1000, 0) # Morphology Visualization

	return Pyr_pop, PV_pop, SOM_pop

def set_ThalamicInputs(Pyr_pop, PV_pop, SOM_pop, thalamic_GIDs):

	if Pyr_pop:

		where_Pyr_syns = ['basal_dendrites']		
		where_Pyr_syns_str = '{}'.format(str(where_Pyr_syns).split('[')[1].split(']')[0].replace(',', ' and').replace('\'',''))
		where_Pyr_syns = 'synapse_locs_instantiations/thalamic_syn_locs_Pyr_basal.p'

		print('\n==================== Connecting thalamic inputs to Pyramidal cell (on {}, standard: {}Hz, input weight: {}uS) ===================='.format(where_Pyr_syns_str, stand_freq, Pyr_input_weight)); sys.stdout.flush()
		Pyr_pop.addInput('Pyr0', std_params=Pyr_input_params, record_syns=record_thalamic_syns, where_synapses=where_Pyr_syns, weight=Pyr_input_weight, thalamic_activations_filename=activated_filename, connecting_gids=thalamic_GIDs['to_pyr'])

	if PV_pop:
		where_PV_syns = ['basal_dendrites', 'apical_dendrites']
		where_PV_syns_str = '{}'.format(str(where_PV_syns).split('[')[1].split(']')[0].replace(',', ' and').replace('\'',''))
		where_PV_syns = 'synapse_locs_instantiations/thalamic_syn_locs_PV_basal_apical.p'

		print('\n==================== Connecting thalamic inputs to PV cells (on {}, standard: {}Hz, input weight: {}uS) ===================='.format(where_PV_syns_str, stand_freq, PV_input_weight)); sys.stdout.flush()

		if PV_to_Pyr_source == 'voltage':
			for i, PV_cell_name in enumerate(tqdm(PV_pop.cells)):
				PV_pop.addInput(PV_cell_name, std_params=Pyr_input_params, record_syns=record_thalamic_syns, where_synapses=where_PV_syns, weight=PV_input_weight, thalamic_activations_filename=activated_filename, connecting_gids=thalamic_GIDs['to_PV'][PV_pop.name_to_gid[PV_cell_name]])	

		elif PV_to_Pyr_source == 'spike_times':
			print('Loading PV spike times from file ({})'.format(filenames['PV_spike_times']))
			PV_spike_times = cPickle.load(open(filenames['PV_spike_times'], 'rb'))

			for PV_cell_name in PV_pop.cells:
				PV_pop.cells[PV_cell_name]['soma_v'] = PV_spike_times['cells'][PV_cell_name]['soma_v']

	if SOM_pop:
		SOM_input_source = 'thalamic_input'

		where_SOM_syns = ['basal_dendrites']
		where_SOM_syns_str = '{}'.format(str(where_SOM_syns).split('[')[1].split(']')[0].replace(',', ' and').replace('\'',''))
		where_SOM_syns = 'synapse_locs_instantiations/thalamic_syn_locs_SOM_basal_high_input.p'
		
		print('\n==================== Connecting thalamic inputs to SOM cell (on {}, standard: {}Hz, input weight: {}uS) ===================='.format(where_SOM_syns_str, stand_freq, SOM_input_weight)); sys.stdout.flush()

		if SOM_input_source == 'thalamic_input':
			SOM_pop.addInput('SOM0', std_params=Pyr_input_params, record_syns=record_thalamic_syns, where_synapses=where_SOM_syns, weight=SOM_input_weight, thalamic_activations_filename=activated_filename, connecting_gids=thalamic_GIDs['to_SOM']) 
		
		elif SOM_input_source == 'Pyr':
			SOM_pop.connectCells('SOM0', Pyr_pop, 'Pyr0', 'voltage', [SOM_pop.cells['SOM0']['soma']], n_Pyr_to_SOM_syns, 'random', weight=SOM_input_weight, delay=SOM_input_delay)

def set_CorticalInputs(pre_pop=None, post_pop=None, connect_pops=None, weight=None, which_secs=[], syn_specs=None, record_syns=None, delay=None):
	def get_post_secs(which_secs, pop, cell):

		assert type(which_secs)==list, 'invalid input type! (which_secs variablem in set_CorticalInputs())'
		pre_to_post_secs = []
		for SEC in which_secs:
			if SEC=='soma':
				pre_to_post_secs = pre_to_post_secs + [pop.cells[cell][SEC]]
			else:
				SEC = SEC + '_dendrites'
				pre_to_post_secs = pre_to_post_secs + pop.cells[cell][SEC]

		return pre_to_post_secs 

	#POST=Pyr, PRE=PV
	if pre_pop and post_pop and connect_pops:
		print('\n========== Connecting {} population to {} (connection weight: {}uS) ===================='.format(pre_pop.population_name, post_pop.population_name, weight))
		
		for post_cell in post_pop.cells:
			pre_to_post_secs = get_post_secs(which_secs, post_pop, post_cell)

			for pre_cell in pre_pop.cells:
				post_pop.connectCells(post_cell, pre_pop, pre_cell, pre_to_post_secs, syn_specs, record_syns=record_syns, weight=weight, delay=delay, input_source='voltage', threshold=spike_threshold) # Adds self.connections to Population				
	return

def putspikes():
	
	# Pyramidal inputs
	if Pyr_pop:
		for axon in Pyr_pop.inputs['Pyr0']:
			stim_times = Pyr_pop.inputs['Pyr0'][axon]['stim_times']
			for netcon in Pyr_pop.inputs['Pyr0'][axon]['netcons']:
				for T in stim_times:
					netcon.event(T + Pyr_input_delay)

	# PV inputs
	if PV_pop:
		for PV_cell in PV_pop.inputs:
			for axon in PV_pop.inputs[PV_cell]:
				stim_times = PV_pop.inputs[PV_cell][axon]['stim_times']
				for netcon in PV_pop.inputs[PV_cell][axon]['netcons']:
					for T in stim_times:
						netcon.event(T + PV_input_delay)

	# SOM inputs
	if SOM_pop:
		for SOM_cell in SOM_pop.inputs:
			for axon in SOM_pop.inputs[SOM_cell]:
				stim_times = SOM_pop.inputs[SOM_cell][axon]['stim_times']
				for netcon in SOM_pop.inputs[SOM_cell][axon]['netcons']:
					for T in stim_times:
						netcon.event(T + SOM_input_delay)

def RunSim(v_init=-75, tstop=154*1000, record_specific=None):
	# Oren's simulation length is 154 seconds, I leave some time for last inputs to decay
	dend_v = []
	if record_specific:
		dend_v = h.Vector()
		dend_v.record(record_specific._ref_v)

	t = h.Vector()
	t.record(h._ref_t, recording_dt)
	h.tstop = tstop

	h.v_init = v_init

	h.run()

	return t, dend_v

def gatherInputs(post_pop):

	if len(post_pop.cell_inputs)>0:		
		inputs_to_post = {post_cell: {} for post_cell in post_pop.cells}
		
		for post_cell in post_pop.cells:
			temp_inputs = {'PV': {'g_GABA': [], 'i_GABA': []}, 'SOM': {'g_GABA': [], 'i_GABA': []}}
			for input_cell in post_pop.cell_inputs[post_cell]:
				cell_key = "".join([i for i in input_cell if not i.isdigit()])
				# temp_inputs[cell_key]['g_GABA'].append(post_pop.cell_inputs[post_cell][input_cell]['g_GABA'])
				# temp_inputs[cell_key]['i_GABA'].append(post_pop.cell_inputs[post_cell][input_cell]['i_GABA'])
				temp_inputs[cell_key]['g_GABA'].append(np.sum(post_pop.cell_inputs[post_cell][input_cell]['g_GABA'], axis=0))
				temp_inputs[cell_key]['i_GABA'].append(np.sum(post_pop.cell_inputs[post_cell][input_cell]['i_GABA'], axis=0))

			inputs_to_post[post_cell].update(temp_inputs)

			flatten = lambda x:[j for i in x for j in i]
			for cell_key in inputs_to_post[post_cell]:
				# inputs_to_post[post_cell][cell_key]['g_GABA'] = np.sum(flatten(inputs_to_post[post_cell][cell_key]['g_GABA']), axis=0)
				# inputs_to_post[post_cell][cell_key]['i_GABA'] = np.sum(flatten(inputs_to_post[post_cell][cell_key]['i_GABA']), axis=0)
				inputs_to_post[post_cell][cell_key]['g_GABA'] = np.sum(inputs_to_post[post_cell][cell_key]['g_GABA'], axis=0)
				inputs_to_post[post_cell][cell_key]['i_GABA'] = np.sum(inputs_to_post[post_cell][cell_key]['i_GABA'], axis=0)

	else:
		inputs_to_post = None

	return inputs_to_post

# ===============================================  Choose GIDs  ===============================================
print('\n========== Choosing GIDs from BlueBrain Database (pyramidal between {}Hz and {}Hz) =========='.format(freq1, freq2))

upload_from = 'GIDs_instantiations/pyr_72851_between_6666_9600' # upload_from = False
chosen_GIDs, chosen_PV_n_contacts, thalamic_GIDs = get_GIDs(upload_from)

activated_filename = get_thalamic_input_filename(thalamic_params)

# ==============================================  Stimulus Analysis  ==============================================
stimulus = Stimulus(thalamic_params, filenames['stim_times'], filenames['thalamic_activations'], tstop=tstop)

# ===============================================  Create Cell Populations  ===============================================
Pyr_pop, PV_pop, SOM_pop = CreatePopulations(n_pyr=1, n_PV=0, n_SOM=0)
# Pyr_pop, PV_pop, SOM_pop = CreatePopulations(n_pyr=1, n_PV=len(chosen_GIDs['PV']), n_SOM=1)

IClamp = h.IClamp(Pyr_pop.cells['Pyr0']['soma'](0.5))
IClamp.amp = Pyr_IClamp_amp
IClamp.dur = tstop
# ==============================================  Connect Populations with Synapses and Inputs  ==============================================
set_ThalamicInputs(Pyr_pop, PV_pop, SOM_pop, thalamic_GIDs)

if upload_cortical_syn_locs:
	PV_to_Pyr_syn_specs = 'synapse_locs_instantiations/PV_to_Pyr_syn_locs_soma.p'
	SOM_to_PV_syn_specs = 'synapse_locs_instantiations/SOM_to_PV_syn_locs_soma_basal.p'
	SOM_to_Pyr_syn_specs = 'synapse_locs_instantiations/SOM_to_Pyr_syn_locs_apical.p'
else:
	PV_to_Pyr_syn_specs = {PV_cell: ['random', chosen_PV_n_contacts[PV_pop.name_to_gid[PV_cell]]] for PV_cell in PV_pop.cells}
	SOM_to_PV_syn_specs = ['random', n_syns_SOM_to_PV]
	SOM_to_Pyr_syn_specs = ['random', n_syns_SOM_to_Pyr]

if connect_PV_to_Pyr: print('\n***WARNING: Assuming isopotential soma and perisomatic PV connections: all PV synapses are placed on soma(0.5)')

# PV ==> Pyr (either from PV spike times or PV soma voltage)
set_CorticalInputs(pre_pop=PV_pop, post_pop=Pyr_pop, connect_pops=connect_PV_to_Pyr, weight=PV_to_Pyr_weight, syn_specs=PV_to_Pyr_syn_specs, which_secs=[i for i in ['soma', 'basal', 'apical'] if i in PV_to_Pyr_syn_specs], record_syns=record_PV_syns, delay=PV_output_delay)

# SOM ==> PV
set_CorticalInputs(pre_pop=SOM_pop, post_pop=PV_pop, connect_pops=connect_SOM_to_PV, weight=SOM_to_PV_weight, syn_specs=SOM_to_PV_syn_specs, which_secs=[i for i in ['soma', 'basal', 'apical'] if i in SOM_to_PV_syn_specs], record_syns=record_SOM_syns, delay=SOM_output_delay)

# SOM ==> Pyr
set_CorticalInputs(pre_pop=SOM_pop, post_pop=Pyr_pop, connect_pops=connect_SOM_to_Pyr, weight=SOM_to_Pyr_weight, syn_specs=SOM_to_Pyr_syn_specs, which_secs=[i for i in ['soma', 'basal', 'apical'] if i in SOM_to_Pyr_syn_specs], record_syns=record_SOM_syns, delay=SOM_output_delay)

# ============================================== Run Simulation ==============================================
if record_channel:
	axon_gsk = h.Vector(); axon_gsk.record(Pyr_pop.cells['Pyr0']['axons'][0](0)._ref_gSK_E2_SK_E2)
	somatic_gsk = h.Vector(); somatic_gsk.record(Pyr_pop.cells['Pyr0']['soma'](0.5)._ref_gSK_E2_SK_E2)

print('\n========== Running Simulation (time: {}:{:02d}) =========='.format(time_module.localtime().tm_hour, time_module.localtime().tm_min))


events = h.FInitializeHandler(putspikes)

record_i_membrane = False
if record_i_membrane:
	mem_I = {}
	for sec in PV_pop.cells['PV0']['cell'].all:
		sec.insert('extracellular')
		mem_I[sec.name()] = {}
		for seg in sec:
			mem_I[sec.name()][seg] = h.Vector()
			mem_I[sec.name()][seg].record(seg._ref_i_membrane) # Units of i_membrane: mA/cm2

record_cap_i = False
if record_cap_i:
	cap_I = {}
	for sec in PV_pop.cells['PV0']['cell'].all:
		cap_I[sec.name()] = {}
		for seg in sec:
			cap_I[sec.name()][seg] = h.Vector().record(seg._ref_i_cap) # (mA/cm2)

disable_NMDA_PV = False
if disable_NMDA_PV:
	print('Disabling NMDA channels for PV0 (NMDA_ratio = 0)')
	for axon in PV_pop.inputs['PV0']:
		for syn in PV_pop.inputs['PV0'][axon]['synapses']:
			syn.NMDA_ratio = 0

record_soma_children = False
if record_soma_children:
	Pyr_soma_children = [i for i in Pyr_pop.cells['Pyr0']['soma'].children() if 'axon' not in i.name()]
	Pyr_children_v = [h.Vector().record(i(1)._ref_v) for i in Pyr_soma_children]
	Pyr_children_ri = [i(1).ri() for i in Pyr_soma_children]
	PV_soma_children = [i for i in PV_pop.cells['PV0']['soma'].children() if 'axon' not in i.name()]
	PV_children_v = [h.Vector().record(i(1)._ref_v) for i in PV_soma_children]
	PV_children_ri = [i(1).ri() for i in PV_soma_children]

start_time = time_module.time()
t, dend_v = RunSim(tstop = tstop)
end_time = time_module.time()

# os.system("say done")
os.system("osascript -e \'display notification \"Simulation took %i seconds\" with title \"Simulation Finished\" sound name \"Submarine\"\'"%(end_time-start_time))
print("Simulation took %i seconds"%(end_time-start_time))

if Pyr_pop.cell_inputs:
	Pyr_WZ_ax = Wehr_Zador(Pyr_pop, 'Pyr0', stimulus, 'Pyramidal', exc_weight=Pyr_input_weight, inh_weight=PV_to_Pyr_weight, input_pop_outputs=Pyr_pop.cell_inputs['Pyr0'], standard_freq=stand_freq, tstop=h.tstop, spike_threshold=spike_threshold, dt=recording_dt, t=t, take_after=250)
else:
	Pyr_WZ_ax = Wehr_Zador(Pyr_pop, 'Pyr0', stimulus, 'Pyramidal', exc_weight=Pyr_input_weight, inh_weight=PV_to_Pyr_weight, input_pop_outputs=None, standard_freq=stand_freq, tstop=h.tstop, spike_threshold=spike_threshold, dt=recording_dt, t=t, take_after=250)

window = 10
axes_h = None
C = iter(['skyblue', 'orange', 'crimson'])
for pop in [PV_pop, SOM_pop, Pyr_pop]:
	if pop:
		temp_cell = list(pop.cells.keys())[0]
		soma_v = pop.cells[temp_cell]['soma_v']
		axes_h = plotFRs(0, stim_times, soma_v, t, tstop=h.tstop, window=window, which_cell=temp_cell[:-1], axes_h=axes_h, color=next(C), take_after=250, take_before=50)


if record_soma_children:
	Pyr_I = []
	for i in range(len(Pyr_soma_children)):
		temp_child_v = Pyr_children_v[i]
		temp_volt_diff = [Pyr_pop.cells['Pyr0']['soma_v'][j]-temp_child_v[j] for j in range(len(t))]
		temp_ri = Pyr_children_ri[i]

		Pyr_I.append([j/temp_ri for j in temp_volt_diff])
	PV_I = []
	for i in range(len(PV_soma_children)):
		temp_child_v = PV_children_v[i]
		temp_volt_diff = [PV_pop.cells['PV0']['soma_v'][j]-temp_child_v[j] for j in range(len(t))]
		temp_ri = PV_children_ri[i]

		PV_I.append([j/temp_ri for j in temp_volt_diff])

if record_i_membrane:
	def from_specific_to_total_current(original_I, from_l_unit='micro', to_l_unit='centi', from_i_unit='milli', to_i_unit='milli'):
		'''
			Factors: to convert from mA to nA, change factor_multiply_current to 1e6

			factor_multiply_area=1e-4, factor_multiply_current=1
		'''

		dict_units = {'nano': 1e9, 'milli': 1e3, 'centi': 1e2, 'micro': 1e6}

		factor_multiply_area 	= dict_units[to_l_unit] / dict_units[from_l_unit]
		factor_multiply_current = dict_units[to_i_unit] / dict_units[from_i_unit]

		I = []

		for sec in original_I:
			sec_obj = [i for i in h.allsec() if i.name()==sec][0]

			for seg in original_I[sec]:
				seg_L = sec_obj.L / sec_obj.nseg
				seg_rad = sec_obj.diam / 2

				seg_area = (2*np.pi) * seg_L * seg_rad # In um
				seg_area = seg_area * factor_multiply_area # Turn from um to specified unit area (mostly it will be cm)

				I.append([i*seg_area for i in original_I[sec][seg]])

		I = np.sum(I, axis=0) # Sum to total current in all segments
		I = I * factor_multiply_current # Convert to specified current units

		return I

	def CumSum_I(I, I_title, window=[0, 100], to_i_unit=''):

		int_I = []

		stim_times = [i[0] for i in cPickle.load(open(filenames['stim_times'], 'rb'))[6666] if i[0]<=h.tstop]

		for T in stim_times:
			idx1 = int((T+window[0])/h.dt)
			idx2 = int((T+window[1])/h.dt)

			# Integrate
			I_window = list(I)[idx1:idx2]
			temp_integral = np.cumsum([i*h.dt for i in I_window])

			int_I.append(temp_integral)

		plt.figure()
		plt.title('Histogram of {} Integral Cumulative Sum in window of {}-{}ms After Stimulus'.format(I_title, window[0], window[1])) 
		plt.xlabel('Integral') 
		plt.ylabel('Count')
		plt.hist([j for i in int_I for j in i])

		plt.figure()
		plt.title('Integral CumSum for total time ({})'.format(I_title))
		plt.xlabel('T (ms)')
		plt.ylabel('Integral Cumulative Sum')
		plt.plot(t, np.cumsum([i*h.dt for i in I]))

		plt.figure()
		plt.title('Total Membrane Current')
		plt.xlabel('T (ms)')
		plt.ylabel('I ({})'.format(to_i_unit[0]+'A'))
		plt.plot(t, I)
		for T in stim_times:
			plt.axvline(T, color='gray', LineStyle='--')

	to_i_unit = 'milli'

	total_I_mem = from_specific_to_total_current(mem_I, to_i_unit=to_i_unit)
	CumSum_I(total_I_mem, 'Membrane Current', to_i_unit=to_i_unit)

if record_cap_i:
	to_i_unit = 'milli'

	total_I_cap = from_specific_to_total_current(cap_I, to_i_unit=to_i_unit)
	CumSum_I(total_I_cap, 'Capacitative Current', to_i_unit=to_i_unit)

dump_somas = False
if dump_somas:
	recorded_str = ''
	if record_thalamic_syns:
		recorded_str = recorded_str + 'Thalamic'
	if record_Pyr_syns:
		recorded_str = recorded_str + (len(recorded_str)>0)*', ' + 'from Pyr'
	if record_PV_syns:
		recorded_str = recorded_str + (len(recorded_str)>0)*', ' + 'from PV'
	if record_SOM_syns:
		recorded_str = recorded_str + (len(recorded_str)>0)*', ' + 'from SOM'

	print('Saving data to file, recorded synapses: {}'.format(recorded_str))

	inh_to_Pyr = gatherInputs(Pyr_pop)
	Pyr_pop.dumpSomaVs('local_simulations/{}'.format(job_id), activated_filename, dump_type='as_dict', inh_inputs=inh_to_Pyr, job_id=job_id)

	inh_to_PV = gatherInputs(PV_pop)
	PV_pop.dumpSomaVs('local_simulations/{}'.format(job_id), activated_filename, dump_type='as_dict', inh_inputs=inh_to_PV, job_id=job_id)

	inh_to_SOM = gatherInputs(SOM_pop)
	SOM_pop.dumpSomaVs('local_simulations/{}'.format(job_id), activated_filename, dump_type='as_dict', inh_inputs=inh_to_SOM, job_id=job_id)

if record_channel:
	f, ax1 = plt.subplots()

	ax1.plot(t, axon_gsk, 'xkcd:aquamarine', label='Axonal $g_{SK}$')
	ax1.plot(t, somatic_gsk, 'xkcd:azure', label='Somatic $g_{SK}$')
	ax1.set_xlabel('T (ms)')
	ax1.set_ylabel('G (S/$cm^2$)')
	ax2.legend()

	ax2.plot(t, Pyr_pop.cells['Pyr0']['soma_v'], label='Pyr Somatic V')
	ax2.set_ylabel('V (mV)')

	stim_times = [i[0] for i in stimulus.stim_times_all]
	for T in stim_times: 
		if T <= h.tstop:
			plt.axvline(T, LineStyle='--',color='gray',LineWidth=1) 

# Cross-correlation of Pyramidal spikes around thalamic spikes
do_cross_correlation_with_thal = False
if Pyr_pop and do_cross_correlation_with_thal:
	Vs = []
	window = 300
	V = Pyr_pop.cells['Pyr0']['soma_v']
	spike_times = [t[i] for i in range(len(t)) if (V[i]>spike_threshold) and (V[i]>V[i-1]) and (V[i]>V[i+1])]
	stim_times = cPickle.load(open(filenames['stim_times'], 'rb'))[6666]
	stim_times = [i[0] for i in stim_times if i[0]<=tstop]
	stim_window_after = 50
	stim_intervals = [[i, i+stim_window_after] for i in stim_times]

	dt = h.dt 

	spont_window_spikes, stim_window_spikes = [], []
	spont_thalamic, stim_thalamic = [], []
	spont_v, stim_v = [], []
	for axon in Pyr_pop.inputs['Pyr0']:
		for T in Pyr_pop.inputs['Pyr0'][axon]['stim_times']:
			if T<=h.tstop:
				idx1 = int((T-window)/h.dt)
				idx2 = int((T+window)/h.dt)
				if idx1>=0 and idx2*dt<=tstop: 
					T1 = t[idx1] 
					T2 = t[idx2] 

					spikes_in_window = [i-T for i in spike_times if (i>=T1) and (i<=T2)]
					thalamic_spike_is_evoked = any([(T>=i[0]) and (T<=i[1]) for i in stim_intervals])
					
					# thalamic_spike_is_evoked = T>=2000

					if thalamic_spike_is_evoked:
						stim_thalamic.append(T)
						stim_window_spikes.append(spikes_in_window)
						stim_v.append(list(V)[idx1:idx2])
					else:
						spont_thalamic.append(T)
						spont_window_spikes.append(spikes_in_window)
						spont_v.append(list(V)[idx1:idx2])

	flatten = lambda vec: [j for i in vec for j in i]
	plt.figure()
	stim_H, B1 = np.histogram(flatten(stim_window_spikes), bins=50)
	# stim_H = [i/sum(stim_H) for i in stim_H]
	spont_H, B2 = np.histogram(flatten(spont_window_spikes), bins=50)
	# spont_H = [i/sum(spont_H) for i in spont_H]
	plt.bar(B1[:-1], stim_H, alpha=0.5, label='Stimulus', width=np.diff(B1)[0])
	plt.bar(B2[:-1], spont_H, alpha=0.5, label='Spontaneous', width=np.diff(B2)[0])
	plt.axvline(0, LineStyle='--', color='gray', label='Thalamic Spike')
	plt.legend()
	plt.suptitle('Histogram of Pyramidal Spike Times, Around Spontaneous & Evoked Thalamic Spikes') 
	plt.xlabel('Per-Stimulus Time (ms)')
	plt.ylabel('n_spikes')
	plt.title('Simulation length: %sms'%format(int(tstop), ',d')) 





'''

somas_ax = PlotSomas({'Pyr0': Pyr_pop, 'PV0': PV_pop, 'SOM0': SOM_pop}, t, stimuli[stand_freq],  tstop=h.tstop, spike_threshold=spike_threshold, dt=h.dt)
Pyr_WZ_ax = Wehr_Zador(Pyr_pop, 'Pyr0', stimuli, 'Pyramidal', exc_weight=Pyr_input_weight, inh_weight=PV_to_Pyr_weight, input_pop_outputs=Pyr_pop.cell_inputs['Pyr0'], standard_freq=stand_freq, tstop=h.tstop, spike_threshold=spike_threshold, dt=h.dt, t=t)
SOM_WZ_ax = Wehr_Zador(SOM_pop, 'SOM0', stimuli, 'SOM', exc_weight=SOM_input_weight, input_pop_outputs=None, standard_freq=stand_freq, tstop=h.tstop, spike_threshold=spike_threshold, dt=h.dt, t=t)
if PV_to_Pyr_source == 'voltage':
	PV_WZ_ax = Wehr_Zador(PV_pop, 'PV0', stimuli, 'PV', exc_weight=PV_input_weight, inh_weight=SOM_to_PV_weight, input_pop_outputs=PV_pop.cell_inputs['PV0'], standard_freq=stand_freq, tstop=h.tstop, spike_threshold=spike_threshold, dt=h.dt, t=t)
'''
# ============================================== Analyze Input vs. Response ==============================================
'''
print('Analyzing input vs. Pyramidal somatic responses')
def InputvsResponse(Pyr_pop, stimulus, axon_gids, take_before=0, take_after=50, activated_filename=activated_filename):

	axon_gids = [i[0] for i in axon_gids]
	Pyr_response = Pyr_pop.cells['Pyr0']['soma_v']
	thalamic_activations = cPickle.load(open('{}'.format(activated_filename), 'rb'))
	
	times = [i[0] for i in stimuli[standard_freq].stim_times_all]
	times = [i for i in times if i<h.tstop]

	cut_vec = lambda vec, start_idx, end_idx: [vec[i] for i in range(start_idx, end_idx)]
	
	input_groups = {'spike_success': [], 'spike_fail': []}
	spike_count = 0

	for T in tqdm(times):
		temp_timings = {axon: {'times': [], 'stim_FR_sec': None} for axon in axon_gids}
		for axon in axon_gids:
			temp_axon_timings = [i for i in thalamic_activations[axon] if (i >= T-take_before) and (i <= T+take_after)]
			temp_timings[axon]['times'] = temp_axon_timings
			temp_timings[axon]['stim_FR_sec'] = 1000*len(temp_axon_timings) / (take_after - take_before)
		idx1 = (np.abs([i-(T-take_before) for i in t])).argmin()
		idx2 = (np.abs([i-(T+take_after) for i in t])).argmin()
		
		v_vec = cut_vec(Pyr_pop.cells['Pyr0']['soma_v'], idx1, idx2)
		if any([i>=spike_threshold for i in v_vec]):
			spike_count += 1
			input_groups['spike_success'].append(temp_timings)
		else:
			input_groups['spike_fail'].append(temp_timings)

	def plot_stuff():
		fig, axes = plt.subplots(2, 2)
		axes = [j for i in axes for j in i]
		fig.subplots_adjust(hspace=0.34, bottom=0.08, top=0.9) 
		fig.suptitle('Comparison Between Inputs Succeeding & Failing to Elicit Spike in Pyramidal Cell', size=15)

		success_all_FRs = [m for n in [[i[a]['stim_FR_sec'] for a in i] for i in input_groups['spike_success']] for m in n]
		fail_all_FRs = [m for n in [[i[a]['stim_FR_sec'] for a in i] for i in input_groups['spike_fail']] for m in n]
		
		success_FR_mean = np.mean(success_all_FRs)
		fail_FR_mean = np.mean(fail_all_FRs)
		
		success_FR_std = np.std(success_all_FRs)
		fail_FR_std = np.std(fail_all_FRs)
		success_FR_se = success_FR_std / np.sqrt(len(success_all_FRs))
		fail_FR_se = fail_FR_std / np.sqrt(len(fail_all_FRs))

		_, pval0 = ttest_ind(success_all_FRs, fail_all_FRs)
		axes[0].set_title('Mean Firing Rates (p = %.3f)'%pval0)
		axes[0].bar(['Success', 'Fail'], [success_FR_mean, fail_FR_mean], yerr=[success_FR_se, fail_FR_se], alpha=0.5)
		axes[0].set_ylabel(r'FR ($\frac{spikes}{sec}$)')
		axes[0].set_xlim([-1, 2])

		success_all_high_FRs = [m for m in success_all_FRs if m>cutoff_freq]
		fail_all_high_FRs = [m for m in fail_all_FRs if m>cutoff_freq]
		success_high_FR_mean = np.mean(success_all_high_FRs)
		fail_high_FR_mean = np.mean(fail_all_high_FRs)

		success_high_FR_std = np.std(success_all_high_FRs)
		success_high_FR_se = success_high_FR_std / np.sqrt(len(success_all_high_FRs))
		fail_high_FR_std = np.std(fail_all_high_FRs)
		fail_high_FR_se = fail_high_FR_std / np.sqrt(len(fail_all_high_FRs))

		_, pval1 = ttest_ind(success_all_high_FRs, fail_all_high_FRs)
		axes[1].set_title('Mean High Firing Rates - above %.1f (p = %.3f)'%(cutoff_freq, pval1))
		axes[1].bar(['Success', 'Fail'], [success_high_FR_mean, fail_high_FR_mean], yerr=[success_high_FR_se, fail_high_FR_se], alpha=0.5, label='Mean')
		axes[1].bar(['Success', 'Fail'], [np.median(success_all_high_FRs), np.median(fail_all_high_FRs)], yerr=[success_high_FR_se, fail_high_FR_se], alpha=0.5, label='median')
		axes[1].set_ylabel(r'FR ($\frac{spikes}{sec}$)')
		axes[1].set_xlim([-1, 2])
		axes[1].legend()
		
		success_ISI = [[list(np.diff(i[axon]['times'])) for axon in i if len(i[axon]['times'])>1] for i in input_groups['spike_success']]
		fail_ISI = [[list(np.diff(i[axon]['times'])) for axon in i if len(i[axon]['times'])>1] for i in input_groups['spike_fail']]

		success_all_ISI = [m for k in [ j for i in success_ISI for j in i] for m in k]
		fail_all_ISI = [m for k in [ j for i in fail_ISI for j in i] for m in k]

		success_ISI_mean = np.mean(success_all_ISI)  
		fail_ISI_mean = np.mean(fail_all_ISI) 

		success_ISI_std = np.std(success_all_ISI)  
		fail_ISI_std = np.std(fail_all_ISI)  

		success_ISI_se = success_ISI_std / np.sqrt(len(success_all_ISI))
		fail_ISI_se = fail_ISI_std / np.sqrt(len(fail_all_ISI))

		_, pval2 = ttest_ind(success_all_ISI, fail_all_ISI)
		axes[2].set_title('Mean Inter-Spike-Interval (p = %.3f)'%pval2)
		axes[2].bar(['Success', 'Fail'], [success_ISI_mean, fail_ISI_mean], yerr=[success_ISI_se, fail_ISI_se], alpha=0.5)
		axes[2].set_ylabel('Mean ISI (ms)')
		axes[2].set_xlim([-1, 2])

		success_non_responsive = [len([i for i in t.values() if i['stim_FR_sec']<=cutoff_freq]) for t in input_groups['spike_success']]
		fail_non_responsive = [len([i for i in t.values() if i['stim_FR_sec']<=cutoff_freq]) for t in input_groups['spike_fail']]
		success_se = np.std(success_non_responsive) / np.sqrt(len(success_non_responsive))
		fail_se = np.std(fail_non_responsive) / np.sqrt(len(fail_non_responsive))
		_, pval3 = ttest_ind(success_non_responsive, fail_non_responsive)
		axes[3].set_title('Number of Non-Responsive (p = %.3f)'%pval3)
		axes[3].bar(['Success', 'Fail'], [np.mean(success_non_responsive), np.mean(fail_non_responsive)], yerr=[success_se, fail_se], alpha=0.5)
		axes[3].set_ylabel('Number of Non-Responsive')
		axes[3].set_xlim([-1, 2])
	# plot_stuff()
	return input_groups

input_groups = InputvsResponse(Pyr_pop, stimuli[stand_freq], thalamic_GIDs['to_pyr'], take_before=0, take_after=50)
'''

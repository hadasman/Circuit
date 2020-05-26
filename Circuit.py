# import matplotlib
# matplotlib.use("TkAgg")
import pandas as pd
import numpy as np
import _pickle as cPickle
import matplotlib.pyplot as plt
import time as time_module
plt.ion()

import pdb, os, sys

from neuron import gui, h
from math import log
from tqdm import tqdm
from scipy.stats import ttest_ind

from Population import Population
from Stimulus import Stimulus
from Connectivity import Connectivity
from Parameter_Initialization import * # Initialize parameters before anything else!
from plotting_functions import plotThalamicResponses, Wehr_Zador, PlotSomas, plotFRs

assert os.getcwd().split('/')[-1] == 'Circuit', 'Wrong directory'

if len(sys.argv)>1:
	tstop = sys.argv[1]
# ============================================  Define Functions & Constants  ============================================
'''
Plan:
	- V - now: 5000ms, with PV connections, 9600Hz => 6 spikes out of 9
	- V - 5000ms without PV connections, 6666Hz (maybe also 9600Hz) => 7 out of 9 [same synapses+PV: 5] (6666)
	- V - 10000ms without+with PV connections (for same synapses), 6666Hz (maybe also 9600Hz) => 13 (with PV) & 20 (without PV) out of 25
'''
activated_filename = filenames['thalamic_activations_6666']
activated_standard_freq = freq1*(str(freq1) in activated_filename) + freq2*(str(freq2) in activated_filename)
alternative_freq = freq1*(str(freq1) not in activated_filename) + freq2*(str(freq2) not in activated_filename)

def get_GIDs(upload_from):

	if upload_from:
		chosen_GIDs = {}
		chosen_GIDs['pyr'] 			  = cPickle.load(open('{}/chosen_pyr.p'.format(upload_from), 'rb'))
		chosen_GIDs['PV'] 			  = cPickle.load(open('{}/chosen_PV.p'.format(upload_from), 'rb'))
		chosen_GIDs['SOM'] 			  = cPickle.load(open('{}/chosen_SOM.p'.format(upload_from), 'rb'))
		chosen_PV_n_contacts  		  = cPickle.load(open('{}/chosen_PV_n_contacts.p'.format(upload_from), 'rb'))

		thalamic_GIDs = {}
		thalamic_GIDs['to_pyr'] = cPickle.load(open('{}/connecting_gids_to_pyr.p'.format(upload_from), 'rb'))
		thalamic_GIDs['to_PV']  = cPickle.load(open('{}/connecting_gids_to_PV.p'.format(upload_from), 'rb'))
		thalamic_GIDs['to_SOM'] = cPickle.load(open('{}/connecting_gids_to_SOM.p'.format(upload_from), 'rb'))
	else:
		connectivity = Connectivity(pyr_type, [not_PVs, 'exclude'], [SOM_types, 'include'], 
									cell_type_to_gids=filenames['cell_type_gids'],
									thalamic_locs=filenames['thalamic_locs'], 
									cell_details=filenames['cell_details'],
									cortex_connectivity=filenames['cortex_connectivity'],
									thal_connections=filenames['thalamic_connections'])

		chosen_GIDs = {}
		chosen_GIDs['pyr'] = connectivity.choose_GID_between_freqs(connectivity.pyr_GIDs, freq1, freq2)
		chosen_GIDs['PV'], chosen_PV_n_contacts = connectivity.choose_GID_by_post(connectivity.PV_GIDs, chosen_GIDs['pyr'])
		chosen_GIDs['SOM'] = connectivity.choose_GID_between_freqs(connectivity.SOM_GIDs, freq1, freq2)
		
		# Find thalamic GIDs connecting the the pyramidal cell
		thalamic_GIDs = {}
		thalamic_GIDs['to_pyr'] = connectivity.find_PresynGIDs(chosen_GIDs['pyr'])
		
		thalamic_GIDs['to_PV'] = {}
		for PV_gid in chosen_GIDs['PV']:
			thalamic_GIDs['to_PV'][PV_gid] = connectivity.find_PresynGIDs(PV_gid)
		
		thalamic_GIDs['to_SOM'] = connectivity.find_PresynGIDs(chosen_GIDs['SOM'])

	return chosen_GIDs, chosen_PV_n_contacts, thalamic_GIDs

def CreatePopulations(n_pyr=0, n_PV=0, n_SOM=0):
	Pyr_pop, PV_pop, SOM_pop = [None]*3

	if n_pyr > 0:
		print('\n==================== Creating pyramidal cell ====================')
		Pyr_pop = Population('Pyr', pyr_morph_path, pyr_template_path, pyr_template_name, verbose=False)
		Pyr_pop.addCell()
		Pyr_pop.name_to_gid['Pyr0'] = chosen_GIDs['pyr']

	if n_PV > 0:
		print('\n==================== Creating PV population ====================')
		PV_pop = Population('PV', PV_morph_path, PV_template_path, PV_template_name, verbose=False)
		for i in tqdm(range(n_PV)):
			PV_cell_name = 'PV%i'%i
			PV_pop.addCell()
			PV_pop.name_to_gid[PV_cell_name] = chosen_GIDs['PV'][i]
			PV_pop.moveCell(PV_pop.cells[PV_cell_name]['cell'], (i*350)-(100*(n_PV+1)), -500, 0) # Morphology Visualization

	if n_SOM > 0:
		print('\n==================== Creating SOM population ====================')
		SOM_pop = Population('SOM', SOM_morph_path, SOM_template_path, SOM_template_name, verbose=False)
		SOM_pop.addCell()	
		SOM_pop.name_to_gid['SOM0'] = chosen_GIDs['SOM']
		SOM_pop.moveCell(SOM_pop.cells['SOM0']['cell'], 0, -1000, 0) # Morphology Visualization

	return Pyr_pop, PV_pop, SOM_pop

def CreateSpikeEvents(synapse_data, pop_name, cell_name, input_delay, given_stim_times=None): 
	'''
	Custom initialization: insert event to queue (if h.FInitializeHandler not called, 
	event is erased by h.run() because it clears the queue)
	'''
	events = [] 

	for axon in synapse_data[cell_name]: 
		if not given_stim_times:
			stim_times = synapse_data[cell_name][axon]['stim_times'] 
		
		for i in range(len(synapse_data[cell_name][axon]['netcons'])):
			for T in stim_times: 
				events.append(h.FInitializeHandler('nrnpython("{}.inputs[\'{}\'][\'{}\'][\'netcons\'][{}].event({})")'\
					.format(pop_name, cell_name, axon, i, T+input_delay))) 
	
	return events 

def set_ThalamicInputs(Pyr_pop, PV_pop, SOM_pop, thalamic_GIDs):
	pyr_events, PV_events, SOM_events = [], [], []

	if Pyr_pop:
		print('\n==================== Connecting thalamic inputs to Pyramidal cell (standard: {}Hz, input weight: {}uS) ===================='.format(activated_standard_freq, Pyr_input_weight))
		Pyr_pop.addInput(list(Pyr_pop.cells.keys())[0], weight=Pyr_input_weight, thalamic_activations_filename=activated_filename, connecting_gids=thalamic_GIDs['to_pyr'])
		pyr_events = CreateSpikeEvents(Pyr_pop.inputs, 'Pyr_pop', 'Pyr0', Pyr_input_delay)

	if PV_pop:
		print('\n==================== Connecting thalamic inputs to PV cells (standard: {}Hz, input weight: {}uS) ===================='.format(activated_standard_freq, PV_input_weight))
		if PV_to_Pyr_source == 'voltage':
			PV_events = []
			for i, PV_cell_name in enumerate(tqdm(PV_pop.cells)):
				PV_pop.addInput(PV_cell_name, weight=PV_input_weight, thalamic_activations_filename=activated_filename, connecting_gids=thalamic_GIDs['to_PV'][PV_pop.name_to_gid[PV_cell_name]])	
				PV_events.append(CreateSpikeEvents(PV_pop.inputs, 'PV_pop', PV_cell_name, PV_input_delay))
			PV_events = [j for i in PV_events for j in i]

		elif PV_to_Pyr_source == 'spike_times':
			print('Loading PV spike times from file ({})'.format(filenames['PV_spike_times']))
			PV_spike_times = cPickle.load(open(filenames['PV_spike_times'], 'rb'))

			for PV_cell_name in PV_pop.cells:
				PV_pop.cells[PV_cell_name]['soma_v'] = PV_spike_times['cells'][PV_cell_name]['soma_v']

	if SOM_pop:
		SOM_input_source = 'thalamic_input'
		print('\n==================== Connecting {} inputs to SOM cell (standard: {}Hz, input weight: {}uS) ===================='.format(SOM_input_source, activated_standard_freq, SOM_input_weight))
		if SOM_input_source == 'Pyr':
			Pyr_pop.connectCells('Pyr0', 'voltage', [SOM_pop.cells['SOM0']['soma']], n_Pyr_to_SOM_syns, 'random', weight=SOM_input_weight, delay=SOM_input_delay)

		elif SOM_input_source == 'thalamic_input':
			SOM_pop.addInput('SOM0', weight=SOM_input_weight, thalamic_activations_filename=activated_filename, connecting_gids=thalamic_GIDs['to_SOM']) 
			SOM_events = CreateSpikeEvents(SOM_pop.inputs, 'SOM_pop', 'SOM0', SOM_input_delay)

	return pyr_events, PV_events, SOM_events

def RunSim(v_init=-75, tstop=154*1000, verbose=True, record_specific=None):
	# Oren's simulation length is 154 seconds, I leave some time for last inputs to decay
	dend_v = []
	if record_specific:
		dend_v = h.Vector()
		dend_v.record(record_specific._ref_v)

	t = h.Vector()
	t.record(h._ref_t)
	h.tstop = tstop

	h.v_init = v_init

	h.finitialize()
	h.run()

	if verbose:
		os.system('say "Simulation finished"') 

	return t, dend_v

# ===============================================  Choose GIDs  ===============================================
print('\n========== Choosing GIDs from BlueBrain Database (pyramidal between {}Hz and {}Hz) =========='.format(freq1, freq2))

upload_from = 'GIDs_instantiations/pyr_72851_between_6666_9600' # upload_from = False
chosen_GIDs, chosen_PV_n_contacts, thalamic_GIDs = get_GIDs(upload_from)

# ==============================================  Stimulus Analysis  ==============================================
stimuli[activated_standard_freq] = Stimulus(activated_standard_freq, alternative_freq, filenames['stim_times'], filenames['thalamic_activations_6666'], axon_gids=[i[0] for i in thalamic_GIDs['to_pyr']])

run_plot_function = False
if run_plot_function:
	stimuli[alternative_freq] = Stimulus(alternative_freq, activated_standard_freq, filenames['stim_times'], filenames['thalamic_activations_9600'], axon_gids=[i[0] for i in thalamic_GIDs['to_pyr']])
	stim_ax = plotThalamicResponses(stimuli, activated_standard_freq, alternative_freq, thalamic_locations, run_function=True)


# ===============================================  Create Cell Populations  ===============================================
Pyr_pop, PV_pop, SOM_pop = CreatePopulations(n_pyr=1, n_PV=1, n_SOM=1)

# ==============================================  Connect Populations with Synapses and Inputs  ==============================================
pyr_events, PV_events, SOM_events = set_ThalamicInputs(Pyr_pop, PV_pop, SOM_pop, thalamic_GIDs)

connect_interneurons = True
if connect_interneurons:
	
	# PV ==> Pyr (either from PV spike times or PV soma voltage)
	if PV_pop and Pyr_pop:
		print('Connecting PV population to Pyramidal cell (connection weight: {}uS)'.format(PV_to_Pyr_weight))	
		PV_to_Pyr_events = []
		PV_to_Pyr_post_secs = [Pyr_pop.cells['Pyr0']['soma']]
		
		for PV_cell_name in tqdm(PV_pop.cells):
			temp_PV_gid = PV_pop.name_to_gid[PV_cell_name]
			temp_n_syns = chosen_PV_n_contacts[temp_PV_gid]
			PV_pop.connectCells(PV_cell_name, PV_to_Pyr_post_secs, temp_n_syns, 'random', input_source=PV_to_Pyr_source, weight=PV_to_Pyr_weight, delay=PV_output_delay, threshold=spike_threshold) # Adds self.connections to Population
			
			if PV_to_Pyr_source == 'spike_times':
				spike_time_func = eval(PV_spike_times['spike_times_func_as_string'])
				t = PV_spike_times['t']
				
				stim_times = spike_time_func(PV_spike_times['cells'][PV_cell_name]['soma_v'], spike_threshold)
				SOM_events = CreateSpikeEvents(PV_pop.outputs, 'PV_pop', PV_cell_name, PV_output_delay, given_stim_times=stim_times)

				
				# for post_branch in PV_pop.outputs[PV_cell_name]:
				# 	for i in range(len(PV_pop.outputs[PV_cell_name][post_branch]['netcons'])):
				# 		for T in stim_times:
				# 			PV_to_Pyr_events.append(h.FInitializeHandler('nrnpython("PV_pop.outputs[\'{}\'][\'{}\'][\'netcons\'][{}].event({})")'.format(PV_cell_name, post_branch, i, T+PV_output_delay)))


		print('\n***WARNING: Assuming isopotential soma and perisomatic PV connections: all PV synapses are placed on soma(0.5)')
	
	# SOM ==> PV
	if SOM_pop and PV_pop:
		print('Connecting SOM population to PV Population (connection weight: {}uS)'.format(SOM_to_PV_weight))
		for PV_cell in PV_pop.cells:
			SOM_to_PV_post_secs = PV_pop.cells[PV_cell]['basal_dendrites'] + PV_pop.cells[PV_cell]['apical_dendrites']
			SOM_pop.connectCells('SOM0', SOM_to_PV_post_secs, n_SOM_to_PV_syns, 'random', input_source='voltage', weight=SOM_to_PV_weight, delay=SOM_output_delay, threshold=spike_threshold)

	# SOM ==> Pyr
	if SOM_pop and Pyr_pop:
		print('Connecting SOM population to Pyramidal Population (connection weight: {}uS)'.format(SOM_to_Pyr_weight))
		for Pyr_cell in Pyr_pop.cells:
			SOM_to_Pyr_post_secs = Pyr_pop.cells[Pyr_cell]['apical_dendrites'] # + Pyr_pop.cells[Pyr_cell]['basal_dendrites']
			SOM_pop.connectCells('SOM0', SOM_to_Pyr_post_secs, n_SOM_to_Pyr_syns, 'random', input_source='voltage', weight=SOM_to_Pyr_weight, delay=SOM_output_delay, threshold=spike_threshold)

# ============================================== Plot example responses ==============================================

record_PV_dendrite = False
recorded_segment = []
if record_PV_dendrite:
	dend = [i for i in PV_pop.cells['PV3']['basal_dendrites'] if 'dend[42]' in i.name()][0]
	seg = 0.5

	recorded_segment = dend(seg)

print('\n========== Running Simulation (time: {}:{:02d}) =========='.format(time_module.localtime().tm_hour, time_module.localtime().tm_min))
start_time = time_module.time()
t, dend_v = RunSim(tstop = tstop, record_specific=recorded_segment)
end_time = time_module.time()
# os.system("say simulation took %i seconds"%(end_time-start_time))
print("Simulation took %i seconds"%(end_time-start_time))

stim_times = [i[0] for i in stimuli[6666].stim_times_all]

axes_h = None
C = iter(['skyblue', 'orange', 'crimson'])
for pop in [PV_pop, SOM_pop, Pyr_pop]:
	if pop:
		temp_cell = list(pop.cells.keys())[0]
		soma_v = pop.cells[temp_cell]['soma_v']
		axes_h = plotFRs(stim_times, soma_v, t, tstop=h.tstop, window=6, which_cell=temp_cell[:-1], axes_h=axes_h, color=next(C))

# soma_v = PV_pop.cells['PV0']['soma_v']
# plotFRs(stim_times, soma_v, t, tstop=h.tstop, window=5, which_cell='PV')


# PV_pop.dumpSomaVs(t, activated_filename)

if PV_pop and Pyr_pop:
	if PV_to_Pyr_source == 'voltage':
		PV_WZ_ax = Wehr_Zador(PV_pop, 'PV0', stimuli, 'PV', exc_weight=PV_input_weight, input_pop_outputs=None, standard_freq=activated_standard_freq, tstop=h.tstop, spike_threshold=spike_threshold, dt=h.dt, t=t)
	Pyr_WZ_ax = Wehr_Zador(Pyr_pop, 'Pyr0', stimuli, 'Pyramidal', exc_weight=Pyr_input_weight, inh_weight=PV_to_Pyr_weight, input_pop_outputs=PV_pop.outputs, standard_freq=activated_standard_freq, tstop=h.tstop, spike_threshold=spike_threshold, dt=h.dt, t=t)
	
	if SOM_pop:
		somas_ax = PlotSomas({'Pyr0': Pyr_pop, 'PV0': PV_pop, 'SOM0': SOM_pop}, t, stimuli[activated_standard_freq],  tstop=h.tstop, spike_threshold=spike_threshold, dt=h.dt)
		SOM_WZ_ax = Wehr_Zador(SOM_pop, 'SOM0', stimuli, 'SOM', exc_weight=SOM_input_weight, input_pop_outputs=None, standard_freq=activated_standard_freq, tstop=h.tstop, spike_threshold=spike_threshold, dt=h.dt, t=t)

'''
if record_PV_dendrite:
	plt.figure()
	plt.plot(t, dend_v, label='dend')
	plt.plot(t, PV_pop.cells['PV3']['soma_v'], label='soma')
	plt.legend()
	plt.title('Voltage in {}({})'.format(dend, seg))
	plt.xlabel('T (ms)')
	plt.ylabel('V (mV)')

# ============================================== Analyze Input vs. Response ==============================================

print('Analyzing input vs. Pyramidal somatic responses')
def InputvsResponse(Pyr_pop, standard_freq, stimuli, axon_gids, take_before=0, take_after=100, activated_filename=activated_filename):

	Pyr_response = Pyr_pop.cells['Pyr0']['soma_v']
	# thalamic_activations = cPickle.load(open('thalamocortical_Oren/SSA_spike_times/input%s_by_gid.p'%standard_freq, 'rb'))
	thalamic_activations = cPickle.load(open('{}'.format(activated_filename), 'rb'))
	
	# times = stimuli[standard_freq].stim_times_standard
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
	return input_groups

try:
	cutoff_freq = cutoff_freq
except:
	cutoff_freq = 1.8
pyr_input_gids = [int(i.split('gid_')[1]) for i in Pyr_pop.inputs['Pyr0'].keys()]  
input_groups = InputvsResponse(Pyr_pop, 6666, stimuli, pyr_input_gids, take_before=0, take_after=100)
'''




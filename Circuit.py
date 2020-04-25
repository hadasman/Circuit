import matplotlib
matplotlib.use("TkAgg")
import pandas as pd
import numpy as np
from neuron import gui, h
import pdb, os, sys
import matplotlib.pyplot as plt
plt.ion()
from Population import Population
from Stimulus import Stimulus
from math import log
import _pickle as cPickle
os.chdir('../Circuit')
from tqdm import tqdm

# ============================================  Define Functions & Constants  ============================================
'''
Plan:
	- V - now: 5000ms, with PV connections, 9600Hz => 6 spikes out of 9
	- V - 5000ms without PV connections, 6666Hz (maybe also 9600Hz) => 7 out of 9 [same synapses+PV: 5] (6666)
	- V - 10000ms without+with PV connections (for same synapses), 6666Hz (maybe also 9600Hz) => 13 (with PV) & 20 (without PV) out of 25
'''
def initializeParameters():
	global pyr_template_path, pyr_template_name, pyr_morph_path
	global PV_template_path, PV_template_name, PV_morph_path
	global PV_input_weight, PV_to_Pyr_weight, Pyr_input_weight
	global pyr_type, not_PVs, PV_input_delay, Pyr_input_delay, PV_output_delay, freq1, freq2, simulation_time, spike_threshold
	global filenames, cell_type_gids, thal_connections, thalamic_locations

	pyr_template_path 	= 'EPFL_models/L4_PC_cADpyr230_1' # '../MIT_spines/cell_templates'
	pyr_template_name 	= 'cADpyr230_L4_PC_f15e35e578' # 'whole_cell'
	pyr_morph_path 		= '{}/morphology'.format(pyr_template_path) # 'L5PC/'

	PV_template_path 	= 'EPFL_models/L4_LBC_cNAC187_1'
	PV_template_name 	= 'cNAC187_L4_LBC_990b7ac7df'
	PV_morph_path 		= '{}/morphology/'.format(PV_template_path)

	PV_input_weight  = 0.4
	PV_to_Pyr_weight = 0.5
	Pyr_input_weight = 0.5

	pyr_type = 'L4_PC'
	not_PVs = ['PC', 'SP', 'SS', 'MC', 'BTC', 'L1']

	PV_input_delay 	= 0 # TEMPORARY: CHANGE THIS
	Pyr_input_delay = PV_input_delay+1 # TEMPORARY: CHECK THIS
	PV_output_delay = Pyr_input_delay+5 # TEMPORARY: CHECK THIS
	freq1 = 6666
	freq2 = 9600
	simulation_time = 154*1000 # in ms
	spike_threshold = 0

	filenames = {'cell_details': 'thalamocortical_Oren/thalamic_data/cells_details.pkl', 
			 'thalamic_locs': 'thalamocortical_Oren/thalamic_data/thalamic_axons_location_by_gid.pkl',
			 'thalamic_connections': 'thalamocortical_Oren/thalamic_data/thalamo_cortical_connectivity.pkl',
			 'thalamic_activations_6666': 'thalamocortical_Oren/SSA_spike_times/input6666_106746.dat',
			 'thalamic_activations_9600': 'thalamocortical_Oren/SSA_spike_times/input9600_109680.dat',
			 'pyr_connectivity': 'thalamocortical_Oren/pyramidal_connectivity_num_connections.p',
			 'cell_type_gids': 'thalamocortical_Oren/thalamic_data/cell_type_gids.pkl',
			 'stim_times': 'thalamocortical_Oren/SSA_spike_times/stim_times.p',
			 'thalamic_activations': {6666: 'thalamocortical_Oren/SSA_spike_times/input6666_106746.dat', 9600: 'thalamocortical_Oren/SSA_spike_times/input9600_109680.dat'}}

	cell_type_gids 	= cPickle.load(open(filenames['cell_type_gids'],'rb'))     
	thal_connections = pd.read_pickle(filenames['thalamic_connections'])
	thalamic_locations = pd.read_pickle(filenames['thalamic_locs'])
initializeParameters() # Keep here

def getChosenGIDs(pyr_gids, PV_gids, freq1, freq2, min_freq=4000, df_dx=3.5, filenames=filenames):
	# df/dx is n units [octave/mm]
	thalamic_locations   = pd.read_pickle(filenames['thalamic_locs'])
	cell_details 	 	 = pd.read_pickle(filenames['cell_details'])
	all_pyr_connectivity = pd.read_pickle(filenames['pyr_connectivity'])
	
	min_freq_loc 	 = min(thalamic_locations.x) 

	def get_AxonLoc(freq, min_freq, min_freq_loc, df_dx=df_dx):

		dOctave = log(freq / min_freq, 2) # In octaves
		d_mm =  dOctave / df_dx			  # In millimeters
		d_microne = d_mm * 1000 # In micrones
		freq_loc = min_freq_loc + d_microne

		return freq_loc

	def get_pyramidal(pyr_gids, freq1_loc, freq2_loc, cell_details):		
		mid_loc = np.mean([freq1_loc, freq2_loc])

		dists = [abs(cell_details.loc[gid].x-mid_loc) for gid in pyr_gids]	
		sorted_idx = [dists.index(i) for i in sorted(dists)]	

		chosen_pyr_gid = pyr_gids[sorted_idx[0]]
		return chosen_pyr_gid

	def get_PVs(PV_gids, chosen_pyr_gid, all_pyr_connectivity):

		pyr_connectivity = all_pyr_connectivity[chosen_pyr_gid]

		chosen_PV_gids = []
		chosen_PV_n_contacts = {}
		for gid in pyr_connectivity:
			if int(gid) in PV_gids:
				chosen_PV_gids.append(int(gid))
				chosen_PV_n_contacts[int(gid)] = pyr_connectivity[gid]

		return chosen_PV_gids, chosen_PV_n_contacts

	freq1_loc = get_AxonLoc(freq1, min_freq, min_freq_loc)
	freq2_loc = get_AxonLoc(freq2, min_freq, min_freq_loc)
	
	# Pyramidal
	chosen_pyr_gid = get_pyramidal(pyr_gids, freq1_loc, freq2_loc, cell_details)

	# PV
	chosen_PV_gids, chosen_PV_n_contacts = get_PVs(PV_gids, chosen_pyr_gid, all_pyr_connectivity)

	return chosen_pyr_gid, chosen_PV_gids, chosen_PV_n_contacts

def plotThalamicResponses(stimuli, freq1, freq2, thalamic_locations, run_function=False):
	if run_function:
		stim_ax = stimuli[freq1].axonResponses(thalamic_locations, color='red')
		stim_ax = stimuli[freq2].axonResponses(thalamic_locations, color='blue', h_ax=stim_ax)

		stimuli[freq1].tonotopic_location(thalamic_locations, color='red', h_ax=stim_ax)
		stimuli[freq2].tonotopic_location(thalamic_locations, color='blue', h_ax=stim_ax)

		stim_ax.set_title('Thalamic Axons (Connected to Chosen Pyramidal) Firing Rate for 2 Frequencies\nPyramidal GID: {}, Spontaneous Axon FR: {}'.format(chosen_pyr, 0.5))
		stim_ax.set_xlim([65, 630])

		return stim_ax

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

activated_filename = filenames['thalamic_activations_6666']
# activated_filename = filenames['thalamic_activations_9600']
activated_standard_freq = freq1*(str(freq1) in activated_filename) + freq2*(str(freq2) in activated_filename)

def getResponsiveAxons(standard_freqs, activations_filename, stim_times_filename, simulation_time=simulation_time, cutoff_freq=1.8):
	def get_activations(activations_filename):
		temp_data = [i.strip().split() for i in open(activations_filename).readlines()]
		activations = [] 
		for i in range(len(temp_data)): 
			if temp_data[i][0].replace('.', '').isdigit():
				activations.append([float(temp_data[i][0]), int(float(temp_data[i][1]))]) 

		return activations

	stim_times_dict  = pd.read_pickle(stim_times_filename)
	responses_dict = {}
	responsive_axons = {f: [] for f in standard_freqs}	
	simulation_time_secs = simulation_time/1000 # Convert to seconds

	for freq in standard_freqs:
		
		activations = get_activations(activations_filename[freq])
		axons = list(set([i[1] for i in activations]))                                                                                  

		total_FRs = {a: {'times': [], 'FR': 0} for a in axons}
		for a in activations:
			time = a[0]
			gid = a[1]
			total_FRs[gid]['times'].append(time)


		for gid in axons:
			total_FRs[gid]['FR'] = len(total_FRs[gid]['times']) / simulation_time_secs

			if total_FRs[gid]['FR'] >= cutoff_freq:
				responsive_axons[freq].append(gid)

	return responsive_axons

cutoff_freq = 1.8
if 'responsive_axons_cutoff_{}.p'.format(cutoff_freq) in os.listdir('thalamocortical_Oren/thalamic_data'):
	responsive_axons_dict = cPickle.load(open('thalamocortical_Oren/thalamic_data/responsive_axons_cutoff_{}.p'.format(cutoff_freq), 'rb'))
else:
	responsive_axons_dict = getResponsiveAxons([6666, 9600], filenames['thalamic_activations'], filenames['stim_times'])
	cPickle.dump(responsive_axons, open('thalamocortical_Oren/thalamic_data/responsive_axons_cutoff_{}.p'.format(cutoff_freq), 'wb'))
responsive_axons = responsive_axons_dict[activated_standard_freq]

# ===============================================  Choose GIDs  ===============================================
print('\n========== Choosing GIDs from BlueBrain Database (pyramidal between {}Hz and {}Hz) =========='.format(freq1, freq2))

pyr_GIDs = cell_type_gids[pyr_type]
PV_types = []
for t in cell_type_gids.keys():
	if all([i not in t for i in not_PVs]):
		PV_types.append(t)
PV_GIDs = [j for i in [cell_type_gids[t] for t in PV_types] for j in i]

chosen_pyr, chosen_PV, chosen_PV_n_contacts = getChosenGIDs(pyr_GIDs, PV_GIDs, freq1, freq2) # chosen_pyr==gid, chosen_V==[[gid, no_contancts],...]

# ===============================================  Create Cell Populations  ===============================================

print('\n========== Creating pyramidal cell ==========')
Pyr_pop = Population('Pyr', pyr_morph_path, pyr_template_path, pyr_template_name)
Pyr_pop.addCell()
Pyr_pop.name_to_gid['Pyr0'] = chosen_pyr

print('\n========== Creating PV population ==========')
n_PV  		  = len(chosen_PV)
PV_pop = Population('PV', PV_morph_path, PV_template_path, PV_template_name)
for i in tqdm(range(n_PV)):
	PV_pop.addCell()
	PV_pop.name_to_gid['PV%i'%i] = chosen_PV[i]

# ==============================================  Morphology Visualization  ==============================================

for i in range(n_PV):  
	PV_pop.moveCell(PV_pop.cells['PV{}'.format(i)]['cell'], (i*350)-(100*(n_PV+1)), -500, 0)  

# ==============================================  Stimulus Analysis  ==============================================
# Only axons who connect to the chosen Pyramidal cell
stimuli, connecting_gids = {}, []

# Find thalamic GIDs connecting the the pyramidal cell
for con in thal_connections.iterrows():
	if con[1].post_gid == chosen_pyr:
		connecting_gids.append(con[1].pre_gid) # [presynaptic gid, no. of contacts]
stimuli[freq1] = Stimulus(freq1, freq2, filenames['stim_times'], filenames['thalamic_activations_6666'], axon_gids=connecting_gids)
stimuli[freq2] = Stimulus(freq2, freq1, filenames['stim_times'], filenames['thalamic_activations_9600'], axon_gids=connecting_gids)

plot_thalamic_responses = False
stim_ax = plotThalamicResponses(stimuli, freq1, freq2, thalamic_locations, run_function=plot_thalamic_responses)
# ==============================================  Connect Populations with Synapses and Inputs  ==============================================

print('\n========== Connecting thalamic inputs to PV (standard: {}Hz, input weight: {}uS) =========='.format(activated_standard_freq, PV_input_weight))
PV_events = [] 
for i, PV_cell_name in enumerate(tqdm(PV_pop.cells)):
	PV_gid = PV_pop.name_to_gid[PV_cell_name]

	# Custom initialization: insert event to queue (if h.FInitializeHandler no called, event is erased by h.run() because it clears the queue)
	PV_pop.addInput(PV_cell_name, PV_gid, weight=PV_input_weight, thalamic_activations_filename=activated_filename, thalamic_connections_filename=filenames['thalamic_connections'])	
	for axon in list(PV_pop.inputs[PV_cell_name].keys()):
		stim_times = PV_pop.inputs[PV_cell_name][axon]['stim_times']
		for i in range(len(PV_pop.inputs[PV_cell_name][axon]['netcons'])):	
			for time in stim_times:
				PV_events.append(h.FInitializeHandler('nrnpython("PV_pop.inputs[\'{}\'][\'{}\'][\'netcons\'][{}].event({})")'.format(PV_cell_name, axon, i, time+PV_input_delay)))
# del i, time, axon

print('\n========== Connecting thalamic inputs to Pyramidal cell (standard: {}Hz, input weight: {}uS) =========='.format(activated_standard_freq, Pyr_input_weight))
Pyr_pop.addInput(list(Pyr_pop.cells.keys())[0], chosen_pyr, weight=Pyr_input_weight, thalamic_activations_filename=activated_filename, thalamic_connections_filename=filenames['thalamic_connections'])
# IMPORTANT NOTIC: MUST be defined outside of functino or else hoc doesn't recognize netcon name!
pyr_events = []
for axon in Pyr_pop.inputs['Pyr0']:
	stim_times = Pyr_pop.inputs['Pyr0'][axon]['stim_times']
	for i in range(len(Pyr_pop.inputs['Pyr0'][axon]['netcons'])):		
		for time in stim_times:
			pyr_events.append(h.FInitializeHandler('nrnpython("Pyr_pop.inputs[\'Pyr0\'][\'{}\'][\'netcons\'][{}].event({})")'.format(axon, i, time+Pyr_input_delay)))
# del i, time, axon


# print('Connecting PV population to Pyramidal cell (connection weight: {}uS)'.format(PV_to_Pyr_weight))
# Pyr_soma = Pyr_pop.cells['Pyr0']['cell'].soma[0]
# for PV_cell_name in PV_pop.cells:
# 	temp_PV_gid = PV_pop.name_to_gid[PV_cell_name]
# 	PV_pop.connectCells(PV_cell_name, [Pyr_soma], chosen_PV_n_contacts[temp_PV_gid], 'random', weight=PV_to_Pyr_weight, delay=PV_output_delay, threshold=spike_threshold) # Adds self.connections to Population
# print('\n***WARNING: Assuming isopotential soma and perisomatic PV connections: all PV synapses are placed on soma(0.5)')

# ============================================== Plot example responses ==============================================

record_PV_dendrite = True
recorded_segment = []
if record_PV_dendrite:
	dend = [i for i in PV_pop.cells['PV3']['basal_dendrites'] if 'dend[42]' in i.name()][0]
	seg = 0.5

	recorded_segment = dend(seg)

t, dend_v = RunSim(tstop = 10000, record_specific=recorded_segment)

def Wehr_Zador(population, cell_name, title_, exc_weight=0, inh_weight=0, standard_freq=activated_standard_freq, input_pop_outputs=None, take_before=20, take_after=155):

	fig, axes = plt.subplots(3, 1)
	fig.subplots_adjust(hspace=0.34, bottom=0.08, top=0.9) 
	# times = stimuli[activated_standard_freq].stim_times_all
	# times  = [i[0] for i in times if i[1]==standard_freq and i[0]<h.tstop] # Take only times of standard frequency stimulus
	
	times = stimuli[standard_freq].stim_times_standard
	times = [i for i in times if i<h.tstop]

	cut_vec = lambda vec, start_idx, end_idx: [vec[i] for i in range(start_idx, end_idx)]
	
	spike_count = 0
	all_AMPA, all_NMDA, all_GABA 		= [], [], []
	all_g_AMPA, all_g_NMDA, all_g_GABA 	= [], [], []
	print('Analyzing conductances and currents')
	for T in tqdm(times):
		idx1 = (np.abs([i-(T-take_before) for i in t])).argmin()
		idx2 = (np.abs([i-(T+take_after) for i in t])).argmin()
		
		v_vec = cut_vec(population.cells[cell_name]['soma_v'], idx1, idx2)
		if any([i>=spike_threshold for i in v_vec]):
			spike_count += 1
		# t_vec = [i*h.dt for i in range(0, len(v_vec))]
		t_vec = cut_vec(t, idx1, idx2)
		t_vec = [i-t_vec[0] for i in t_vec]

		mean_AMPA, mean_NMDA, mean_GABA  		= [], [], []
		mean_g_AMPA, mean_g_NMDA, mean_g_GABA 	= [], [], []

		# E to pyramidal

		for axon in population.inputs[cell_name]:
			AMPA_vec = [cut_vec(vec, idx1, idx2) for vec in population.inputs[cell_name][axon]['i_AMPA']]
			NMDA_vec = [cut_vec(vec, idx1, idx2) for vec in population.inputs[cell_name][axon]['i_NMDA']]	
			g_AMPA_vec = [cut_vec(vec, idx1, idx2) for vec in population.inputs[cell_name][axon]['g_AMPA']]
			g_NMDA_vec = [cut_vec(vec, idx1, idx2) for vec in population.inputs[cell_name][axon]['g_NMDA']]
			
			mean_AMPA.append(np.sum(AMPA_vec, axis=0))			
			mean_NMDA.append(np.sum(NMDA_vec, axis=0))
			mean_g_AMPA.append(np.sum(g_AMPA_vec, axis=0))
			mean_g_NMDA.append(np.sum(g_NMDA_vec, axis=0))

		# I to pyramidal
		if input_pop_outputs:
			for pre_PV in input_pop_outputs:
				for sec in input_pop_outputs[pre_PV]:
					GABA_vec = [cut_vec(vec, idx1, idx2) for vec in input_pop_outputs[pre_PV][sec]['syn_i']]
					g_GABA_vec = [cut_vec(vec, idx1, idx2) for vec in input_pop_outputs[pre_PV][sec]['syn_g']]

					mean_GABA.append(np.sum(GABA_vec, axis=0))
					mean_g_GABA.append(np.sum(g_GABA_vec, axis=0))

		mean_AMPA = np.sum(mean_AMPA, axis=0)
		mean_NMDA = np.sum(mean_NMDA, axis=0)
		mean_GABA = np.sum(mean_GABA, axis=0)
		mean_g_AMPA = np.sum(mean_g_AMPA, axis=0)
		mean_g_NMDA = np.sum(mean_g_NMDA, axis=0)
		mean_g_GABA = np.sum(mean_g_GABA, axis=0)
		
		all_AMPA.append(mean_AMPA)
		all_NMDA.append(mean_NMDA)
		all_GABA.append(mean_GABA)

		all_g_AMPA.append(mean_g_AMPA)
		all_g_NMDA.append(mean_g_NMDA)
		all_g_GABA.append(mean_g_GABA)

		axes[0].plot(t_vec, v_vec, 'k', LineWidth=0.7)

		if T==times[0]:
			axes[0].legend(['%s soma v'%cell_name], loc='upper right')
			# ax2.legend(['AMPA', 'NMDA'], loc='upper right')
	all_AMPA = np.mean(all_AMPA[:-1], axis=0)
	all_NMDA = np.mean(all_NMDA[:-1], axis=0)
	all_GABA = np.mean(all_GABA[:-1], axis=0)
	all_g_AMPA = np.mean(all_g_AMPA[:-1], axis=0)
	all_g_NMDA = np.mean(all_g_NMDA[:-1], axis=0)
	all_g_GABA = np.mean(all_g_GABA[:-1], axis=0)

	plt.suptitle('{} Cell ({} spikes out of {})'.format(title_, spike_count, len(times)))
	axes[0].axvline(take_before, LineStyle='--', color='gray')
	axes[1].axvline(take_before, LineStyle='--', color='gray')
	axes[2].axvline(take_before, LineStyle='--', color='gray')

	axes[0].set_title('Overlay of Somatic Responses to %sHz Simulus (locked to stimulus prestntation)'%standard_freq)
	axes[0].set_ylabel('V (mV)')
	axes[0].set_xlim([0, take_before+take_after])

	t_vec = [i*h.dt for i in range(0, len(all_AMPA))]
	# axes[1].plot(t_vec, all_AMPA, 'r', LineWidth=0.7, label='AMPA')
	# axes[1].plot(t_vec, all_NMDA, 'g', LineWidth=0.7, label='NMDA')
	axes[1].plot(t_vec, [all_AMPA[i]+all_NMDA[i] for i in range(len(all_AMPA))], 'purple', label='i$_{AMPA}$ + i$_{NMDA}$')
	if input_pop_outputs:
		if list(input_pop_outputs.values())[0]:
			axes[1].plot(t_vec, all_GABA, 'b', label='i$_{GABA}$')
			axes[1].plot(t_vec, [all_GABA[i]+all_AMPA[i]+all_NMDA[i] for i in range(len(all_GABA))], label='i$_{tot}$')
		axes[1].legend(loc='upper right')
	axes[1].set_title('Mean Synaptic Currents')
	axes[1].set_ylabel('I (nA)')
	axes[1].set_xlim([0, take_before+take_after])

	# axes[2].plot(t_vec, [i*1000 for i in all_g_AMPA], 'r', LineWidth=0.7, label='AMPA')
	# axes[2].plot(t_vec, [i*1000 for i in all_g_NMDA], 'g', LineWidth=0.7, label='NMDA')
	axes[2].plot(t_vec, [1000*(all_g_AMPA[i]+all_g_NMDA[i]) for i in range(len(all_g_AMPA))], 'purple', label='g$_{AMPA}$ + g$_{NMDA}$ (g$_{max}$=%.1f)'%exc_weight)
	if input_pop_outputs:
		if list(input_pop_outputs.values())[0]:
			axes[2].plot(t_vec, [i*1000 for i in all_g_GABA], 'b', label='g$_{GABA}$ (g$_{max}$=%.1f)'%inh_weight)
			axes[2].plot(t_vec, [1000*(all_g_GABA[i]+all_g_AMPA[i]+all_g_NMDA[i]) for i in range(len(all_g_GABA))], label='g$_{tot}$')

	axes[2].legend(loc='upper right')
	axes[2].set_title('Mean Synaptic Conductances')
	axes[2].set_ylabel('g (nS)')
	axes[2].set_xlabel('T (ms)')
	axes[2].set_xlim([0, take_before+take_after])	

	os.system('say "Everything plotted"')
Wehr_Zador(PV_pop, 'PV0', 'PV', input_pop_outputs=None, exc_weight=PV_input_weight)
Wehr_Zador(Pyr_pop, 'Pyr0', 'Pyramidal', input_pop_outputs=PV_pop.outputs, exc_weight=Pyr_input_weight, inh_weight=PV_to_Pyr_weight)

def PlotSomas(Pyr_pop, PV_pop, t, stimulus, which_PV='PV0'):
	plt.figure()
	plt.plot(t, Pyr_pop.cells['Pyr0']['soma_v'], label='Pyr ({})'.format(chosen_pyr))
	plt.plot(t, PV_pop.cells[which_PV]['soma_v'], label='PV ({})'.format(PV_pop.name_to_gid[which_PV]))

	stan_times = stimulus.stim_times_standard
	for s in stan_times:  
		if s<h.tstop:  
			if s==stan_times[0]:
				plt.axvline(s, LineStyle='--', color='k', alpha=0.5, label='Standard Stimulus') 
			else:
				plt.axvline(s, LineStyle='--', color='k', alpha=0.5)

	dev_times = stimulus.stim_times_deviant
	for s in dev_times:  
		if s<h.tstop:  
			if s==dev_times[0]:
				plt.axvline(s, LineStyle='--', color='g', alpha=0.5, label='Deviant Stimulus') 
			else:
				plt.axvline(s, LineStyle='--', color='g', alpha=0.5)

	plt.legend()
	plt.title('Example of Pyramidal and PV Responses to 2 tones ({}Hz, {}Hz)\n(at tonotopical position between tones, standard: {}'.format(freq1, freq2, activated_standard_freq)) 
	plt.xlabel('T (ms)')
	plt.ylabel('V (mV)')
	plt.xlim([0, h.tstop])
PlotSomas(Pyr_pop, PV_pop, t, stimuli[activated_standard_freq], which_PV='PV3')

if record_PV_dendrite:
	plt.figure()
	plt.plot(t, dend_v, label='dend')
	plt.plot(t, PV_pop.cells['PV3']['soma_v'], label='soma')
	plt.legend()
	plt.title('Voltage in {}({})'.format(dend, seg))
	plt.xlabel('T (ms)')
	plt.ylabel('V (mV)')
# ============================================== Analyze Connectivity ==============================================

analyze_connectivity = False
if analyze_connectivity:
	_, bar_ax = plt.subplots()
	pyr_connectivity = pd.read_pickle(filenames['pyr_connectivity'])
	PV_contacts = {pyr: [] for pyr in pyr_connectivity}
	for pyr in pyr_connectivity:
		for contact in pyr_connectivity[pyr]:
			if contact in PV_GIDs:
				PV_contacts[pyr].append(pyr_connectivity[pyr][contact])
	mean_pre_PV = np.mean([len(PV_contacts[i]) for i in PV_contacts])                                                                                                                 
	mean_PV_contacts = np.mean([sum(PV_contacts[i]) for i in PV_contacts]) 
	bar_ax.bar(['Presynaptic PV', 'No. PV contacts'], [mean_pre_PV, mean_PV_contacts], color='g')
	bar_ax.set_title('Presynaptic PV connections to Pyramidal cells (PV defined as: {})'.format(PV_type))


	all_pyr_connectivity = pd.read_pickle(filenames['pyr_connectivity'])
	pyr_connectivity = all_pyr_connectivity[chosen_pyr]
	all_cons = {i: {'count':0,'contacts':0} for i in mtypes} 
	for con in pyr_connectivity: 
		if con in GIDS: 
			M = cell_details.loc[con].mtype 
			all_cons[M]['count']+=1 
			all_cons[M]['contacts']+=pyr_connectivity[con] 

	for i in list(all_cons.keys()): 
		if all_cons[i]['count']==0: 
			del all_cons[i] 
	for i in list(all_cons.keys()):
		if 'BTC' in i or 'L1' in i or 'MC' in i or 'PC' in i or 'SP' in i or 'SS' in i:
			del all_cons[i]

	sum=0;types=[] 
	for i in all_cons: 
		if 'L4' in i: 
			sum+=all_cons[i] 
			types.append(i)

'''
from neuron import h,gui
import matplotlib.pyplot as plt
dend=h.Section('dend')
dend.insert('pas')
h.v_init = dend.e_pas
syn1 = h.ProbAMPANMDA2_RATIO(dend(0.5))
NET1 = h.NetCon(None, syn1)
NET1.weight[0]=0.4
time1 = 50
syn2 = h.ProbAMPANMDA2_RATIO(dend(0.2))
NET2 = h.NetCon(None, syn2)
NET2.weight[0]=0.4
time2 = 100
times = [time1, time2]
NETS = [NET1, NET2]
events = []
for i in range(len(NETS)):
	T = times[i]
	print('time: {}, con: {}'.format(T, NETS[i]))
	events.append(h.FInitializeHandler('nrnpython("NETS[{}].event({})")'.format(i, T)))
h.tstop=500;t=h.Vector();t.record(h._ref_t)
v=h.Vector();v.record(dend(0.5)._ref_v) 
h.run();plt.plot(t,v)  


events2 = []
syn3 = h.ProbAMPANMDA2_RATIO(dend(0.5))
NET3 = h.NetCon(None, syn3)
NET3.weight[0]=0.4
time3 = 300
syn4 = h.ProbAMPANMDA2_RATIO(dend(0.2))
NET4 = h.NetCon(None, syn4)
NET4.weight[0]=0.4
time4 = 400
times2 = [time3, time4]
NETS2 = [NET3, NET4]
for con in NETS2:
	T = times2[NETS2.index(con)]
	print('time: {}, con: {}'.format(T, con))
	events2.append(h.FInitializeHandler('nrnpython("con.event({})")'.format(T)))
h.tstop=500;t=h.Vector();t.record(h._ref_t)
v=h.Vector();v.record(dend(0.5)._ref_v) 
h.run();plt.plot(t,v)  









for axon in ['thalamic_gid_221076', 'thalamic_gid_221082']:
	stim_times = Pyr_pop.inputs['Pyr0'][axon]['stim_times']
	for i in range(2):
		for time in stim_times:
			pyr_events.append(h.FInitializeHandler('nrnpython("Pyr_pop.inputs[\'Pyr0\'][\'{}\'][\'netcons\'][{}].event({})")'.format(axon, i, time+delay)))



pyr_events = []
for axon in ['thalamic_gid_221076', 'thalamic_gid_221082']:
	stim_times = Pyr_pop.inputs['Pyr0'][axon]['stim_times']
	for con in Pyr_pop.inputs['Pyr0'][axon]['netcons'][:2]:
		for time in stim_times:
			pyr_events.append(h.FInitializeHandler('nrnpython("con.event({})")'.format(time+delay)))
'''
'''
def event_python_function(): #for this to work you must name every variable differently (i.e. time1, time2, etc..  or stim_times[i] inside function)
	global current_netcon, current_time, delay
	current_netcon.event(current_time+delay) 

'''
'''
cvode = h.CVode()
temp_vec = h.Vector()
cvode.spike_stat(temp_vec)
n_NetCons = int(temp_vec.x[1])
events = []
for i in range(n_NetCons):
	events.append(h.FInitializeHandler('nrnpython("NetCon[{}].event({})")'.format(i, time+delay)))
'''

# ==============================================  Run Simulation & Plot Results  ==============================================
# PYR_cell, PV_pop = RunSim(PYR_cell, PV_pop)

# ==============================================  Connectivity Analysis  ==============================================
def analyzeConnectivityFromThalamus(which_layer, which_mtype, filenames=filenames):

	cell_details = pd.read_pickle(filenames['cell_details'])
	thal_connections = pd.read_pickle(filenames['thalamic_connections'])

	if which_layer=='all':
		which_layer = ''
	else:
		if which_layer.isdigit():
			which_layer = 'L' + which_layer

	cell_type = which_layer + '_' + which_mtype
	GIDs = [cell_details.index[i] for i in range(len(cell_details)) if cell_details.iloc[i].mtype==cell_type]

	# How many contacts each thalamic axon *makes* on each of the specified cell
	contacts_per_axon = [thal_connections.iloc[i].contacts for i in range(len(thal_connections)) if thal_connections.iloc[i].post_gid in GIDs]

	# pyramidal and all thalamic axons connected to it
	per_gid = {gid: [] for gid in GIDs}
	for i in range(len(thal_connections)):
		POST = thal_connections.iloc[i].post_gid
		if POST in GIDs:
			per_gid[POST].append(thal_connections.iloc[i].contacts)

	axons_per_gid = {gid: len(per_gid[gid]) for gid in GIDs}
	contacts_per_incoming_axon = {gid: np.mean(per_gid[gid]) for gid in GIDs if axons_per_gid[gid]!=0}

	stats = {'contacts_per_axon': {'mean_out_contacts': np.mean(contacts_per_axon), # How many contacts a thalamic axon makes on the specified cell on average
						  'std': np.std(contacts_per_axon)},
			 'per_cell': {'mean_incoming_contacts': np.mean(list(contacts_per_incoming_axon.values())), # How many contacts *on* cell from each axon (on average)
			 			  'std': np.std(list(contacts_per_incoming_axon.values())),
			 			  'mean_incoming_axons': np.mean(list(axons_per_gid.values()))}
			}

	return stats

# thalamo_to_pyr_stats = analyzeConnectivityFromThalamus('L23', 'PC')
# thalamo_to_PV_stats  = analyzeConnectivityFromThalamus('L23', 'LBC')

# # Number of thalamic axons contacting each cell
# n_axons_to_pyr = thalamo_to_pyr_stats['per_cell']['mean_incoming_axons']
# n_axons_to_PV  = thalamo_to_PV_stats['per_cell']['mean_incoming_axons']

# # Number of contacts a thalamic axon makes on each cell
# n_contacts_to_pyr = thalamo_to_pyr_stats['per_cell']['mean_incoming_contacts']
# n_contacts_to_PV  = thalamo_to_PV_stats['per_cell']['mean_incoming_contacts']

'''
GENERAL- EPFL cell models:
	- Template: under the name template.py. In the template the morphology is loaded, biophys() is called and 
				synapses are added (if input to template is 1). <morphology.hoc, biophysics.hoc, synapses/synapses.hoc>
	
	- Adding synapses: synaptic inputs to loaded cell can be enabled (if input to template is 1). "..all the synapses 
					   that cells from the specified m-type (?) make on the simulated cell will  become active". Each
					   presynaptic cell is represented by a Poisson spike train. Default firing rate is set to 10Hz,
					   but can be changed (I need to figure out how, outside of the GUI).
	
	- ?? Deleteing axons ??: In template (for now I know about LBC-cNAC), all axons (many) are deleted and replaced
							 by 1 short axon ("stub axon" in documentation). WHY?

	- ?? How to model thalamic input ??: Eli said to look at Oren's input.

PV Population:
	I will start with the following model from nmc-portal (EPFL), based on Markram et al., 2004 (interneuron review): 
	- L23 
	- LBC (large basket cell) 
	- cNAC (classic non-accommodating)

	* Basket cells typically express PV 
	* cNAC look the most simple firing pattern. 
	* There are 5 different models for this in  nmc portal. For simplicity I downloaded number1  for now.
	* "Most PV+ neurons are chandelier and basket cells, which make powerful inhibitory synapses onto the 
		somatic and perisomatic regions of pyramidal cells" (Moore and Wehr, 2013)
	* "... PV cells, the delay is very short (1–2 ms), creating a limited temporal 'window of opportunity'
		(Alonso and Swadlow, 2005; Pinto et al., 2000) for PCs to summate afferent inputs" (Tremblay, Lee, and Rudy, 2016)

Pyramidal cell:
	Currently I am using Itay Hay's L5PC model. Possibly use EPFL model also here- check which layer.

	! Make sure my biophys is like Itay Hay's !
'''






'''
import matplotlib.pyplot as plt
from neuron import gui,h
h('create dend1')
dend1 = h.dend1
dend1.insert('pas')
h('create dend2')
dend2 = h.dend2
dend2.insert('pas')


syn1 = h.ProbAMPANMDA2_RATIO(0.5, sec=dend1)
netstim1 = h.NetStim(0.5, sec=dend1)
netstim1.start = 100
netcon1 = h.NetCon(netstim1, syn1)
netcon1.weight[0] = 0.5

syn2 = h.ProbAMPANMDA2_RATIO(0.5, sec=dend2)
netcon2 = h.NetCon(dend1(0.5)._ref_v, syn2, sec=dend1)
netcon2.weight[0] = 0.5
netcon2.delay = 1
netcon2.threshold = -69.995

v1 = h.Vector(); v1.record(dend1(0.5)._ref_v)
v2 = h.Vector(); v2.record(dend2(0.5)._ref_v)
g_AMPA = h.Vector(); g_AMPA.record(syn2._ref_g_AMPA)
g_NMDA = h.Vector(); g_NMDA.record(syn2._ref_g_NMDA)

t = h.Vector(); t.record(h._ref_t)
h.tstop = 250
h.v_init = -70
h.finitialize()
h.run()

plt.plot(t, v1)
plt.plot(t, v2)
plt.figure()
plt.plot(t,g_AMPA)
plt.plot(t, g_NMDA)
'''










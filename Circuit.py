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
from scipy.stats import ttest_ind
from Connectivity import Connectivity

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
	global pyr_type, not_PVs, SOM_types
	global PV_input_delay, Pyr_input_delay, PV_output_delay, freq1, freq2, simulation_time, spike_threshold
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
	SOM_types = ['MC']

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
			 'thalamic_activations_6666': 'thalamocortical_Oren/SSA_spike_times/input6666_artificial.p',
			 # 'thalamic_activations_6666': 'thalamocortical_Oren/SSA_spike_times/input6666_successes_by_gid.p',
			 # 'thalamic_activations_9600': 'thalamocortical_Oren/SSA_spike_times/input9600_109680.p',
			 'thalamic_activations_9600': 'thalamocortical_Oren/SSA_spike_times/input9600_by_gid.p',
			 'pyr_connectivity': 'thalamocortical_Oren/pyramidal_connectivity_num_connections.p',
			 'cortex_connectivity': 'thalamocortical_Oren/cortex_connectivity_num_connections.p',       
			 'cell_type_gids': 'thalamocortical_Oren/thalamic_data/cell_type_gids.pkl',
			 'stim_times': 'thalamocortical_Oren/SSA_spike_times/stim_times.p',
			 'thalamic_activations': {6666: 'thalamocortical_Oren/SSA_spike_times/input6666_106746.dat', 9600: 'thalamocortical_Oren/SSA_spike_times/input9600_109680.dat'}}

	# cell_type_gids 	= cPickle.load(open(filenames['cell_type_gids'],'rb'))     
	thal_connections = pd.read_pickle(filenames['thalamic_connections'])
	thalamic_locations = pd.read_pickle(filenames['thalamic_locs'])
initializeParameters() # Keep here

'''
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
'''

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

def get_activations(activations_filename):
	temp_data = [i.strip().split() for i in open(activations_filename).readlines()]
	activations = [] 
	for i in range(len(temp_data)): 
		if temp_data[i][0].replace('.', '').isdigit():
			activations.append([float(temp_data[i][0]), int(float(temp_data[i][1]))]) 

	return activations

def getResponsiveAxons(standard_freqs, activations_filename, stim_times_filename, simulation_time=simulation_time, cutoff_freq=1.8):

	stim_times_dict  = pd.read_pickle(stim_times_filename)
	responses_dict = {}
	responsive_axons = {f: [] for f in standard_freqs}	
	simulation_time_secs = simulation_time/1000 # Convert to seconds

	for freq in standard_freqs:
		
		# activations = get_activations(activations_filename[freq])
		activations = cPickle.load(open(activations_filename, 'rb'))
		# axons = list(set([i[1] for i in activations]))                                                                                  
		axons = [i for i in activations]

		total_FRs = {a: {'times': [], 'FR': 0} for a in axons}
		for a in activations:
			# time = a[0]
			# gid = a[1]
			# total_FRs[gid]['times'].append(time)
			gid = a
			times = activations[a]
			total_FRs[gid]['times'] = times
			
		for gid in axons:
			total_FRs[gid]['FR'] = len(total_FRs[gid]['times']) / simulation_time_secs

			if total_FRs[gid]['FR'] >= cutoff_freq:
				responsive_axons[freq].append(gid)

	return responsive_axons

# cutoff_freq = 1.8
# if 'responsive_axons_cutoff_{}.p'.format(cutoff_freq) in os.listdir('thalamocortical_Oren/thalamic_data'):
# 	responsive_axons_dict = cPickle.load(open('thalamocortical_Oren/thalamic_data/responsive_axons_cutoff_{}.p'.format(cutoff_freq), 'rb'))
# else:
# 	responsive_axons_dict = getResponsiveAxons([6666, 9600], filenames['thalamic_activations'], filenames['stim_times'])
# 	cPickle.dump(responsive_axons, open('thalamocortical_Oren/thalamic_data/responsive_axons_cutoff_{}.p'.format(cutoff_freq), 'wb'))
# responsive_axons = responsive_axons_dict[activated_standard_freq]

# ===============================================  Choose GIDs  ===============================================
print('\n========== Choosing GIDs from BlueBrain Database (pyramidal between {}Hz and {}Hz) =========='.format(freq1, freq2))

# pyr_GIDs = cell_type_gids[pyr_type]
# PV_types = []
# for t in cell_type_gids.keys():
# 	if all([i not in t for i in not_PVs]):
# 		PV_types.append(t)
# PV_GIDs = [j for i in [cell_type_gids[t] for t in PV_types] for j in i]

# chosen_pyr, chosen_PV, chosen_PV_n_contacts = getChosenGIDs(pyr_GIDs, PV_GIDs, freq1, freq2) # chosen_pyr==gid, chosen_V==[[gid, no_contancts],...]

connectivity = Connectivity(pyr_type, [not_PVs, 'exclude'], [SOM_types, 'include'], 
							cell_type_to_gids=filenames['cell_type_gids'],
							thalamic_locs=filenames['thalamic_locs'], 
							cell_details=filenames['cell_details'],
							cortex_connectivity=filenames['cortex_connectivity'])

chosen_pyr = connectivity.choose_GID_between_freqs(connectivity.pyr_GIDs)
chosen_PV, chosen_PV_gids = connectivity.choose_GID_by_post(connectivity.PV_GIDs, connectivity.pyr_GIDs)

# No need to chooes SOM GIDs because they don't get thalamic input?


# ===============================================  Create Cell Populations  ===============================================

print('\n========== Creating pyramidal cell ==========')
Pyr_pop = Population('Pyr', pyr_morph_path, pyr_template_path, pyr_template_name, verbose=False)
Pyr_pop.addCell()
Pyr_pop.name_to_gid['Pyr0'] = chosen_pyr

print('\n========== Creating PV population ==========')
n_PV   = len(chosen_PV)
PV_pop = Population('PV', PV_morph_path, PV_template_path, PV_template_name, verbose=False)
for i in tqdm(range(n_PV)):
	PV_pop.addCell()
	PV_pop.name_to_gid['PV%i'%i] = chosen_PV[i]

print('\n========== Creating SOM population ==========')
n_SOM  = 9
SOM_pop = Population('SOM', SOM_morph_path, SOM_template_path, SOM_template_name, verbose=False)
for i in tqdm(range(SOM_pop)):
	SOM_pop.addCell()

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

connect_interneurons = False
if connect_interneurons:
	print('Connecting PV population to Pyramidal cell (connection weight: {}uS)'.format(PV_to_Pyr_weight))
	Pyr_soma = Pyr_pop.cells['Pyr0']['cell'].soma[0]
	for PV_cell_name in (tqdm):
		temp_PV_gid = PV_pop.name_to_gid[PV_cell_name]
		PV_pop.connectCells(PV_cell_name, [Pyr_soma], chosen_PV_n_contacts[temp_PV_gid], 'random', weight=PV_to_Pyr_weight, delay=PV_output_delay, threshold=spike_threshold) # Adds self.connections to Population
	print('\n***WARNING: Assuming isopotential soma and perisomatic PV connections: all PV synapses are placed on soma(0.5)')
	
	SOM_to_PV_weight = PV_to_Pyr_weight * n_SOM
	for PV_cell in PV_pop.cells:
		temp_PV_soma = PV_pop.cells[PV_cell]['cell'].soma[0]
		SOM_pop.connectCells(...)

	print('Connecting SOM population to PV Population (connection weight: {}uS)'.format(SOM_to_PV_weight))
# ============================================== Plot example responses ==============================================

record_PV_dendrite = False
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
	
	# times = stimuli[standard_freq].stim_times_standard
	times = [i[0] for i in stimuli[standard_freq].stim_times_all]
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

# ============================================== Analyze Input vs. Response ==============================================

print('Analyzing input vs. Pyramidal somatic responses')
def InputvsResponse(Pyr_pop, input_pop, standard_freq, stimuli, axon_gids, take_before=0, take_after=100, activated_filename=activated_filename):

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
input_groups = InputvsResponse(Pyr_pop, PV_pop, 6666, stimuli, pyr_input_gids, take_before=0, take_after=100)





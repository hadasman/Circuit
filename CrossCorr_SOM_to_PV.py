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
from plotting_functions import plotThalamicResponses, Wehr_Zador, PlotSomas, plotFRs

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
activated_filename = filenames['thalamic_activations_6666']
stand_freq = freq1*(str(freq1) in activated_filename) + freq2*(str(freq2) in activated_filename)
dev_freq = freq1*(str(freq1) not in activated_filename) + freq2*(str(freq2) not in activated_filename)
spont_FR_PV = 1.5
spont_FR_SOM = 1.5
record_thalamic_syns = True

get_spike_times = lambda soma_v, thresh: [t[idx] for idx in [i for i in range(len(soma_v)) if (soma_v[i]>thresh) and (soma_v[i]>soma_v[i-1]) and (soma_v[i]>soma_v[i+1])]]

def get_GIDs(upload_from):

	chosen_GIDs = {}
	chosen_GIDs['pyr'] 			  = cPickle.load(open('{}/chosen_pyr.p'.format(upload_from), 'rb'))
	chosen_GIDs['PV'] 			  = cPickle.load(open('{}/chosen_PV.p'.format(upload_from), 'rb'))
	
	# high_input is SOM connecting to either the PV or the pyr that had maximal thalamic contacts
	chosen_GIDs['SOM'] = cPickle.load(open('{}/chosen_SOM_high_input.p'.format(upload_from), 'rb'))

	# between_freq is the GID chosen between the frequencies; has too few thalamic connections
	# chosen_GIDs['SOM'] 			  = cPickle.load(open('{}/chosen_SOM_between_freq.p'.format(upload_from), 'rb'))

	chosen_PV_n_contacts  		  = cPickle.load(open('{}/chosen_PV_n_contacts.p'.format(upload_from), 'rb'))

	thalamic_GIDs = {}
	thalamic_GIDs['to_pyr'] = cPickle.load(open('{}/connecting_gids_to_pyr.p'.format(upload_from), 'rb'))
	thalamic_GIDs['to_PV']  = cPickle.load(open('{}/connecting_gids_to_PV.p'.format(upload_from), 'rb'))
	
	thalamic_GIDs['to_SOM'] = cPickle.load(open('{}/connecting_gids_to_SOM_high_input.p'.format(upload_from), 'rb'))
	# thalamic_GIDs['to_SOM'] = cPickle.load(open('{}/connecting_gids_to_SOM_between_freq.p'.format(upload_from), 'rb'))

	return chosen_GIDs, chosen_PV_n_contacts, thalamic_GIDs

def CreatePopulations(n_pyr=0, n_PV=0, n_SOM=0):
	Pyr_pop, PV_pop, SOM_pop = [None]*3

	if n_pyr > 0:
		print('\n==================== Creating pyramidal cell (n = {}) ===================='.format(n_pyr))
		Pyr_pop = Population('Pyr', pyr_morph_path, pyr_template_path, pyr_template_name, verbose=False)
		Pyr_pop.addCell()
		Pyr_pop.name_to_gid['Pyr0'] = chosen_GIDs['pyr']

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
		where_Pyr_syns = 'synapse_locs_instantiations/thalamic_syn_locs_Pyr_basal.p'

		print('\n==================== Connecting thalamic inputs to Pyramidal cell (on {}, standard: {}Hz, input weight: {}uS) ===================='.format(where_Pyr_syns_str, stand_freq, Pyr_input_weight)); sys.stdout.flush()
		Pyr_pop.addInput('Pyr0', record_syns=record_thalamic_syns, where_synapses=where_Pyr_syns, weight=Pyr_input_weight, thalamic_activations_filename=activated_filename, connecting_gids=thalamic_GIDs['to_pyr'])

	if PV_pop:
		where_PV_syns = ['basal_dendrites', 'apical_dendrites']
		where_PV_syns_str = '{}'.format(str(where_PV_syns).split('[')[1].split(']')[0].replace(',', ' and').replace('\'',''))
		where_PV_syns = 'synapse_locs_instantiations/thalamic_syn_locs_PV_basal_apical.p'

		print('\n==================== Connecting thalamic inputs to PV cells (on {}, standard: {}Hz, input weight: {}uS) ===================='.format(where_PV_syns_str, stand_freq, PV_input_weight)); sys.stdout.flush()

		if PV_to_Pyr_source == 'voltage':
			for i, PV_cell_name in enumerate(tqdm(PV_pop.cells)):
				PV_pop.addInput(PV_cell_name, record_syns=record_thalamic_syns, where_synapses=where_PV_syns, weight=PV_input_weight, thalamic_activations_filename=activated_filename, connecting_gids=thalamic_GIDs['to_PV'][PV_pop.name_to_gid[PV_cell_name]])	

		elif PV_to_Pyr_source == 'spike_times':
			print('Loading PV spike times from file ({})'.format(filenames['PV_spike_times']))
			PV_spike_times = cPickle.load(open(filenames['PV_spike_times'], 'rb'))

			for PV_cell_name in PV_pop.cells:
				PV_pop.cells[PV_cell_name]['soma_v'] = PV_spike_times['cells'][PV_cell_name]['soma_v']

	if SOM_pop:

		where_SOM_syns = ['soma']
		where_SOM_syns_str = '{}'.format(str(where_SOM_syns).split('[')[1].split(']')[0].replace(',', ' and').replace('\'',''))
		# where_SOM_syns = 'synapse_locs_instantiations/thalamic_syn_locs_SOM_basal.p'
		
		print('\n==================== Connecting thalamic inputs to SOM cell (on {}, standard: {}Hz, input weight: {}uS) ===================='.format(where_SOM_syns_str, stand_freq, SOM_input_weight)); sys.stdout.flush()

		SOM_pop.addInput('SOM0', record_syns=record_thalamic_syns, where_synapses=where_SOM_syns, weight=SOM_input_weight, thalamic_activations_filename=activated_filename, connecting_gids=thalamic_GIDs['to_SOM']) 

def CreatePoisson(spont_FR, tstop, bins=5, verbose=True, rnd_seed=None):
	spike_times = []  
	t = 0   
	lamd = spont_FR/1000 

	while (t<=tstop):#{ :this Algorithm is from http://freakonometrics.hypotheses.org/724  from  
		
		if rnd_seed:
			rnd_stream = np.random.RandomState(rnd_seed)
		else:
			rnd_stream = np.random.RandomState()
		u = rnd_stream.uniform()   

	 	# Update t with the drawn ISI  
		t = t - np.log(u)/lamd # u is uniform(0,1) so drawing (1-u) is identical to drawing u.  
		
		if t<=tstop:  
			invl=t  
			spike_times.append(invl)  
	
	FRS = []   
	for i in range(0, tstop, bins):   
		interval = [i, i+bins]  
		FRS.append(len([T for T in spike_times if T>=interval[0] and T<interval[1]])/bins)

	mean_FR = 1000*np.mean(FRS)
	if verbose:   
		print('Created Poisson spike train with FR = {} (required FR: {})'.format(1000*np.mean(FRS), spont_FR))

	return spike_times, mean_FR

def CreateThalamicSpikeTimes(PV_pop, SOM_pop, plot_FR_hists=True, SOM_start=0):

	spikes_dict = {'PV': {}, 'SOM': {}}

	for PV_cell in PV_pop.inputs:
		cell_ID = list(PV_pop.inputs.keys()).index(PV_cell)
		
		spikes_dict['PV'][PV_cell] = {}

		for axon in PV_pop.inputs[PV_cell]:
			axon_ID = list(PV_pop.inputs[PV_cell].keys()).index(axon)

			spikes_dict['PV'][PV_cell][axon] = {}

			stim_times, mean_FR = CreatePoisson(spont_FR_PV, tstop, verbose=False)
			spikes_dict['PV'][PV_cell][axon]['stim_times'], spikes_dict['PV'][PV_cell][axon]['mean_FR'] = stim_times, mean_FR

	for SOM_cell in SOM_pop.inputs:
		cell_ID = list(SOM_pop.inputs.keys()).index(SOM_cell)		

		spikes_dict['SOM'][SOM_cell] = {}

		for axon in SOM_pop.inputs[SOM_cell]:
			axon_ID = list(SOM_pop.inputs[SOM_cell].keys()).index(axon)

			spikes_dict['SOM'][SOM_cell][axon] = {}

			stim_times, mean_FR = CreatePoisson(spont_FR_SOM, tstop, verbose=False)
			spikes_dict['SOM'][SOM_cell][axon]['stim_times'], spikes_dict['SOM'][SOM_cell][axon]['mean_FR'] = [i+SOM_start for i in stim_times], mean_FR

	if plot_FR_hists:
		plotSpontaneous(spikes_dict, tstop, bins=4)

	return spikes_dict

def plotSpontaneous(spikes_dict, tstop, bins=0):

	PV_FRs = []
	for PV_cell in spikes_dict['PV']:
		for axon in spikes_dict['PV'][PV_cell]:
			PV_FRs.append(spikes_dict['PV'][PV_cell][axon]['mean_FR'])
	SOM_FRs = []
	for SOM_cell in spikes_dict['SOM']:
		for axon in spikes_dict['SOM'][SOM_cell]:
			SOM_FRs.append(spikes_dict['SOM'][SOM_cell][axon]['mean_FR'])


	plt.figure()
	plt.hist(PV_FRs, bins=bins, alpha=0.5, color='xkcd:blue', label='To PV')
	plt.hist(SOM_FRs, bins=bins, alpha=0.5, color='xkcd:orchid', label='To SOM')
	plt.legend()
	plt.title('Spontaneous Firing Rate in Thalamic Axons')
	plt.xlabel(r'FR ($\frac{spikes}{sec}$)')
	plt.ylabel('No. of Axons')

def putspikes():

	FRs = []
	# PV inputs
	if PV_pop:
		for PV_cell in PV_pop.inputs:
			for axon in PV_pop.inputs[PV_cell]:

				if input_source == 'spont':
					stim_times = spikes_dict['PV'][PV_cell][axon]['stim_times']
					FRs.append(spikes_dict['PV'][PV_cell][axon]['mean_FR'])
				elif input_source == 'stim':
					stim_times = PV_pop.inputs[PV_cell][axon]['stim_times']

				for netcon in PV_pop.inputs[PV_cell][axon]['netcons']:
					for T in stim_times:

						netcon.event(T + PV_input_delay)

	print('Mean PV FR: {}'.format(np.mean(FRs)))

	FRs = []
	# SOM inputs
	if SOM_pop:
		for SOM_cell in SOM_pop.inputs:
			for axon in SOM_pop.inputs[SOM_cell]:

				if input_source == 'spont':
					stim_times = spikes_dict['SOM'][SOM_cell][axon]['stim_times']
					FRs.append(spikes_dict['SOM'][SOM_cell][axon]['mean_FR'])
				elif input_source == 'stim':
					stim_times = SOM_pop.inputs[SOM_cell][axon]['stim_times']

				for netcon in SOM_pop.inputs[SOM_cell][axon]['netcons']:
					for T in stim_times:

						netcon.event(T + SOM_input_delay)

	print('Mean SOM FR: {}'.format(np.mean(FRs)))

def RunSim(v_init=-75, tstop=154*1000, print_=''):
	# Oren's simulation length is 154 seconds, I leave some time for last inputs to decay

	print(print_)

	h.tstop = tstop
	h.v_init = v_init

	t = h.Vector()
	t.record(h._ref_t)

	h.run()

	PV_soma = PV_pop.cells['PV0']['soma_v']
	PV_n_spikes = len([i for i in range(len(PV_soma)) if (PV_soma[i]>spike_threshold) and (PV_soma[i]>PV_soma[i-1]) and (PV_soma[i]>PV_soma[i+1])])


	return t, PV_n_spikes

input_source = 'spont' # or: 'stim'
if input_source == 'spont':
	SOM_start 		 = 5000
else:
	SOM_start = None

n_syns_SOM_to_PV = 500
tstop 			 = 10000
SOM_input_delay  = 0
SOM_output_delay = 0
dt = h.dt

print('\n========== Choosing GIDs from BlueBrain Database (pyramidal between {}Hz and {}Hz) =========='.format(freq1, freq2))
upload_from = 'GIDs_instantiations/pyr_72851_between_6666_9600' # upload_from = False
chosen_GIDs, chosen_PV_n_contacts, thalamic_GIDs = get_GIDs(upload_from)

# ========================================================= Set Circuit =======================================================================================
Pyr_pop, PV_pop, SOM_pop = CreatePopulations(n_pyr=1, n_PV=1, n_SOM=1)

set_ThalamicInputs(Pyr_pop, PV_pop, SOM_pop, thalamic_GIDs)

if input_source == 'spont':
	spikes_dict = CreateThalamicSpikeTimes(PV_pop, SOM_pop, SOM_start=SOM_start)

events = h.FInitializeHandler(putspikes)

# ========================================================= 1st simulation =======================================================================================
t, n_spikes_unconnected = RunSim(tstop = tstop, print_='Running simulation (only PV)')

PV_soma_before = copy(PV_pop.cells['PV0']['soma_v'])
PV_inputs_before = copy(PV_pop.inputs['PV0'])

# ========================================================= 2nd simulation =======================================================================================
PV_pop.connectCells('PV0', SOM_pop, 'SOM0', [PV_pop.cells['PV0']['soma']], n_syns_SOM_to_PV, 'random', record_syns=True, 
																									   input_source='voltage', 
																									   weight=SOM_to_PV_weight, 
																									   delay=SOM_output_delay, 
																									   threshold=spike_threshold) 

t, n_spikes_connected = RunSim(tstop = tstop, print_='Running simulation (SOM inputs to PV)')

PV_soma_after = PV_pop.cells['PV0']['soma_v']
PV_inputs_after = PV_pop.inputs['PV0']

# ========================================================= Plot PV Voltage Before & After =======================================================================================
soma_f, soma_ax = plt.subplots() 

soma_ax.set_title(r'PV_Somatic Voltage Before & After Connecting to SOM ($\frac{spikes_{inhibited}}{spikes_{uninhibited}}$: %.3f)'%(n_spikes_connected/n_spikes_unconnected))
soma_ax.set_xlabel('T (ms)')
soma_ax.set_ylabel('V (mV)')

soma_ax.plot(t, PV_soma_before, label='Uninhibited', color='xkcd:azure', LineWidth=2.5) 
soma_ax.plot(t, PV_soma_after, label='Connected to SOM', color='xkcd:coral', LineWidth=2.5, alpha=0.6) 
soma_ax.legend()

# ========================================================= Plot Cross Correlation =======================================================================================
def CrossCorr(SOM_pop, SOM_cell, PV_pop, PV_cell, window=100, bins=7, to_plot=''):
	'''
	cross-correlation- for each SOM spike, look when PV spiked before and after. Run on SOM trace and for each SOM spike: 
	1) go to PV and take 100ms before and after
	2) create raster where each row is one spike of SOM
	3) do PSTH of PV where t=0 is SOM spike (same bins as I used before for PSTH and FRs)).
	'''
	def get_spikesInWindow(spike_times, interval):
		times_in_window = [i for i in spike_times if (i>=interval[0]) and (i<=interval[1])]
		shifted_in_window = [i-T for i in times_in_window]

		return shifted_in_window

	def get_hists(without_I_arr, with_I_arr):
		without_I_arr = [j for i in without_I_arr for j in i]
		with_I_arr 	  = [j for i in with_I_arr for j in i]

		H = {'without_I': [], 'with_I': []}
		B = {'without_I': [], 'with_I': []}

		MIN = min(min(without_I_arr), min(with_I_arr))
		if MIN == min(without_I_arr):			

			H['without_I'], B['without_I'] = np.histogram(without_I_arr, bins=int((window*2)/bins))
			# H['without_I'] = [i/max(H['without_I']) for i in H['without_I']]
			H['without_I'] = [i/sum(H['without_I']) for i in H['without_I']]

			H['with_I'], B['with_I'] = np.histogram(with_I_arr, bins=B['without_I'])
			# H['with_I'] = [i/max(H['with_I']) for i in H['with_I']]
			H['with_I'] = [i/sum(H['with_I']) for i in H['with_I']]
		else:
			H['with_I'], B['with_I'] = np.histogram(with_I_arr, bins=int((window*2)/bins))
			# H['with_I'] = [i/max(H['with_I']) for i in H['with_I']]
			H['with_I'] = [i/sum(H['with_I']) for i in H['with_I']]

			H['without_I'], B['without_I'] = np.histogram(without_I_arr, bins=B['with_I'])
			# H['without_I'] = [i/max(H['without_I']) for i in H['without_I']]
			H['without_I'] = [i/sum(H['without_I']) for i in H['without_I']]

		return H, B

	SOM_v = SOM_pop.cells[SOM_cell]['soma_v']
	PV_v  = PV_pop.cells[PV_cell]['soma_v']

	spike_times 				= {'SOM': None, 'PV_without_I': None, 'PV_with_I': None}
	spike_times['SOM'] 			= get_spike_times(SOM_v, spike_threshold)
	spike_times['PV_without_I'] = get_spike_times(PV_soma_before, spike_threshold)	
	spike_times['PV_with_I'] 	= get_spike_times(PV_v, spike_threshold)

	PV_spikes_array, PV_vs_in_window = {'without_I': [], 'with_I': []}, {'without_I': [], 'with_I': []}
	for T in spike_times['SOM']:

		interval = [T-window, T+window]
		idx1 = int((T-window)/dt)
		idx2 = int((T+window)/dt)

		PV_in_window = [i for i in spike_times['PV_with_I'] if (i>=interval[0]) and (i<=interval[1])]
		shifted_PV_in_window = [i - T for i in PV_in_window]

		PV_spikes_array['with_I'].append(get_spikesInWindow(spike_times['PV_with_I'], interval))		
		PV_spikes_array['without_I'].append(get_spikesInWindow(spike_times['PV_without_I'], interval))

		PV_vs_in_window['with_I'].append(list(PV_v)[idx1:idx2])
		PV_vs_in_window['without_I'].append(list(PV_soma_before)[idx1:idx2])

	if spike_times['SOM'][-1]+window>tstop:
		PV_vs_in_window['with_I'] = PV_vs_in_window['with_I'][:-1]
		PV_vs_in_window['without_I'] = PV_vs_in_window['without_I'][:-1]

	if to_plot == 'n_spikes':
		H, B = get_hists(PV_spikes_array['without_I'], PV_spikes_array['with_I'])

	elif to_plot == 'voltage':
		H, B = get_hists(PV_vs_in_window['without_I'], PV_vs_in_window['with_I'])
	
	else:
		raise Exception('What to plot?')

	f, h_ax = plt.subplots(2, 1)
	h_ax[0].bar(B['without_I'][:-1], H['without_I'], align='edge', width=np.diff(B['with_I'])[0], alpha=0.5, label='No Inhibition')
	h_ax[0].bar(B['with_I'][:-1], H['with_I'], align='edge', width=np.diff(B['with_I'])[0], alpha=0.5, label='SOM connected')
	h_ax[0].set_ylabel('Count')

	if to_plot == 'n_spikes':
		h_ax[0].set_title('PSTH of PV Spike Times Around SOM Spikes')		
		h_ax[0].axvline(0, LineStyle='--', LineWidth=1, label='SOM spike time')

		h_ax[1].plot(B['with_I'][:-1], [H['without_I'][i]-H['with_I'][i] for i in range(len(H))])
		h_ax[1].set_title('Difference in Spike Count (before - after)')
		h_ax[1].set_ylabel('(>0) --> uninhibited is larger')
		h_ax[1].set_xlabel('Peri-SOM-spike time (ms)')
		
	elif to_plot == 'voltage':
		h_ax[0].set_title('Mean PV Somatic Voltage Around SOM Spikes')

		h_ax[1].set_xlabel('Voltage (mV)')
		h_ax[1].plot(np.arange(-window, window, dt), np.mean(PV_vs_in_window['without_I'], axis=0), label='No Inhibition')
		h_ax[1].plot(np.arange(-window, window, dt), np.mean(PV_vs_in_window['with_I'], axis=0), label='SOM connected')

	h_ax[1].axhline(0, LineStyle='--', color='gray', label='SOM spike')		
	h_ax[0].legend()
	h_ax[1].legend()

	return h_ax, PV_spikes_array

h_ax, PV_spikes_array = CrossCorr(SOM_pop, 'SOM0', PV_pop, 'PV0', bins=2, window=100, to_plot='voltage')

# ========================================================= Plot Firing Rate Bars =======================================================================================
def plot_splitFR(PV_v, SOM_start=None, bins=1):

	spike_times = get_spike_times(PV_v, spike_threshold)
	interval = range(0, tstop, bins)

	PV_no_I = [i for i in spike_times if i < SOM_start]
	FR_no_I = 1000*len(PV_no_I)/SOM_start
	temp, mean_FR_no_I = [], []
	for i in range(len(interval)-1):
		T1 = interval[i]
		T2 = interval[i+1]
		temp.append([i for i in PV_no_I if i>=T1 and i<T2])
		mean_FR_no_I.append(1000*len(temp[-1])/bins)
	se_no_I = np.std(mean_FR_no_I)/np.sqrt(len(mean_FR_no_I))
	mean_FR_no_I = np.mean(mean_FR_no_I)



	PV_with_I = [i for i in spike_times if i >= SOM_start]
	FR_with_I = 1000*len(PV_with_I)/(tstop-SOM_start)
	temp, mean_FR_with_I = [], []
	for i in range(len(interval)-1):
		T1 = interval[i]
		T2 = interval[i+1]
		temp.append([i for i in PV_with_I if i>=T1 and i<T2])
		mean_FR_with_I.append(1000*len(temp[-1])/bins)
	se_with_I = np.std(mean_FR_with_I)/np.sqrt(len(mean_FR_with_I))
	mean_FR_with_I = np.mean(mean_FR_with_I)


	f, ax = plt.subplots()
	# ax.bar([1, 2], [FR_no_I, FR_with_I])
	ax.bar([1, 2], [mean_FR_no_I, mean_FR_with_I], yerr=[se_no_I, se_with_I])
	ax.set_xticks([1, 2])
	ax.set_xticklabels(['No Inh.', 'SOM inh.'])
	ax.set_title('PV Firing Rate With & Without SOM Inhibition (bins = %i)'%bins)
	ax.set_ylabel('FR (Hz)')

if input_source=='spont':
	plot_splitFR(PV_soma_after, bins=10, SOM_start=SOM_start)

# ========================================================= Plot Spike Count =======================================================================================
def plot_spikeCount(PV_soma_before, PV_soma_after, bins=200):

	spikes_before = get_spike_times(PV_soma_before, spike_threshold)
	spikes_after = get_spike_times(PV_soma_after, spike_threshold)

	bins_before, bins_after = [], [] 
	intervals = range(0, tstop, bins) 
	
	for i in range(len(intervals)-1): 
		T1 = intervals[i] 
		T2 = intervals[i+1] 
		bins_before.append(len([i for i in spikes_before if i>=T1 and i<T2])) 
		bins_after.append(len([i for i in spikes_after if i>=T1 and i<T2])) 

	f, ax = plt.subplots()
	f.suptitle('Spike Count of PV Uninhibited & With SOM Inhibition (bins = %ims)'%bins)
	ax.set_xlabel('T (ms)')
	ax.set_ylabel('Spike Count')
	ax.bar(intervals[:-1], bins_before, color='xkcd:azure', label='before', width=bins)
	ax.bar(intervals[:-1], bins_after, color='xkcd:magenta', label='after', alpha=0.5, width=bins)

	if SOM_start:
		ax.axvline(SOM_start, LineStyle='--', color='gray', LineWidth=1.5, label='SOM starts spiking')
	ax.legend()  

plot_spikeCount(PV_soma_before, PV_soma_after)

# ========================================================= Plot SOM & PV Voltages =======================================================================================
comp_f, comp_ax = plt.subplots()
comp_ax.set_title('Somatic Voltages of SOM and the PV it inhibited')
comp_ax.set_xlabel('T (ms)')
comp_ax.set_ylabel('V (mV)')
comp_ax.plot(t, SOM_pop.cells['SOM0']['soma_v'], label='SOM')
comp_ax.plot(t, PV_pop.cells['PV0']['soma_v'], label='PV') 
comp_ax.legend()

plt.figure();plt.plot(t, SOM_pop.cells['SOM0']['soma_v'])
plt.title('SOM Somatic Voltage')
plt.xlabel('T (ms)')
plt.ylabel('V (mV)')
# ========================================================= I Conductances =======================================================================================

g_GABA = np.sum(PV_pop.cell_inputs['PV0']['SOM0']['g_GABA'], axis=0)
g_GABA = [i*1000 for i in g_GABA] # convert to nS
g_f, g_ax = plt.subplots()
g_ax.plot(t, g_GABA)
g_ax.set_title('GABA Conductance From SOM to Pyr')
g_ax.set_ylabel('g_GABA (nS)')
g_ax.set_xlabel('T (ms)')

# ========================================================= E Conductances Before & After =======================================================================================

# g_f, g_ax = plt.subplots()

# g_ax.set_title('Total Excitatory Conductances in PV Cell Before & After Connecting to SOM')
# g_ax.set_xlabel('T (ms)')
# g_ax.set_ylabel(r'G ($\mu$V)')

# g_ax.plot(t, np.sum([np.sum(PV_inputs_before[a]['g_AMPA'], axis=0) for a in PV_inputs_before], axis=0), color='lightblue', label='AMPA (before)')
# g_ax.plot(t, np.sum([np.sum(PV_inputs_before[a]['g_NMDA'], axis=0) for a in PV_inputs_before], axis=0), color='xkcd:aqua', label='NMDA (before)')
# g_ax.plot(t, np.sum([np.sum(PV_inputs_after[a]['g_AMPA'], axis=0) for a in PV_inputs_after], axis=0), color='coral', alpha=0.5, label='AMPA (after)')
# g_ax.plot(t, np.sum([np.sum(PV_inputs_after[a]['g_NMDA'], axis=0) for a in PV_inputs_after], axis=0), color='orchid', alpha=0.5, label='NMDA (after)')
# g_ax.legend()





























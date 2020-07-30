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
# ============================================  Define Functions  ============================================
'''
Plan:
	- V - now: 5000ms, with PV connections, 9600Hz => 6 spikes out of 9
	- V - 5000ms without PV connections, 6666Hz (maybe also 9600Hz) => 7 out of 9 [same synapses+PV: 5] (6666)
	- V - 10000ms without+with PV connections (for same synapses), 6666Hz (maybe also 9600Hz) => 13 (with PV) & 20 (without PV) out of 25
'''
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
		where_SOM_syns = 'synapse_locs_instantiations/thalamic_syn_locs_SOM_basal_high_input.p'
		
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

def CreateThalamicSpikeTimes(post_pop, pre_pop, plot_FR_hists=True, SOM_start=0):

	spikes_dict = {types_dict['post']: {}, types_dict['pre']: {}}

	for post_cell in post_pop.inputs:
		cell_ID = list(post_pop.inputs.keys()).index(post_cell)
		
		spikes_dict[types_dict['post']][post_cell] = {}

		for axon in post_pop.inputs[post_cell]:
			axon_ID = list(post_pop.inputs[post_cell].keys()).index(axon)

			spikes_dict[types_dict['post']][post_cell][axon] = {}

			stim_times, mean_FR = CreatePoisson(spont_FR[types_dict['post']], tstop, verbose=False)
			spikes_dict[types_dict['post']][post_cell][axon]['stim_times'], spikes_dict[types_dict['post']][post_cell][axon]['mean_FR'] = stim_times, mean_FR

	spikes_dict[types_dict['pre']] = {}

	for pre_cell in pre_pop.inputs:
		cell_ID = list(pre_pop.inputs.keys()).index(pre_cell)		

		spikes_dict[types_dict['pre']][pre_cell] = {}

		for axon in pre_pop.inputs[pre_cell]:
			axon_ID = list(pre_pop.inputs[pre_cell].keys()).index(axon)

			spikes_dict[types_dict['pre']][pre_cell][axon] = {}

			stim_times, mean_FR = CreatePoisson(spont_FR[types_dict['pre']], tstop, verbose=False)
			spikes_dict[types_dict['pre']][pre_cell][axon]['stim_times'], spikes_dict[types_dict['pre']][pre_cell][axon]['mean_FR'] = [i+SOM_start for i in stim_times], mean_FR

	if plot_FR_hists:
		plotSpontaneous(spikes_dict, tstop, bins=4)

	return spikes_dict

def plotSpontaneous(spikes_dict, tstop, bins=0):

	PV_FRs = []
	for PV_cell in spikes_dict[types_dict['post']]:
		for axon in spikes_dict[types_dict['post']][PV_cell]:
			PV_FRs.append(spikes_dict[types_dict['post']][PV_cell][axon]['mean_FR'])
	SOM_FRs = []
	for SOM_cell in spikes_dict[types_dict['pre']]:
		for axon in spikes_dict[types_dict['pre']][SOM_cell]:
			SOM_FRs.append(spikes_dict[types_dict['pre']][SOM_cell][axon]['mean_FR'])


	plt.figure()
	plt.hist(PV_FRs, bins=bins, alpha=0.5, color='xkcd:blue', label='To {}'.format(types_dict['post']))
	plt.hist(SOM_FRs, bins=bins, alpha=0.5, color='xkcd:orchid', label='To {}'.format(types_dict['pre']))
	plt.legend()
	plt.title('Spontaneous Firing Rate in Thalamic Axons')
	plt.xlabel(r'FR ($\frac{spikes}{sec}$)')
	plt.ylabel('No. of Axons')

def putspikes():

	FRs = []
	for cell in post_pop.inputs:
		for axon in post_pop.inputs[cell]:

			if input_source == 'spont':
				stim_times = spikes_dict[types_dict['post']][cell][axon]['stim_times']
				FRs.append(spikes_dict[types_dict['post']][cell][axon]['mean_FR'])
			elif input_source == 'stim':
				stim_times = post_pop.inputs[cell][axon]['stim_times']

			for netcon in post_pop.inputs[cell][axon]['netcons']:
				for T in stim_times:
					netcon.event(T + input_delay_dict[types_dict['post']])
	print('Mean {} FR: {}'.format(types_dict['post'], np.mean(FRs)))

	FRs = []
	for cell in pre_pop.inputs:
		for axon in pre_pop.inputs[cell]:

			if input_source == 'spont':
				stim_times = spikes_dict[types_dict['pre']][cell][axon]['stim_times']
				FRs.append(spikes_dict[types_dict['pre']][cell][axon]['mean_FR'])
			elif input_source == 'stim':
				stim_times = pre_pop.inputs[cell][axon]['stim_times']

			for netcon in pre_pop.inputs[cell][axon]['netcons']:
				for T in stim_times:
					netcon.event(T + input_delay_dict[types_dict['pre']])
	print('Mean {} FR: {}'.format(types_dict['pre'], np.mean(FRs)))

def RunSim(v_init=-75, tstop=154*1000, print_=''):
	# Oren's simulation length is 154 seconds, I leave some time for last inputs to decay

	print(print_)

	h.tstop = tstop
	h.v_init = v_init

	t = h.Vector()
	t.record(h._ref_t)

	h.run()

	post_soma = post_pop.cells[post_cell]['soma_v']
	post_n_spikes = len([i for i in range(len(post_soma)) if (post_soma[i]>spike_threshold) and (post_soma[i]>post_soma[i-1]) and (post_soma[i]>post_soma[i+1])])

	return t, post_n_spikes

record_thalamic_syns = True

# ============================================  Define Constants  ============================================

n_syns_SOM_to_PV = 200
tstop 			 = 20000
SOM_input_delay  = 0
SOM_output_delay = 0
dt = h.dt
SOM_to_Pyr_weight = 0.3
PV_to_Pyr_weight = 0.3
SOM_to_PV_weight = 0.3

input_source = 'spont' # or: 'spont' or stim'
types_dict = {'pre': 'SOM', 
			  'post': 'PV'}

spont_FR = {'PV': 1.5, 'SOM': 1.5, 'Pyr': 3}


get_spike_times = lambda soma_v, thresh: [t[idx] for idx in [i for i in range(len(soma_v)) if (soma_v[i]>thresh) and (soma_v[i]>soma_v[i-1]) and (soma_v[i]>soma_v[i+1])]]

if input_source == 'spont':
	SOM_start 		 = 5000
else:
	SOM_start = None

# ========================================================= Set Circuit =======================================================================================
print('\n========== Choosing GIDs from BlueBrain Database (pyramidal between {}Hz and {}Hz) =========='.format(freq1, freq2))
upload_from = 'GIDs_instantiations/pyr_72851_between_6666_9600' # upload_from = False
chosen_GIDs, chosen_PV_n_contacts, thalamic_GIDs = get_GIDs(upload_from)

input_delay_dict = {'PV': PV_input_delay, 'SOM': SOM_input_delay, 'Pyr': Pyr_input_delay}

Pyr_pop, PV_pop, SOM_pop = CreatePopulations(n_pyr=1*('Pyr' in types_dict.values()), n_PV=1*('PV' in types_dict.values()), n_SOM=1*('SOM' in types_dict.values()))

set_ThalamicInputs(Pyr_pop, PV_pop, SOM_pop, thalamic_GIDs)

if types_dict['pre'] == 'SOM':
	if types_dict['post'] == 'PV':
		n_syns = n_syns_SOM_to_PV
	elif types_dict['post'] == 'Pyr':
		n_syns = n_syns_SOM_to_Pyr
	else:
		raise Exception('post cell undefined')
elif types_dict['pre'] == 'PV':
	if types_dict['post'] == 'Pyr':
		n_syns = chosen_PV_n_contacts[PV_pop.name_to_gid['PV0']]
		# n_syns=500
	else:
		raise Exception('post cell undefined')
else:
	raise Exception('pre cell undefined')

# ========================================================= Set Specific Elements =======================================================================================
pre_pop = [pop for pop in [P for P in [Pyr_pop, PV_pop, SOM_pop] if P] if pop.population_name==types_dict['pre']][0]
pre_cell = types_dict['pre'] + '0'

post_pop = [pop for pop in [P for P in [Pyr_pop, PV_pop, SOM_pop] if P] if pop.population_name==types_dict['post']][0]
post_cell = types_dict['post'] + '0'

if input_source == 'spont':
	spikes_dict = CreateThalamicSpikeTimes(post_pop, pre_pop, SOM_start=SOM_start)

events = h.FInitializeHandler(putspikes)

# ========================================================= 1st simulation =======================================================================================
t, n_spikes_unconnected = RunSim(tstop = tstop, print_='Running simulation (only {})'.format(types_dict['post']))

post_v_without_I = copy(post_pop.cells[post_cell]['soma_v'])
post_inputs_without_I = copy(post_pop.inputs[post_cell])

# ========================================================= 2nd simulation =======================================================================================
post_pop.connectCells(post_cell, pre_pop, pre_cell, [post_pop.cells[post_cell]['soma']], ['random', n_syns], record_syns=True, 
																									   input_source='voltage', 
																									   weight=SOM_to_PV_weight, 
																									   delay=SOM_output_delay, 
																									   threshold=spike_threshold) 

t, n_spikes_connected = RunSim(tstop = tstop, print_='Running simulation ({} inputs to {})'.format(types_dict['pre'], types_dict['post']))

post_v_with_I = post_pop.cells[post_cell]['soma_v']
post_inputs_with_I = post_pop.inputs[post_cell]

# ========================================================= Plot PV Voltage Before & After =======================================================================================
soma_f, soma_ax = plt.subplots() 

soma_ax.set_title('{} Somatic Voltage Before & After Connecting to {}'.format(types_dict['post'], types_dict['pre']) + r' ($\frac{spikes_{inhibited}}{spikes_{uninhibited}}$: %.3f)'%(n_spikes_connected/n_spikes_unconnected))
soma_ax.set_xlabel('T (ms)')
soma_ax.set_ylabel('V (mV)')

soma_ax.plot(t, post_v_without_I, label='Uninhibited', color='xkcd:azure', LineWidth=2.5) 
soma_ax.plot(t, post_v_with_I, label='Connected to SOM', color='xkcd:coral', LineWidth=2.5, alpha=0.6) 
soma_ax.legend()

# ========================================================= Plot Cross Correlation =======================================================================================
def CrossCorr(pre_pop, pre_cell, post_pop, post_cell, window=100, bins=7, to_plot=''):
	'''
	cross-correlation- for each presynaptic spike, look when PV spiked before and after. Run on presynaptic trace and for each presynaptic spike: 
	1) go to PV and take 100ms before and after
	2) create raster where each row is one spike of SOM
	3) do PSTH of PV where t=0 is presynaptic spike (same bins as I used before for PSTH and FRs)).
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

		if len(without_I_arr)>0 and len(with_I_arr)>0:
			MIN = min(min(without_I_arr), min(with_I_arr))
			take_bins_from = 'with'*(MIN==min(with_I_arr)) + 'whithout'*(MIN==min(without_I_arr))
		else:
			take_bins_from = ''

		if take_bins_from == 'without':			

			H['without_I'], B['without_I'] = np.histogram(without_I_arr, bins=int((window*2)/bins))
			H['with_I'], B['with_I'] = np.histogram(with_I_arr, bins=B['without_I'])

		else:
			H['with_I'], B['with_I'] = np.histogram(with_I_arr, bins=int((window*2)/bins))			
			H['without_I'], B['without_I'] = np.histogram(without_I_arr, bins=B['with_I'])
		
		H['with_I']    = [i/sum(H['with_I']) for i in H['with_I']]
		H['without_I'] = [i/sum(H['without_I']) for i in H['without_I']]

		return H, B

	def plot_CrossCorr(B, H):
		f, h_ax = plt.subplots(2, 1)
		h_ax[0].bar(B['without_I'][:-1], H['without_I'], align='edge', width=np.diff(B['with_I'])[0], alpha=0.5, label='No Inhibition')
		h_ax[0].bar(B['with_I'][:-1], H['with_I'], align='edge', width=np.diff(B['with_I'])[0], alpha=0.5, label='{} connected'.format(types_dict['pre']))
		h_ax[0].set_ylabel('Frequency')

		if to_plot == 'n_spikes':
			h_ax[0].set_title('PSTH of {} Spike Times Around {} Spikes'.format(types_dict['post'], types_dict['pre']))		
			h_ax[0].axvline(0, color='gray', LineStyle='--', LineWidth=1, label='{} spike time'.format(types_dict['pre']))

			if np.isnan(sum(H['with_I'])):
				H['with_I'] = [0]*len(H['with_I'])
			h_ax[1].plot(B['with_I'][:-1], [H['without_I'][i]-H['with_I'][i] for i in range(len(B['with_I'][:-1]))])
			h_ax[1].set_title('Difference in Spike Count (before - after)')
			h_ax[1].set_ylabel('(>0) --> uninhibited is larger')
			h_ax[1].set_xlabel('Peri-{}-spike time (ms)'.format(types_dict['pre']))
			h_ax[1].axhline(0, LineStyle='--', color='gray', label='{} spike'.format(types_dict['pre']))
			
		elif to_plot == 'voltage':
			h_ax[0].set_title('Mean {} Somatic Voltage Around {} Spikes'.format(types_dict['post'], types_dict['pre']))
			h_ax[0].set_xlabel('Somatic Voltage (ms)')

			h_ax[1].set_title('{} Somatic Voltage Around {} Spike'.format(types_dict['post'], types_dict['pre']))
			h_ax[1].set_ylabel('Somatic Voltage (mV)')
			h_ax[1].set_xlabel('T (ms)')
			h_ax[1].plot(np.arange(-window, window, dt), np.mean(post_vs_in_window['without_I'], axis=0), label='No Inhibition')
			h_ax[1].plot(np.arange(-window, window, dt), np.mean(post_vs_in_window['with_I'], axis=0), label='{} connected'.format(types_dict['pre']))
			h_ax[1].axvline(0, color='gray', LineStyle='--', LineWidth=1, label='{} spike time'.format(types_dict['pre']))

				
		h_ax[0].legend()
		h_ax[1].legend()

		return h_ax

	pre_v = pre_pop.cells[pre_cell]['soma_v']

	spike_times 				= {'pre': None, 'post_without_I': None, 'post_with_I': None}
	spike_times['pre'] 			= get_spike_times(pre_v, spike_threshold)
	spike_times['post_without_I'] = get_spike_times(post_v_without_I, spike_threshold)	
	spike_times['post_with_I'] 	= get_spike_times(post_v_with_I, spike_threshold)

	post_spikes_array, post_vs_in_window = {'without_I': [], 'with_I': []}, {'without_I': [], 'with_I': []}
	for T in spike_times['pre']:

		interval = [T-window, T+window]
		idx1 = int((T-window)/dt)
		idx2 = int((T+window)/dt)

		if idx1>=0 and idx2*dt<=tstop:

			temp_with = get_spikesInWindow(spike_times['post_with_I'], interval)
			if len(temp_with) > 0:
				post_spikes_array['with_I'].append(temp_with)		

			temp_without = get_spikesInWindow(spike_times['post_without_I'], interval)
			if len(temp_without) > 0:
				post_spikes_array['without_I'].append(temp_without)

			post_vs_in_window['with_I'].append(list(post_v_with_I)[idx1:idx2])
			post_vs_in_window['without_I'].append(list(post_v_without_I)[idx1:idx2])



	# trim_start = len([i for i in spike_times['pre'] if i-100<0])
	# trim_end = len([i for i in spike_times['pre'] if i+100>tstop])
	# if trim_:
	# 	post_vs_in_window['with_I'] = post_vs_in_window['with_I'][:-trim_]
	# 	post_vs_in_window['without_I'] = post_vs_in_window['without_I'][:-trim_]
	
	if to_plot == 'n_spikes':
		H, B = get_hists(post_spikes_array['without_I'], post_spikes_array['with_I'])

	elif to_plot == 'voltage':
		
		H, B = get_hists(post_vs_in_window['without_I'], post_vs_in_window['with_I'])
	
	else:
		raise Exception('What to plot?')

	h_ax = plot_CrossCorr(B, H)

	return h_ax

h_ax = CrossCorr(pre_pop, pre_cell, post_pop, post_cell, bins=10, window=100, to_plot='n_spikes')
h_ax = CrossCorr(pre_pop, pre_cell, post_pop, post_cell, bins=2, window=100, to_plot='voltage')  # bins => bin length
# plt.suptitle('n_syns =')
# ========================================================= Plot Firing Rate Bars =======================================================================================
def plot_splitFR(post_v, SOM_start=None, bins=1):

	spike_times = get_spike_times(post_v, spike_threshold)
	interval = range(0, tstop, bins)

	spikes_no_I = [i for i in spike_times if i < SOM_start]
	FR_no_I = 1000*len(spikes_no_I)/SOM_start
	temp, mean_FR_no_I = [], []
	for i in range(len(interval)-1):
		T1 = interval[i]
		T2 = interval[i+1]
		temp.append([i for i in spikes_no_I if i>=T1 and i<T2])
		mean_FR_no_I.append(1000*len(temp[-1])/bins)
	se_no_I = np.std(mean_FR_no_I)/np.sqrt(len(mean_FR_no_I))
	mean_FR_no_I = np.mean(mean_FR_no_I)



	spikes_with_I = [i for i in spike_times if i >= SOM_start and i<SOM_start*2]
	FR_with_I = 1000*len(spikes_with_I)/(2*SOM_start)
	temp, mean_FR_with_I = [], []
	for i in range(len(interval)-1):
		T1 = interval[i]
		T2 = interval[i+1]
		temp.append([i for i in spikes_with_I if i>=T1 and i<T2])
		mean_FR_with_I.append(1000*len(temp[-1])/bins)
	se_with_I = np.std(mean_FR_with_I)/np.sqrt(len(mean_FR_with_I))
	mean_FR_with_I = np.mean(mean_FR_with_I)


	f, ax = plt.subplots()
	# ax.bar([1, 2], [FR_no_I, FR_with_I])
	ax.bar([1, 2], [mean_FR_no_I, mean_FR_with_I], yerr=[se_no_I, se_with_I])
	ax.set_xticks([1, 2])
	ax.set_xticklabels(['No Inh.', '{} inh.'.format(types_dict['pre'])])
	ax.set_title('{} Firing Rate With & Without {} Inhibition (bins = {})'.format(types_dict['post'], types_dict['pre'], bins))
	ax.set_ylabel('FR (Hz)')

if input_source=='spont':
	plot_splitFR(post_v_with_I, bins=10, SOM_start=SOM_start)

# ========================================================= Plot Spike Count =======================================================================================
def plot_spikeCount(post_v_without_I, post_v_with_I, bins=200):

	spikes_before = get_spike_times(post_v_without_I, spike_threshold)
	spikes_after = get_spike_times(post_v_with_I, spike_threshold)

	bins_before, bins_after = [], [] 
	intervals = range(0, tstop, bins) 
	
	for i in range(len(intervals)-1): 
		T1 = intervals[i] 
		T2 = intervals[i+1] 
		bins_before.append(len([i for i in spikes_before if i>=T1 and i<T2])) 
		bins_after.append(len([i for i in spikes_after if i>=T1 and i<T2])) 

	f, ax = plt.subplots()
	f.suptitle('Spike Count of {} Uninhibited & With {} Inhibition (bins = {}ms)'.format(types_dict['post'], types_dict['pre'], bins))
	ax.set_title('n_syns = {}, stim type: {}'.format(n_syns, input_source))
	ax.set_xlabel('T (ms)')
	ax.set_ylabel('Spike Count')
	ax.bar(intervals[:-1], bins_before, color='xkcd:azure', label='before', width=bins)
	ax.bar(intervals[:-1], bins_after, color='xkcd:magenta', label='after', alpha=0.5, width=bins)

	if SOM_start:
		ax.axvline(SOM_start, LineStyle='--', color='gray', LineWidth=1.5, label='{} starts spiking'.format(types_dict['pre']))
	ax.legend()  

plot_spikeCount(post_v_without_I, post_v_with_I)

# ========================================================= Plot SOM & PV Voltages =======================================================================================
comp_f, comp_ax = plt.subplots()
comp_f.suptitle('n_syns = {}, stim type: {}'.format(n_syns, input_source))
comp_ax.set_title('Somatic Voltages of {} and the {} it inhibited'.format(types_dict['pre'], types_dict['post']))
comp_ax.set_xlabel('T (ms)')
comp_ax.set_ylabel('V (mV)')
comp_ax.plot(t, pre_pop.cells[pre_cell]['soma_v'], label=types_dict['pre'])
comp_ax.plot(t, post_pop.cells[post_cell]['soma_v'], label=types_dict['post']) 
comp_ax.legend()

plt.figure();plt.plot(t, pre_pop.cells[pre_cell]['soma_v'])
plt.title('{} Somatic Voltage'.format(types_dict['pre']))
plt.xlabel('T (ms)')
plt.ylabel('V (mV)')
# ========================================================= I Conductances =======================================================================================

# g_GABA = np.sum(post_pop.cell_inputs[post_cell][pre_cell]['g_GABA'], axis=0)
# g_GABA = [i*1000 for i in g_GABA] # convert to nS
# g_f, g_ax = plt.subplots()
# g_ax.plot(t, g_GABA)
# g_ax.set_title('GABA Conductance From {} to {}'.format(types_dict['pre'], types_dict['post']))
# g_ax.set_ylabel('g_GABA (nS)')
# g_ax.set_xlabel('T (ms)')

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





























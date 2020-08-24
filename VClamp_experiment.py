import pandas as pd
import numpy as np
import _pickle as cPickle
import matplotlib.pyplot as plt
plt.ion()

import pdb, os, sys

from tqdm import tqdm
from neuron import gui, h
from scipy.stats import ttest_ind

from Population import Population
from Stimulus import Stimulus
from Connectivity import Connectivity
from Parameter_Initialization import * # Initialize parameters before anything else!

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

def remove_mechs(which_pop, mech_list, which_secs='all'):
	'''
	which_secs is either the string "all" or a list which elements may be: soma, dend, apic, axon.
	'''

	mt = h.MechanismType(0)
	for cell in which_pop.cells:
		for sec in which_pop.cells[cell]['cell'].all:
			if which_secs == 'all' or any([j in sec.name() for j in which_secs]):
				for mech in mech_list:
					mt.select(mech)
					mt.remove(sec=sec)

def putspikes():
	
	# Pyramidal inputs
	if Pyr_pop:
		if thalamic_to_Pyr:
			for axon in Pyr_pop.inputs['Pyr0']:

				stim_times = Pyr_pop.inputs['Pyr0'][axon]['stim_times']
				for netcon in Pyr_pop.inputs['Pyr0'][axon]['netcons']:
					for T in stim_times:
						if T>dur1:
							netcon.event(T + Pyr_input_delay + clamp_stabilization_time)
						else:
							netcon.event(T + Pyr_input_delay)
		else:
			for a in Pyr_pop.inputs['Pyr0']: 
				if 'synapses' in Pyr_pop.inputs['Pyr0'][a].keys():
					del Pyr_pop.inputs['Pyr0'][a]['synapses']

	# PV inputs
	if PV_pop:
		if thalamic_to_PV:
			for PV_cell in PV_pop.inputs:
				for axon in PV_pop.inputs[PV_cell]:
					stim_times = PV_pop.inputs[PV_cell][axon]['stim_times']
					for netcon in PV_pop.inputs[PV_cell][axon]['netcons']:
						for T in stim_times:
							if T>dur1:
								netcon.event(T + PV_input_delay + clamp_stabilization_time)
							else:
								netcon.event(T + PV_input_delay)
		else:
			for a in PV_pop.inputs['PV0']: 
				if 'synapses' in PV_pop.inputs['PV0'][a].keys():
					del PV_pop.inputs['PV0'][a]['synapses']

	# SOM inputs
	if SOM_pop:
		if thalamic_to_SOM:
			for SOM_cell in SOM_pop.inputs:
				for axon in SOM_pop.inputs[SOM_cell]:
					stim_times = SOM_pop.inputs[SOM_cell][axon]['stim_times']
					for netcon in SOM_pop.inputs[SOM_cell][axon]['netcons']:
						for T in stim_times:
							if T>dur1:
								netcon.event(T + SOM_input_delay + clamp_stabilization_time)
							else:
								netcon.event(T + SOM_input_delay)
		else:
			for a in SOM_pop.inputs['SOM0']: 
				if 'synapses' in SOM_pop.inputs['SOM0'][a].keys():
					del SOM_pop.inputs['SOM0'][a]['synapses']

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

# ============================================  Define Constants  ============================================
# Pyr_input_weight = 0.2
# PV_to_Pyr_weight = 0.2
# SOM_to_Pyr_weight = 0.2


# n_syns_SOM_to_PV = 200
# SOM_to_PV_weight = 0.4
SOM_to_Pyr_weight = 0.3
PV_to_Pyr_weight = 0.5
SOM_to_PV_weight = 0.3

thalamic_to_Pyr = True
thalamic_to_PV = True
thalamic_to_SOM = True

connect_PV_to_Pyr = True
connect_SOM_to_PV = False
connect_SOM_to_Pyr = True

clamp_stabilization_time = 0

PV_output_delay = 0
SOM_output_delay = 0

PV_input_delay = 0 + clamp_stabilization_time
SOM_input_delay = 0 + clamp_stabilization_time
Pyr_input_delay = 0 + clamp_stabilization_time

upload_from = 'GIDs_instantiations/pyr_72851_between_6666_9600' # upload_from = False
chosen_GIDs, chosen_PV_n_contacts, thalamic_GIDs = get_GIDs(upload_from)


Pyr_pop, PV_pop, SOM_pop = CreatePopulations(n_pyr=1, n_PV=1, n_SOM=1)
set_ThalamicInputs(Pyr_pop, PV_pop, SOM_pop, thalamic_GIDs)

clamped_pop = Pyr_pop

# ============================================  Set Circuit  ============================================
if connect_PV_to_Pyr:
	print('Connecting PV population to Pyramidal cell (connection weight: {}uS)'.format(PV_to_Pyr_weight))	
	
	for Pyr_cell in Pyr_pop.cells:
		PV_to_Pyr_post_secs = [Pyr_pop.cells[Pyr_cell]['soma']]

		for PV_cell in PV_pop.cells:
			temp_PV_gid = PV_pop.name_to_gid[PV_cell]
			temp_n_syns = chosen_PV_n_contacts[temp_PV_gid]
			
			PV_to_Pyr_syn_locs_filename = 'synapse_locs_instantiations/PV_to_Pyr_syn_locs_soma.p'
			PV_to_Pyr_syn_dist 			= ['random', temp_n_syns]
			PV_to_Pyr_syn_specs 		= PV_to_Pyr_syn_locs_filename

			Pyr_pop.connectCells(Pyr_cell, PV_pop, PV_cell, PV_to_Pyr_post_secs, PV_to_Pyr_syn_specs, record_syns=record_PV_syns, input_source=PV_to_Pyr_source, weight=PV_to_Pyr_weight, delay=PV_output_delay, threshold=spike_threshold) # Adds self.connections to Population				

	print('\n***WARNING: Assuming isopotential soma and perisomatic PV connections: all PV synapses are placed on soma(0.5)')

# SOM ==> PV
if connect_SOM_to_PV:
	print('Connecting SOM population to PV Population (connection weight: {}uS)'.format(SOM_to_PV_weight))

	for PV_cell in PV_pop.cells:
		SOM_to_PV_post_secs = [PV_pop.cells[PV_cell]['soma']] + PV_pop.cells[PV_cell]['basal_dendrites']

		for SOM_cell in SOM_pop.cells:
			SOM_to_PV_syn_locs_filename = 'synapse_locs_instantiations/SOM_to_PV_syn_locs_soma_basal.p'
			SOM_to_PV_syn_dist 			= ['random', n_syns_SOM_to_PV]
			SOM_to_PV_syn_specs 		= SOM_to_PV_syn_locs_filename

			PV_pop.connectCells(PV_cell, SOM_pop, SOM_cell, SOM_to_PV_post_secs, SOM_to_PV_syn_specs, record_syns=record_SOM_syns, input_source='voltage', weight=SOM_to_PV_weight, delay=SOM_output_delay, threshold=spike_threshold)

# SOM ==> Pyr
if connect_SOM_to_Pyr:
	print('Connecting SOM population to Pyramidal Population (connection weight: {}uS)'.format(SOM_to_Pyr_weight))
	
	for Pyr_cell in Pyr_pop.cells:
		SOM_to_Pyr_post_secs = Pyr_pop.cells[Pyr_cell]['apical_dendrites'] # + Pyr_pop.cells[Pyr_cell]['basal_dendrites']

		for SOM_cell in SOM_pop.cells:
			SOM_to_Pyr_syn_locs_filename = 'synapse_locs_instantiations/SOM_to_Pyr_syn_locs_apical.p'
			SOM_to_Pyr_syn_dist	 		= ['random', n_syns_SOM_to_Pyr]
			SOM_to_Pyr_syn_specs 		= SOM_to_Pyr_syn_dist

			Pyr_pop.connectCells(Pyr_cell, SOM_pop, SOM_cell, SOM_to_Pyr_post_secs, SOM_to_Pyr_syn_specs, record_syns=record_SOM_syns, input_source='voltage', weight=SOM_to_Pyr_weight, delay=SOM_output_delay, threshold=spike_threshold)

# Clamp soma
clamped_cell = list(clamped_pop.cells.keys())[0]

clamped_sec = clamped_pop.cells[clamped_cell]['soma'](0.5)
VClamp = h.VClamp(clamped_sec)

# Clamp axon
clamped_sec2 = clamped_pop.cells[clamped_cell]['axons'][0](0.5)
VClamp2 = h.VClamp(clamped_sec2)

t = h.Vector()
t.record(h._ref_t)
i_clamp = h.Vector()
i_clamp.record(VClamp._ref_i)
v = h.Vector()
v = v.record(clamped_sec2._ref_v)

events = h.FInitializeHandler(putspikes)

# remove_mechs(clamped_pop, ['Ca_LVAst','Ca_HVA','Ih','Im','K_Pst','K_Tst','NaTa_t','NaTs2_t','Nap_Et2','SK_E2','SKv3_1'])

# exp = []
# for i in Pyr_pop.cells['Pyr0']['basal_dendrites']: 
# 	exp.append(h.Vector().record(i(0)._ref_v)) 
# ============================================  Perform Experiment  ============================================
exp_type = 'one_step'
if exp_type == 'two_steps':

	h.tstop = 2000
	# Start by measuring IPSCs
	amp1 = -80
	dur1 = 1000
	VClamp.amp[0] = amp1
	VClamp.dur[0] = dur1
	VClamp2.amp[0] = amp1
	VClamp2.dur[0] = dur1 

	# Measure EPSCs
	amp2 = 0
	dur2 = 1000
	VClamp.amp[1] = amp2
	VClamp.dur[1] = dur2
	VClamp2.amp[1] = amp2
	VClamp2.dur[1] = dur2

	h.run()

	plt.figure()
	plt.plot(list(t)[1:], list(i_clamp)[1:])
	plt.title('Synaptic Currents Recorded at {} Soma While Voltage-Clamped'.format(clamped_pop.population_name))
	plt.xlabel('T (ms')
	plt.ylabel('I (nA)')
	plt.suptitle('{}, {}'.format(VClamp.amp[0],VClamp.amp[1]))

	f, ax = plt.subplots(3, 1)
	f.subplots_adjust(hspace=0.4, top=0.9)
	f.suptitle('{} V-Clamped to {} and {})'.format(clamped_pop.population_name, VClamp.amp[0], VClamp.amp[1]), size=13)
	ax[0].set_title('Current Through Voltage Clamp')
	ax[0].set_ylabel('I (nA)')
	ax[0].plot(list(t)[1:], list(i_clamp)[1:])
	
	idx_start = [j for j in range(len(i_clamp)-1) if (i_clamp[j+1]-i_clamp[j])<0.0001][0] 
	idx_mid = int(dur1 / h.dt)
	ax[1].set_title('EPSCs (clamped to {})'.format(VClamp.amp[0]))
	ax[1].set_ylabel('I (nA)')
	ax[1].plot(list(t)[idx_start:idx_mid], list(i_clamp)[idx_start:idx_mid])

	idx_start2 = [j for j in range(len(i_clamp)-1) if (j>idx_mid) and (abs(i_clamp[j+1]-i_clamp[j])<0.0001)][0]
	ax[2].set_title('IPSCs (clamped to {})'.format(VClamp.amp[1]))
	ax[2].set_ylabel('I (nA)')
	ax[2].set_xlabel('T (ms)')
	ax[2].plot(list(t)[idx_start2:], list(i_clamp)[idx_start2:])

elif exp_type == 'one_step':

	h.tstop = 1000 + clamp_stabilization_time
	
	dur1 = h.tstop
	amp1 = 0
	VClamp.amp[0] = amp1
	VClamp.dur[0] = dur1
	VClamp2.amp[0] = amp1
	VClamp2.dur[0] = dur1

	h.run()

	# if connect_SOM_to_Pyr or connect_PV_to_Pyr:
	f, ax = plt.subplots()
	if PV_pop and PV_pop != clamped_pop:
		ax.plot(t, PV_pop.cells['PV0']['soma_v'], label='PV')

	if SOM_pop and SOM_pop != clamped_pop:
		ax.plot(t, SOM_pop.cells['SOM0']['soma_v'], label='SOM')

	ax2 = ax.twinx()		
	ax2.plot(t, i_clamp, 'k', label='Current')
	ax2.set_ylim([-0.9, 0.2])
	ax.set_xlabel('T (ms)')                                                 
	ax.set_ylabel('Presynaptic Soma Voltages')                           
	ax2.set_ylabel('Current through VClamp (nA)')  
	plt.suptitle('Postsynaptic Soma ({} cell) Clamped to {}'.format(clamped_cell, VClamp.amp[0]))   
	plt.title('SOM_to_Pyr_weight = {}, PV_to_Pyr_weight = {}'.format(SOM_to_Pyr_weight, PV_to_Pyr_weight))     

	ax.legend(loc='upper left')
	ax2.legend()  
	
	if thalamic_to_Pyr:
		baseline = min([abs(j) for j in list(i_clamp)[int(200/h.dt):]])
		f, ax = plt.subplots()
		ax.plot(t, Pyr_pop.cells['Pyr0']['soma_v'], label='Voltage')
		ax2 = ax.twinx()
		ax2.plot(t, i_clamp, 'k', label='Current')
		ax2.set_ylim([baseline-0.8, baseline+0.2])
		ax.set_xlabel('T (ms)')                                                 
		ax.set_ylabel('Postsynaptic Soma Voltage')                           
		ax2.set_ylabel('Current through VClamp (nA)')  
		plt.suptitle('Postsynaptic Soma ({} cell) Clamped to {}'.format(clamped_cell, VClamp.amp[0]))        

		ax.legend(loc='upper left')
		ax2.legend()   

	# Plot current through VClamp for different holding potentials
	flatten = lambda vec: [j for m in vec for j in m] 
	h.tstop = 2200 
	plt.figure() 
	all_mean = []
	COLORS = iter(['skyblue', 'orange', 'crimson', 'blue', 'xkcd:magenta', 'orchid', 'green'])
	AMPs = range(-100, 21, 20)

	# COLORS = iter(['k', 'darkblue', 'skyblue','orange','xkcd:magenta'])
	# AMPs = [-95, -85, -70, -40, -15]
	for amp1 in AMPs: 
		VClamp.amp[0] = amp1 
		VClamp2.amp[0] = amp1 
		h.run() 
		color_=next(COLORS) 
		MEAN = [] 
		for axon in Pyr_pop.inputs['Pyr0']: 
			# for C in range(len(Pyr_pop.inputs['Pyr0'][axon]['synapses'])): 
				# AMPA = Pyr_pop.inputs['Pyr0'][axon]['i_AMPA'][C] 
 				# NMDA = Pyr_pop.inputs['Pyr0'][axon]['i_NMDA'][C] 
				# TOT = [AMPA[j]+NMDA[j] for j in range(len(AMPA))] 
				# peaks = [1000*(TOT[j]) for j in range(len(TOT)-1) if (TOT[j]>TOT[j-1]) and (TOT[j]>TOT[j+1])] 

			peaks = [1000*i_clamp[j] for j in range(2000, len(i_clamp)-1) if (abs(i_clamp[j])>abs(i_clamp[j-1])) and (abs(i_clamp[j])>abs(i_clamp[j+1]))]
			plt.plot([amp1]*len(peaks), peaks,'.',color=color_) 
			MEAN.append(peaks) 
		
		plt.plot(amp1, np.mean(flatten(MEAN)), 'o', color="none", markeredgecolor='k') 
		all_mean.append(np.mean(flatten(MEAN)))
	
	plt.ylabel('I (pA)') 
	plt.plot(AMPs, all_mean, 'k', LineWidth=0.3)            
           

elif exp_type == 'find_reversal':
	plt.figure()
	h.tstop = 1000 + clamp_stabilization_time
	VClamp.dur[0] = h.tstop
	VClamp2.dur[0] = h.tstop
	for amp in np.arange(-2, 2, 0.25):
		VClamp.amp[0] = amp
		VClamp2.amp[0] = amp

		h.run()

		baseline = min([abs(j) for j in list(i_clamp)[int(200/h.dt):]])

		plt.plot(t, [j-baseline for j in i_clamp], label=r'$V_{amp}$ = '+str(VClamp.amp[0]))

	plt.legend()
	plt.ylim([0.02, 0.05])


plt.figure()
plt.plot(v, i_clamp)








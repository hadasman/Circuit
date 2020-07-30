from neuron import h,gui
import pdb, os
import matplotlib.pyplot as plt
import _pickle as cPickle
from tqdm import tqdm
import numpy as np

from Population import Population
from Parameter_Initialization import *

def get_GIDs(upload_from):

	chosen_GIDs = {}
	chosen_GIDs['pyr'] 			  = cPickle.load(open('{}/chosen_pyr.p'.format(upload_from), 'rb'))
	chosen_GIDs['PV'] 			  = cPickle.load(open('{}/chosen_PV.p'.format(upload_from), 'rb'))
	chosen_GIDs['SOM'] 			  = cPickle.load(open('{}/chosen_SOM.p'.format(upload_from), 'rb'))
	chosen_PV_n_contacts  		  = cPickle.load(open('{}/chosen_PV_n_contacts.p'.format(upload_from), 'rb'))

	thalamic_GIDs = {}
	thalamic_GIDs['to_pyr'] = cPickle.load(open('{}/connecting_gids_to_pyr.p'.format(upload_from), 'rb'))
	thalamic_GIDs['to_PV']  = cPickle.load(open('{}/connecting_gids_to_PV.p'.format(upload_from), 'rb'))
	thalamic_GIDs['to_SOM'] = cPickle.load(open('{}/connecting_gids_to_SOM.p'.format(upload_from), 'rb'))
	
	return chosen_GIDs, chosen_PV_n_contacts, thalamic_GIDs

def CreatePopulations(n_pyr=0, n_PV=0, n_SOM=0):
	Pyr_pop, PV_pop, SOM_pop = [None]*3

	if n_pyr > 0:
		print('\n==================== Creating pyramidal cell ====================')
		Pyr_pop = Population('Pyr', pyr_morph_path, pyr_template_path, pyr_template_name)
		Pyr_pop.addCell()
		Pyr_pop.name_to_gid['Pyr0'] = chosen_GIDs['pyr']

	if n_PV > 0:
		print('\n==================== Creating PV population ====================')
		PV_pop = Population('PV', PV_morph_path, PV_template_path, PV_template_name)
		for i in tqdm(range(n_PV)):
			PV_cell_name = 'PV%i'%i
			PV_pop.addCell()
			PV_pop.name_to_gid[PV_cell_name] = chosen_GIDs['PV'][i]
			PV_pop.moveCell(PV_pop.cells[PV_cell_name]['cell'], (i*350)-(100*(n_PV+1)), -500, 0) # Morphology Visualization

	if n_SOM > 0:
		print('\n==================== Creating SOM population ====================')
		SOM_pop = Population('SOM', SOM_morph_path, SOM_template_path, SOM_template_name)
		SOM_pop.addCell()	
		SOM_pop.name_to_gid['SOM0'] = chosen_GIDs['SOM']
		SOM_pop.moveCell(SOM_pop.cells['SOM0']['cell'], 0, -1000, 0) # Morphology Visualization

	return Pyr_pop, PV_pop, SOM_pop

def set_ThalamicInputs(Pyr_pop, PV_pop, SOM_pop, thalamic_GIDs):

	if Pyr_pop:
		where_Pyr_syns = ['basal_dendrites'] # Constant random from file

		print('\n==================== Connecting thalamic inputs to Pyramidal cell (input weight: {}uS) ===================='.format(Pyr_input_weight))
		Pyr_pop.addInput(list(Pyr_pop.cells.keys())[0], record_syns=True, where_synapses=where_Pyr_syns, weight=Pyr_input_weight, thalamic_activations_filename=activated_filename, connecting_gids=thalamic_GIDs['to_pyr'])

	if PV_pop:
		where_PV_syns = ['basal_dendrites', 'apical_dendrites']

		print('\n==================== Connecting thalamic inputs to PV cells (input weight: {}uS) ===================='.format(PV_input_weight))
		if PV_to_Pyr_source == 'voltage':
			for i, PV_cell_name in enumerate(tqdm(PV_pop.cells)):
				PV_pop.addInput(PV_cell_name, record_syns=True, where_synapses=where_PV_syns, weight=PV_input_weight, thalamic_activations_filename=activated_filename, connecting_gids=thalamic_GIDs['to_PV'][PV_pop.name_to_gid[PV_cell_name]])	

		elif PV_to_Pyr_source == 'spike_times':
			print('Loading PV spike times from file ({})'.format(filenames['PV_spike_times']))
			PV_spike_times = cPickle.load(open(filenames['PV_spike_times'], 'rb'))

			for PV_cell_name in PV_pop.cells:
				PV_pop.cells[PV_cell_name]['soma_v'] = PV_spike_times['cells'][PV_cell_name]['soma_v']

	if SOM_pop:
		where_SOM_syns = ['basal_dendrites']

		SOM_input_source = 'thalamic_input'
		print('\n==================== Connecting {} inputs to SOM cell (input weight: {}uS) ===================='.format(SOM_input_source, SOM_input_weight))
		if SOM_input_source == 'Pyr':
			SOM_pop.connectCells('SOM0', Pyr_pop, 'Pyr0', 'voltage', [SOM_pop.cells['SOM0']['soma']], n_Pyr_to_SOM_syns, 'random', weight=SOM_input_weight, delay=SOM_input_delay)

		elif SOM_input_source == 'thalamic_input':
			SOM_pop.addInput('SOM0', record_syns=True, where_synapses=where_SOM_syns, weight=SOM_input_weight, thalamic_activations_filename=activated_filename, connecting_gids=thalamic_GIDs['to_SOM']) 

def set_IClamp(pop, cell_name, input_weight):
	stims, cons = [], []
	
	for axon in pop.inputs[cell_name]:
		# del pop.inputs[cell_name][axon]['netcons']

		for syn in pop.inputs[cell_name][axon]['synapses']:
			loc = syn.get_segment()
			stim = h.IClamp(loc)
			stim.delay = single_stim_time
			stim.dur = 1 #1 ms pulse
			stim.amp = 1

			con = h.NetCon(stim, syn)
			con.weight[0] = input_weight

			stims.append(stim)
			cons.append(cons)

	return stims, cons

def putspikes():
	
	# Pyramidal inputs
	if Pyr_pop:
		for axon in Pyr_pop.inputs['Pyr0']:

			stim_times = Pyr_pop.inputs['Pyr0'][axon]['stim_times']
			
			for netcon in Pyr_pop.inputs['Pyr0'][axon]['netcons']:
				for T in stim_times:
					if (T >= single_stim_time) and ( T<= single_stim_time+syns_window):
						netcon.event(single_stim_time + Pyr_input_delay)
						# netcon.event(T + Pyr_input_delay)

	# PV inputs
	if PV_pop:
		for PV_cell in PV_pop.inputs:
			for axon in PV_pop.inputs[PV_cell]:

				# stim_times = [i for i in PV_pop.inputs['PV0'][axon]['stim_times'] if (i>=single_stim_time)]
				stim_times = PV_pop.inputs[PV_cell][axon]['stim_times']
				for netcon in PV_pop.inputs[PV_cell][axon]['netcons']:
					for T in stim_times:
						if (T >= single_stim_time) and ( T<= single_stim_time+syns_window):
							netcon.event(single_stim_time + PV_input_delay)
							# netcon.event(T + PV_input_delay)
	# SOM inputs
	if SOM_pop:
		for SOM_cell in SOM_pop.inputs:
			for axon in SOM_pop.inputs[SOM_cell]:

				# stim_times = [i for i in SOM_pop.inputs['SOM0'][axon]['stim_times'] if (i>=single_stim_time)]
				stim_times = SOM_pop.inputs[SOM_cell][axon]['stim_times']
				for netcon in SOM_pop.inputs[SOM_cell][axon]['netcons']:
					for T in stim_times:
						if (T >= single_stim_time) and ( T<= single_stim_time+syns_window):
							netcon.event(single_stim_time + SOM_input_delay)
							# netcon.event(T + SOM_input_delay)

def get_EPSPs(single_stim_time):

	stim_idx = int(np.floor(single_stim_time/h.dt))
	end_stim_idx = int(np.floor((single_stim_time+300)/h.dt))

	if Pyr_pop:
		cut_Pyr_v = Pyr_soma_v[stim_idx:end_stim_idx]
		Pyr_peak = [cut_Pyr_v[i] for i in range(len(cut_Pyr_v)-1) if (cut_Pyr_v[i]>cut_Pyr_v[i+1]) and (cut_Pyr_v[i]>cut_Pyr_v[i-1])]
		if len(Pyr_peak)>0:
			Pyr_peak = Pyr_peak[0]
		else:
			Pyr_peak = cut_Pyr_v[0]
		Pyr_peak_idx = Pyr_soma_v.index(Pyr_peak)
		Pyr_EPSP = Pyr_peak - Pyr_soma_v[stim_idx - 1]
	else:
		Pyr_peak, Pyr_peak_idx, Pyr_EPSP = [], [], []
	
	if PV_pop:
		cut_PV_v = PV_soma_v[stim_idx:end_stim_idx]	
		PV_peak = [cut_PV_v[i] for i in range(len(cut_PV_v)-1) if (cut_PV_v[i]>cut_PV_v[i+1]) and (cut_PV_v[i]>cut_PV_v[i-1])]
		if len(PV_peak)>0:
			PV_peak = PV_peak[0]
		else:
			PV_peak = cut_PV_v[0]

		PV_peak_idx = PV_soma_v.index(PV_peak)	
		PV_EPSP = PV_peak - PV_soma_v[stim_idx - 1]
	else:
		PV_peak, PV_peak_idx, PV_EPSP = [], [], []

	if SOM_pop:
		cut_SOM_v = SOM_soma_v[stim_idx:end_stim_idx]	
		SOM_peak = [cut_SOM_v[i] for i in range(len(cut_SOM_v)-1) if (cut_SOM_v[i]>cut_SOM_v[i+1]) and (cut_SOM_v[i]>cut_SOM_v[i-1])]
		if len(SOM_peak)>0:
			SOM_peak = SOM_peak[0]
		else:
			SOM_peak = cut_SOM_v[0]

		SOM_peak_idx = SOM_soma_v.index(SOM_peak)
		SOM_EPSP = SOM_peak - SOM_soma_v[stim_idx - 1]
	else:
		SOM_peak, SOM_peak_idx, SOM_EPSP = [], [], []


	return [Pyr_peak, Pyr_peak_idx, Pyr_EPSP], [PV_peak, PV_peak_idx, PV_EPSP], [SOM_peak, SOM_peak_idx, SOM_EPSP]

def plot_setup():
	fig, ax = plt.subplots(3, 1, figsize=(9, 7.5))
	fig.suptitle('Somatic EPSP from Synchronous Thalamic Input', size=15) # I took all thalamic axons that were active in a 50ms window after stimulus presentation (single_stim_time), and activated them all at 2000ms, to match Yuzhar results.
	fig.subplots_adjust(hspace=0.48, bottom=0.08, top=0.89) 
	ax[0].set_xlim([1997, 2025])
	ax[1].set_xlim([1997, 2025])
	ax[2].set_xlim([1997, 2025])

	f1 = plt.figure(figsize=(9, 7.5))
	ax1 = plt.gca()
	f1.subplots_adjust(hspace=0.48, bottom=0.08, top=0.91, left=0.1, right=0.95) 
	ax1.set_title('Overlay (baseline subtracted)')
	ax1.set_ylabel('V (mV)')
	ax1.set_xlabel('T (ms)')
	ax1.set_xlim([1995, 2050])
	ax1.axvline(2000, color='gray', LineStyle='--', LineWidth=1)

	f2 = plt.figure(figsize=(9, 7.5))
	ax2 = plt.gca()
	f2.subplots_adjust(hspace=0.48, bottom=0.08, top=0.91, left=0.1, right=0.95) 
	ax2.set_title('Normalized Overlay (baseline subtracted)')
	ax2.set_ylabel('V (mV)')
	ax2.set_xlabel('T (ms)')
	ax2.set_xlim([1995, 2050])
	ax2.set_ylim([-0.2, 1.2]) 

	ax2.axvline(2000, color='gray', LineStyle='--', LineWidth=1)

	f3, ax3 = plt.subplots(figsize=(9, 7.5))  
	ax3.set_title('Mean AMPA Conductance of Individual Synapses')                                                       
	ax3.set_xlabel('T (ms)')                                                                                            
	ax3.set_ylabel('G (nS)')                                                                                        
	ax3.set_xlim([1990, 2020])

	F, twiny1 = plt.subplots(3, 1, figsize=(9, 7.5))
	F.subplots_adjust(hspace=0.36, bottom=0.08, top=0.93, left=0.1, right=0.95) 
	twiny2 = [i.twinx() for i in twiny1]

	twiny1[0].set_ylabel('V (mV)')                                                                                       
	twiny2[0].set_ylabel('G (nS)') 
	twiny1[0].set_xlim([1997, 2025])
	twiny1[1].set_ylabel('V (mV)')                                                                                       
	twiny2[1].set_ylabel('G (nS)') 
	twiny1[1].set_xlim([1997, 2025])                                                                                      
	twiny1[2].set_ylabel('V (mV)')                                                                                       
	twiny2[2].set_ylabel('G (nS)')                                                                                                                                                                             
	twiny1[2].set_xlabel('T (ms)') 	                                                                                	
	twiny1[2].set_xlim([1997, 2025]) 

	return fig, ax, f1, ax1, f2, ax2, f3, ax3, F, twiny1, twiny2

def plot_EPSPs():
	if Pyr_pop:
		ax[0].plot(t, Pyr_soma_v, label='g$_{in} = $%.1fnS'%Pyr_input_weight)
		ax[0].plot([t[Pyr_EPSP[1]]]*2, [Pyr_EPSP[0], Pyr_EPSP[0]-Pyr_EPSP[2]], 'black', LineWidth=1)
		ax[0].set_title('Pyramidal (EPSP = %.1fmV, %.0f thalamic axons, %.0f total contacts)'%(Pyr_EPSP[2], len(thalamic_GIDs['to_pyr']), sum([i[1] for i in thalamic_GIDs['to_pyr']])))
		ax[0].set_ylabel('V (mV)')
		ax[0].set_xlim([500, h.tstop])
		ax[0].legend()

		Pyr_base = [i-Pyr_soma_v[79000] for i in Pyr_soma_v]
		Pyr_norm = [i/abs(Pyr_EPSP[0]-Pyr_soma_v[79000]) for i in Pyr_base]
		ax1.plot(t, Pyr_base, label='Pyr')
		ax2.plot(t, Pyr_norm, label='Pyr')
		
	if PV_pop:
		ax[1].plot(t, PV_soma_v, label='g$_{in} = $%.1fnS'%PV_input_weight)
		ax[1].plot([t[PV_EPSP[1]]]*2, [PV_EPSP[0], PV_EPSP[0]-PV_EPSP[2]], 'black', LineWidth=1)
		ax[1].set_title('PV (EPSP = %.1fmV, %.0f thalamic axons, %.0f total contacts)'%(PV_EPSP[2], len(thalamic_GIDs['to_PV'][PV_pop.name_to_gid['PV0']]), sum([i[1] for i in thalamic_GIDs['to_PV'][PV_pop.name_to_gid['PV0']]])))
		ax[1].set_ylabel('V (mV)')
		ax[1].set_xlim([500, h.tstop])
		ax[1].legend()

		PV_base = [i-PV_soma_v[79000] for i in PV_soma_v]
		PV_norm = [i/abs(PV_EPSP[0]-PV_soma_v[79000]) for i in PV_base]
		ax1.plot(t, PV_base, label='PV')
		ax2.plot(t, PV_norm, label='PV')

	if SOM_pop:
		ax[2].plot(t, SOM_soma_v, label='g$_{in} = $%.1fnS'%SOM_input_weight)
		ax[2].plot([t[SOM_EPSP[1]]]*2, [SOM_EPSP[0], SOM_EPSP[0]-SOM_EPSP[2]], 'black', LineWidth=1)
		ax[2].set_title('SOM (EPSP = %.1fmV, %.0f thalamic axons, %.0f total contacts)'%(SOM_EPSP[2], len(thalamic_GIDs['to_SOM']), sum([i[1] for i in thalamic_GIDs['to_SOM']])))
		ax[2].set_ylabel('V (mV)')
		ax[2].set_xlabel('T (ms)')
		ax[2].set_xlim([500, h.tstop])
		ax[2].legend()

		SOM_base = [i-SOM_soma_v[79000] for i in SOM_soma_v]
		SOM_norm = [i/abs(SOM_EPSP[0]-SOM_soma_v[79000]) for i in SOM_base]
		ax1.plot(t, SOM_base, label='SOM')
		ax2.plot(t, SOM_norm, label='SOM')

	return ax, ax1, ax2

# Pyr_input_weight = 0.6
# PV_input_weight = 0.2
upload_from 					  = 'GIDs_instantiations/pyr_72851_between_6666_9600'
chosen_GIDs, _, thalamic_GIDs 	  = get_GIDs(upload_from)

Pyr_pop, PV_pop, SOM_pop 		  = CreatePopulations(n_pyr=1, n_PV=1, n_SOM=1)
t = h.Vector()
t.record(h._ref_t)

activated_filename = filenames['thalamic_activations_6666']
set_ThalamicInputs(Pyr_pop, PV_pop, SOM_pop, thalamic_GIDs)
single_stim_time = 2000

exp_type = 'events' #'events'
if exp_type == 'IClamp':

	if Pyr_pop:
		pyr_stims, pyr_cons = set_IClamp(Pyr_pop, 'Pyr0', Pyr_input_weight)

	if PV_pop:
		pv_stims, pv_cons = set_IClamp(PV_pop, 'PV0', PV_input_weight)

	if SOM_pop:
		som_stims, som_cons = set_IClamp(SOM_pop, 'SOM0', SOM_input_weight)

elif exp_type == 'events':
	events = h.FInitializeHandler(putspikes)

	if Pyr_pop:
		pyr_cons = [j for i in [Pyr_pop.inputs['Pyr0'][axon]['netcons'] for axon in Pyr_pop.inputs['Pyr0']] for j in i]
	if PV_pop:
		pv_cons = [j for i in [[PV_pop.inputs[PV_cell][axon]['netcons'] for axon in PV_pop.inputs[PV_cell]] for PV_cell in PV_pop.cells] for j in i]
	if SOM_pop:
		som_cons = [j for i in [SOM_pop.inputs['SOM0'][axon]['netcons'] for axon in SOM_pop.inputs['SOM0']] for j in i]

syns_window = 100
h.tstop = 4000

# Block AP-related channels
mt = h.MechanismType(0)
mechanism_list = ['K_Tst', 
				  'SKv3_1', 
				  'Nap_Et2', 
				  'NaTs2_t', 
				  'Ih', 
				  'Im', 
				  'NaTa_t', 
				  'K_Pst', 
				  'SK_E2', 
				  'CaDynamics_E2', 
				  'Ca_LVAst', 
				  'Ca', 
				  'Ca_HVA'
				 ]
# SOM_input_weight = 0.2 
# PV_input_weight = 0.2
print('NOTICE: Changing g_pas to 0.001!')
for sec in h.allsec():
	sec.g_pas = 0.001
	
	for mech in mechanism_list:
		
		if mech in sec.psection()['density_mechs'].keys(): # Check if mechanism exists in section
			mt.select(mech) 	# Select the current mechanism
			mt.remove(sec=sec) 	# Remove mechanism from section

print('Running simulation'); h.run()

if Pyr_pop:
	Pyr_soma_v = list(Pyr_pop.cells['Pyr0']['soma_v'])
if PV_pop: 
	PV_soma_v  = list(PV_pop.cells['PV0']['soma_v'])
if SOM_pop:
	SOM_soma_v = list(SOM_pop.cells['SOM0']['soma_v'])

Pyr_EPSP, PV_EPSP, SOM_EPSP = get_EPSPs(single_stim_time)

fig, ax, f1, ax1, f2, ax2, f3, ax3, F, twiny1, twiny2 = plot_setup()

ax, ax1, ax2 = plot_EPSPs()
# stim_times = cPickle.load(open('thalamocortical_Oren/SSA_spike_times/stim_times.p','rb'))[6666]
# stim_times = [i[0] for i in stim_times if i[0]<h.tstop]
stim_times = [single_stim_time]
for T in stim_times:
	ax[0].axvline(T, LineStyle='--', color='gray')
	ax[1].axvline(T, LineStyle='--', color='gray')
	ax[2].axvline(T, LineStyle='--', color='gray')

colors = iter(['orchid', 'coral', 'xkcd:magenta'])                                                                                                
input_weights = {'Pyr': Pyr_input_weight, 'PV': PV_input_weight, 'SOM': SOM_input_weight}
for i, POP in enumerate([Pyr_pop, PV_pop, SOM_pop]): 
	if POP:
		example_cell = [i for i in POP.cells][0]
		C = next(colors)

		# weight = [POP.inputs[example_cell][axon]['netcons'][0].weight[0] for axon in POP.inputs[example_cell]][0]
		weight = input_weights[POP.population_name]
		mean_G = [1000*np.mean([np.mean(POP.inputs[i][axon]['g_AMPA'], axis=0) for axon in POP.inputs[i]], axis=0) for i in POP.cells][0]
		ax3.plot(t, mean_G, color=C, label=POP.population_name + ' (g = %.1f)'%weight) 
		mean_G = [1000*np.mean([np.mean(POP.inputs[i][axon]['g_NMDA'], axis=0) for axon in POP.inputs[i]], axis=0) for i in POP.cells][0]
		ax3.plot(t, mean_G, color=C, LineStyle='--', alpha=0.5) 

		chosen_cell = [i for i in POP.cells][0]

		twiny1[i].plot(t, POP.cells[chosen_cell]['soma_v'], label='Soma V', color='coral', LineWidth=2)
		mean_G = [1000*np.mean([np.mean(POP.inputs[i][axon]['g_AMPA'], axis=0) for axon in POP.inputs[i]], axis=0) for i in POP.cells][0] 
		twiny2[i].plot(t, mean_G, color='xkcd:magenta', label='AMPA (g = %.1f)'%weight)  
		mean_G = [1000*np.mean([np.mean(POP.inputs[i][axon]['g_NMDA'], axis=0) for axon in POP.inputs[i]], axis=0) for i in POP.cells][0] 
		twiny2[i].plot(t, mean_G, color='xkcd:magenta', alpha=0.5, label='NMDA')  

		twiny1[i].legend(loc='upper left')                                                                                                   
		twiny2[i].legend(loc='upper right')  
		twiny1[i].set_ylim([twiny1[i].get_ylim()[0], -60]) 
		twiny2[i].set_ylim([twiny2[i].get_ylim()[0], 0.14]) 
		twiny1[i].set_title('Somatic Voltage & Mean Synapse Conductances ({} cell)'.format(POP.population_name))  
 

ax1.legend()
ax2.legend()
ax3.legend()    




















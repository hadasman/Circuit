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
os.chdir('../MIT_spines')
from Cell import Cell
os.chdir('../Circuit')

# ============================================  Define Functions & Constants  ============================================
cell_details_filename		  		= 'thalamocortical_Oren/thalamic_data/cells_details.pkl'
thalamic_locs_filename		  		= 'thalamocortical_Oren/thalamic_data/thalamic_axons_location_by_gid.pkl'
thal_connections_filename 	  		= 'thalamocortical_Oren/thalamic_data/thalamo_cortical_connectivity.pkl'
thalamic_activations_6666_filename 	= 'thalamocortical_Oren/SSA_spike_times/input6666_106746.dat'
thalamic_activations_9600_filename 	= 'thalamocortical_Oren/SSA_spike_times/input9600_109680.dat'
pyr_connectivity_filename  	  		= 'thalamocortical_Oren/pyramidal_connectivity_num_connections.p'
cell_type_gids_filename		  		= 'thalamocortical_Oren/thalamic_data/cell_type_gids.pkl'

thalamic_locations = pd.read_pickle(thalamic_locs_filename)

def getChosenGIDs(pyr_gids, PV_gids, freq1, freq2, min_freq=4000, df_dx=3.5, thalamic_locs_filename=thalamic_locs_filename, cell_details_filename=cell_details_filename, pyr_connectivity_filename=pyr_connectivity_filename):
	# df/dx is n units [octave/mm]
	thalamic_locations   = pd.read_pickle(thalamic_locs_filename)
	cell_details 	 	 = pd.read_pickle(cell_details_filename)
	all_pyr_connectivity = pd.read_pickle(pyr_connectivity_filename)
	
	min_freq_loc 	 = min(thalamic_locations.x) 

	def get_AxonLoc(freq, min_freq, min_freq_loc, df_dx=df_dx):

		dOctave = log(freq / min_freq, 2) # In octaves
		d_mm =  dOctave / df_dx			  # In millimeters
		d_microne = d_mm * 1000 # In micrones
		freq_loc = min_freq_loc + d_microne

		return freq_loc

	def get_pyramidal(pyr_gids, freq1_loc, freq2_loc, cell_details):		
		mid_loc   = np.mean([freq1_loc, freq2_loc])

		dists = [abs(cell_details.loc[gid].x-mid_loc) for gid in pyr_gids]	
		sorted_idx = [dists.index(i) for i in sorted(dists)]	

		chosen_pyr_gid = pyr_gids[sorted_idx[0]]
		return chosen_pyr_gid

	def get_PVs(PV_gids, chosen_pyr_gid, all_pyr_connectivity):

		pyr_connectivity = all_pyr_connectivity[chosen_pyr_gid]

		chosen_PV_gids = []
		for gid in pyr_connectivity:
			if int(gid) in PV_gids:
				chosen_PV_gids.append([int(gid), pyr_connectivity[gid]])

		return chosen_PV_gids

	freq1_loc = get_AxonLoc(freq1, min_freq, min_freq_loc)
	freq2_loc = get_AxonLoc(freq2, min_freq, min_freq_loc)
	
	# Pyramidal
	chosen_pyr_gid = get_pyramidal(pyr_gids, freq1_loc, freq2_loc, cell_details)

	# PV
	chosen_PV_gids = get_PVs(PV_gids, chosen_pyr_gid, all_pyr_connectivity)

	return chosen_pyr_gid, chosen_PV_gids

def RunSim(Pyr_pop, PV_pop, v_init=-75, tstop=154*1000):
	# Oren's simulation length is 154 seconds, I leave some time for last inputs to decay
	t = h.Vector()
	t.record(h._ref_t)
	h.tstop = tstop

	h.v_init = v_init

	# IMPORTANT NOTIC: MUST be defined in the same level as h.run() is called (inside same function of ourside of any function) or else hoc doesn't recognize netcon name!
	# pyr_events = []
	# for axon in Pyr_pop.inputs['Pyr0']:
	# 	stim_times = Pyr_pop.inputs['Pyr0'][axon]['stim_times']
	# 	for con in Pyr_pop.inputs['Pyr0'][axon]['netcons']:					
	# 		for time in stim_times:
	# 			pyr_events.append(h.FInitializeHandler('nrnpython("con.event({})")'.format(time+delay)))

	# PV_events = []
	# for PV_cell_name in PV_pop.inputs:
	# 	for axon in PV_pop.inputs[PV_cell_name]:
	# 		stim_times = PV_pop.inputs[PV_cell_name][axon]['stim_times']
	# 		for CON in PV_pop.inputs[PV_cell_name][axon]['netcons']:		
	# 			for time in stim_times:
	# 				PV_events.append(h.FInitializeHandler('nrnpython("CON.event({})")'.format(time+delay)))

	h.finitialize()
	h.run()

	_, (ax1, ax2, ax3) = plt.subplots(3, 1)

	PYR_cell = Pyr_pop.cells['Pyr0']['cell']
	ax1.plot(t, PYR_cell.soma_v)
	ax1.set_title('Pyramidal soma voltage')
	ax1.set_xlabel('Time (ms)')
	ax1.set_ylabel('Voltage (mV)')

	for cell in PV_pop.cells:
		ax2.plot(t, PV_pop.cells[cell]['soma_v'], label=cell)
	ax2.set_title('PV soma coltage')
	ax2.set_xlabel('Time (ms)')
	ax2.set_ylabel('Voltage (mV)')

	return PYR_cell, PV_pop

pyr_template_path 	= 'EPFL_models/L4_PC_cADpyr230_1' # '../MIT_spines/cell_templates'
pyr_template_name 	= 'cADpyr230_L4_PC_f15e35e578' # 'whole_cell'
# pyr_morph_filename 	= 'dend-C170897A-P3_axon-C260897C-P4_-_Clone_4.asc' # 'cell1.asc'
pyr_morph_path 		= '{}/morphology'.format(pyr_template_path) # 'L5PC/'

PV_template_path 	= 'EPFL_models/L4_LBC_cNAC187_1'
PV_template_name 	= 'cNAC187_L4_LBC_990b7ac7df'
# PV_morph_filename 	= 'C050398B-I4_-_Clone_3.asc'
PV_morph_path 		= '{}/morphology/'.format(PV_template_path)

pyr_type = 'L4_PC'
PV_type = 'L4_LBC'
n_VIP 		  = 5

delay = 0# TEMPORARY: CHANGE THIS
freq1 = 6666
freq2 = 9600
activated_filename = thalamic_activations_6666_filename
activated_standard_freq = freq1*(str(freq1) in activated_filename) + freq2*(str(freq2) in activated_filename)

# ===============================================  Choose GIDs  ===============================================
print('\n========== Choosing GIDs from BlueBrain Database (pyramidal between {}Hz and P{}Hz) =========='.format(freq1, freq2))

cell_type_gids 	= cPickle.load(open(cell_type_gids_filename,'rb'))     
pyr_GIDs 		= cell_type_gids[pyr_type]
PV_GIDs 		= cell_type_gids[PV_type]

chosen_pyr, chosen_PV = getChosenGIDs(pyr_GIDs, PV_GIDs, freq1, freq2) # chosen_pyr==gid, chosen_V==[[gid, no_contancts],...]

# ===============================================  Create Cell Populations  ===============================================

print('\n========== Creating pyramidal cell ==========')
Pyr_pop = Population('Pyr', pyr_morph_path, pyr_template_path, pyr_template_name)
Pyr_pop.addCell()

print('\n========== Creating PV population ==========')
n_PV  		  = len(chosen_PV)
PV_pop = Population('PV', PV_morph_path, PV_template_path, PV_template_name)
for i in range(n_PV):
	PV_pop.addCell()

# ==============================================  Stimulus Analysis  ==============================================
# !! THINK if i can use the activations here (self.thalamic_activations) for the Population.addInput!

stimuli = {}
connecting_gids = []
thal_connections = pd.read_pickle(thal_connections_filename)

# Find thalamic GIDs connecting the the pyramidal cell
for con in thal_connections.iterrows():
	if con[1].post_gid == chosen_pyr:
		connecting_gids.append(con[1].pre_gid) # [presynaptic gid, no. of contacts]
stimuli[freq1] = Stimulus(freq1, freq2, 'thalamocortical_Oren/SSA_spike_times/stim_times.p', thalamic_activations_6666_filename, axon_gids=connecting_gids)
stimuli[freq2] = Stimulus(freq2, freq1, 'thalamocortical_Oren/SSA_spike_times/stim_times.p', thalamic_activations_9600_filename, axon_gids=connecting_gids)

plot_thalamic_responses = False
if plot_thalamic_responses:

	stim_ax = stimuli[freq1].axonResponses(thalamic_locations, color='red')
	stim_ax = stimuli[freq2].axonResponses(thalamic_locations, color='blue', h_ax=stim_ax)

	stimuli[freq1].tonotopic_location(thalamic_locations, color='red', h_ax=stim_ax)
	stimuli[freq2].tonotopic_location(thalamic_locations, color='blue', h_ax=stim_ax)

	stim_ax.set_title('Thalamic Axons (Connected to Chosen Pyramidal) Firing Rate for 2 Frequencies\nPyramidal GID: {}, Spontaneous Axon FR: {}'.format(chosen_pyr, 0.5))
	stim_ax.set_xlim([65, 630])

# ==============================================  Connect Populations with Synapses and Inputs  ==============================================

print('\n========== Connecting thalamic inputs to PV (standard: {}Hz) =========='.format(activated_standard_freq))
for i, PV_cell_name in enumerate(PV_pop.cells):
	PV_gid = chosen_PV[i][0]

	PV_pop.addInput(PV_cell_name, PV_gid, thalamic_activations_filename=activated_filename, thalamic_connections_filename=thal_connections_filename)
	PV_events = [] 
	for axon in PV_pop.inputs[PV_cell_name]:
		stim_times = PV_pop.inputs[PV_cell_name][axon]['stim_times']
		for con in PV_pop.inputs[PV_cell_name][axon]['netcons']:	
			for time in stim_times:
				PV_events.append(h.FInitializeHandler('nrnpython("PV_pop.inputs[\'PV0\'][\'{}\'][\'netcons\'][{}].event({})")'.format(axon, i, time+delay)))

print('\n========== Connecting thalamic inputs to Pyramidal cell (standard: {}Hz) =========='.format(activated_standard_freq))
Pyr_pop.addInput(list(Pyr_pop.cells.keys())[0], chosen_pyr, thalamic_activations_filename=activated_filename, thalamic_connections_filename=thal_connections_filename)
# IMPORTANT NOTIC: MUST be defined outside of functino or else hoc doesn't recognize netcon name!
pyr_events = []
for axon in Pyr_pop.inputs['Pyr0']:
	stim_times = Pyr_pop.inputs['Pyr0'][axon]['stim_times']
	for i in range(len(Pyr_pop.inputs['Pyr0'][axon]['netcons'])):		
		for time in stim_times:
			pyr_events.append(h.FInitializeHandler('nrnpython("Pyr_pop.inputs[\'Pyr0\'][\'{}\'][\'netcons\'][{}].event({})")'.format(axon, i, time+delay)))


print('Connecting PV population to Pyramidal cell')
PV_pop.connectCells([PV_pop.cells['PV0']['cell'].axon[1](1)]*3, [PYR_cell.cell.soma[0]], PV_density, PV_dist) # Adds self.connections to Population
print('\n***Assuming isopotential soma and perisomatic PV connections: all PV synapses are placed on soma(0.5)')

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
PYR_cell, PV_pop = RunSim(PYR_cell, PV_pop)

# ==============================================  Morphology Visualization  ==============================================

for i in range(n_PV):  
	PV_pop.moveCell(PV_pop.cells['PV{}'.format(i)]['cell'], (i*350)-(100*(n_PV+1)), -500, 0)  


# ==============================================  Connectivity Analysis  ==============================================
def analyzeConnectivityFromThalamus(which_layer, which_mtype, cell_details_filename=cell_details_filename, thal_connections_filename=thal_connections_filename):

	cell_details = pd.read_pickle(cell_details_filename)
	thal_connections = pd.read_pickle(thal_connections_filename)

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










import matplotlib
matplotlib.use("TkAgg")
import pandas as pd
import numpy as np
from neuron import gui, h
import pdb, os, sys
import matplotlib.pyplot as plt
from Population import Population
from math import log

os.chdir('../MIT_spines')
from Cell import Cell
os.chdir('../Circuit')

cell_details_filename		='thalamocortical_Oren/thalamic_data/cells_details.pkl'
thalamic_locs_filename		='thalamocortical_Oren/thalamic_data/thalamic_axons_location.pkl'
thal_connections_filename 	= 'thalamocortical_Oren/thalamic_data/thalamo_cortical_connectivity.pkl'

# ============================================  Define Functions & Constants  ============================================
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

def getGIDs(pyr_type, PV_type):
	
	cell_details = pd.read_pickle(cell_details_filename)
	pyr_gids, PV_gids = [], []

	for i in range(len(cell_details)): 

		if cell_details.iloc[i].mtype==pyr_type:
			pyr_gids.append(cell_details.index[i])
		elif cell_details.iloc[i].mtype==PV_type:
			PV_gids.append(cell_details.index[i])

	return pyr_gids, PV_gids

def getPyramidalGID(pyr_gids, freq1, freq2, min_freq=4000, df_dx=3.5, cell_details_filename=cell_details_filename, thalamic_locs_filename=thalamic_locs_filename):
	# df/dx is n units [octave/mm]

	thalamic_locs = pd.read_pickle(thalamic_locs_filename)
	cell_details = pd.read_pickle(cell_details_filename)
	min_freq_loc = min(thalamic_locs.x) 

	def get_AxonLoc(freq, min_freq, min_freq_loc, df_dx=df_dx):

		dOctave = log(freq / min_freq, 2) # In octaves
		d_mm =  dOctave / df_dx			  # In millimeters
		freq_loc = min_freq_loc + d_mm

		return freq_loc

	freq1_loc = get_AxonLoc(freq1, min_freq, min_freq_loc)
	freq2_loc = get_AxonLoc(freq2, min_freq, min_freq_loc)
	mid_loc   = np.mean([freq1_loc, freq2_loc])

	cur_min_dist = 100
	for gid in pyr_gids:
		if abs(mid_loc - cell_details.loc[gid].x) < cur_min_dist:
			cur_min_dist = abs(mid_loc - cell_details.loc[gid].x)
			cur_gid = gid

	return cur_gid

def RunSim(PYR_cell, PV_pop, VIP_pop, v_init=-75, tstop=170*60):
	# Oren's simulation length is 154 seconds, I leave some time for last inputs to decay
	t = h.Vector()
	t.record(h._ref_t)
	h.tstop = tstop

	h.v_init = v_init

	h.finitialize()
	h.run()

	_, (ax1, ax2, ax3) = plt.subplots(3, 1)

	ax1.plot(t, PYR_cell.soma_v)
	ax1.set_title('Pyramidal soma voltage')
	ax1.set_xlabel('Time (ms)')
	ax1.set_ylabel('Voltage (mV)')

	for cell in PV_pop.cells:
		ax2.plot(t, PV_pop.cells[cell]['soma_v'], label=cell)
	ax2.set_title('PV soma coltage')
	ax2.set_xlabel('Time (ms)')
	ax2.set_ylabel('Voltage (mV)')

	return PYR_cell, PV_pop, VIP_pop

pyr_template_path 	= 'EPFL_models/L23_PC_cADpyr229_1' # '../MIT_spines/cell_templates'
pyr_template_name 	= 'cADpyr229_L23_PC_5ecbf9b163' # 'whole_cell'
pyr_morph_filename 	= 'dend-C170897A-P3_axon-C260897C-P4_-_Clone_4.asc' # 'cell1.asc'
pyr_morph_path 		= '{}/morphology'.format(pyr_template_path) # 'L5PC/'

PV_template_path 	= 'EPFL_models/L23_LBC_cNAC187_1'
PV_template_name 	= 'cNAC187_L23_LBC_df15689e81'
PV_morph_filename 	= 'C050398B-I4_-_Clone_3.asc'
PV_morph_path 		= '{}/morphology/'.format(PV_template_path)

# thalamo_to_pyr_stats = analyzeConnectivityFromThalamus('L23', 'PC')
# thalamo_to_PV_stats  = analyzeConnectivityFromThalamus('L23', 'LBC')

# # Number of thalamic axons contacting each cell
# n_axons_to_pyr = thalamo_to_pyr_stats['per_cell']['mean_incoming_axons']
# n_axons_to_PV  = thalamo_to_PV_stats['per_cell']['mean_incoming_axons']

# # Number of contacts a thalamic axon makes on each cell
# n_contacts_to_pyr = thalamo_to_pyr_stats['per_cell']['mean_incoming_contacts']
# n_contacts_to_PV  = thalamo_to_PV_stats['per_cell']['mean_incoming_contacts']

pyr_type = 'L4_PC'
PV_type = 'L23_LBC'
n_PV  		  = 4
PV_density	  = 0.1 # Density of synapses from PV to pyramidal
PV_dist 	  = 'one'

n_VIP 		  = 5
# VIP_density	  = ... # ?Density of synapses from VIP to pyramidal or PV?

# ===============================================  Create Cell Populations  ===============================================
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

print('# Creating PV population')
PV_pop = Population('PV', PV_morph_path, PV_morph_filename, PV_template_path, PV_template_name)
for i in range(n_PV):
	PV_pop.addCell()

print('# Creating pyramidal cell')
Pyr_pop = Population('Pyr', pyr_morph_path, pyr_morph_filename, pyr_template_path, pyr_template_name)
Pyr_pop.addCell()

'''
PYR_cell = Cell(pyr_morph_path, 'pyr', pyr_morph_filename, pyr_template_name, pyr_template_path, verbose=False)
PYR_cell.loadMorph()

PYR_branches = [a for a in PYR_cell.cell.apic] + [d for d in PYR_cell.cell.dend]
PYR_cell.soma_v = h.Vector()
PYR_cell.soma_v.record(PYR_cell.cell.soma[0](0.5)._ref_v)
'''

# ==============================================  Connect Populations with Synapses and Inputs  ==============================================

pyr_GIDs, PV_GIDs = getGIDs(pyr_type, PV_type)
print('\nConnecting thalamic inputs to PV')
for i, PV_cell_name in enumerate(PV_pop.cells):
	PV_gid = PV_GIDs[i+10]
	PV_pop.addInput(PV_cell_name, PV_gid, thalamic_activations_filename="thalamocortical_Oren/SSA_spike_times/input6666_106746.dat", thalamic_connections_filename=thal_connections_filename)

print('Connecting thalamic inputs to Pyramidal cell')
pyr_gid = getPyramidalGID(pyr_GIDs, 6000, 9600)
Pyr_pop.addInput(list(Pyr_pop.cells.keys())[0], pyr_gid, thalamic_activations_filename="thalamocortical_Oren/SSA_spike_times/input6666_106746.dat", thalamic_connections_filename=thal_connections_filename)

print('Connecting PV population to Pyramidal cell')
PV_pop.connectCells([PV_pop.cells['PV0']['cell'].axon[1](1)]*3, [PYR_cell.cell.soma[0]], PV_density, PV_dist) # Adds self.connections to Population
print('\n***Assuming isopotential soma and perisomatic PV connections: all PV synapses are placed on soma(0.5)')

# ==============================================  Run Simulation & Plot Results  ==============================================
PYR_cell, PV_pop, VIP_pop = RunSim(PYR_cell, PV_pop, VIP_pop)

# ==============================================  Morphology Visualization  ==============================================


for i in range(n_PV):  
	PV_pop.moveCell(PV_pop.cells['PV{}'.format(i)]['cell'], i*300, -700, 0)  


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










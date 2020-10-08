import pandas as pd
import numpy as np
import _pickle as cPickle
import matplotlib.pyplot as plt
plt.ion()

import pdb, os, sys

from neuron import gui, h
from math import log
from tqdm import tqdm

from Population import Population
from Parameter_Initialization import * # Initialize parameters before anything else!
from plotting_functions import plotFRs

def get_GIDs(upload_from):

	chosen_GIDs = {}
	chosen_GIDs['pyr'] 			  = cPickle.load(open('{}/chosen_pyr.p'.format(upload_from), 'rb'))
	chosen_GIDs['PV'] 			  = cPickle.load(open('{}/chosen_PV.p'.format(upload_from), 'rb'))
	chosen_GIDs['SOM'] 			  = cPickle.load(open('{}/chosen_SOM_high_input.p'.format(upload_from), 'rb'))
	chosen_PV_n_contacts  		  = cPickle.load(open('{}/chosen_PV_n_contacts.p'.format(upload_from), 'rb'))

	thalamic_GIDs = {}
	thalamic_GIDs['to_pyr'] = cPickle.load(open('{}/connecting_gids_to_pyr.p'.format(upload_from), 'rb'))
	thalamic_GIDs['to_PV']  = cPickle.load(open('{}/connecting_gids_to_PV.p'.format(upload_from), 'rb'))
	thalamic_GIDs['to_SOM'] = cPickle.load(open('{}/connecting_gids_to_SOM_high_input.p'.format(upload_from), 'rb'))

	return chosen_GIDs, chosen_PV_n_contacts, thalamic_GIDs

def putspikes():
	
	for pop in POPs:
		for axon in pop.inputs[pop.cell_name]:
			stim_times = pop.inputs[pop.cell_name][axon]['stim_times']
			for netcon in pop.inputs[pop.cell_name][axon]['netcons']:
				for T in stim_times:
					netcon.event(T)

chosen_GIDs, chosen_PV_n_contacts, thalamic_GIDs = get_GIDs('GIDs_instantiations/pyr_72851_between_6666_9600')

PV_models = {
	'LBC_cNAC': {'template_path': 'EPFL_models/PV_models/L4_LBC_cNAC187_1', 'template_name': 'cNAC187_L4_LBC_990b7ac7df'},
	'LBC_dNAC': {'template_path': 'EPFL_models/PV_models/L4_LBC_dNAC222_1', 'template_name': 'dNAC222_L4_LBC_f6a71a338d'},
	'NBC_dNAC': {'template_path': 'EPFL_models/PV_models/L4_NBC_dNAC222_1', 'template_name': 'dNAC222_L4_NBC_aa36da75a3'},
	'NBC_cNAC': {'template_path': 'EPFL_models/PV_models/L4_NBC_cNAC187_1', 'template_name': 'cNAC187_L4_NBC_36cd91dc08'},
	'ChC_cNAC': {'template_path': 'EPFL_models/PV_models/L4_ChC_cNAC187_1', 'template_name': 'cNAC187_L4_ChC_22d43e8e41'},
	'ChC_dNAC': {'template_path': 'EPFL_models/PV_models/L4_ChC_dNAC222_1', 'template_name': 'dNAC222_L4_ChC_5e109d198d'}
	}

PV_gid = 66778
activations_filename = 'thalamocortical_Oren/SSA_spike_times/input6666_by_gid.p'
POPs = []
count = 0
for model in tqdm(PV_models):
	template_path 	= PV_models[model]['template_path']
	morph_path 		= '{}/morphology'.format(template_path)
	template_name 	= PV_models[model]['template_name']

	POPs.append(Population(model, morph_path, template_path, template_name))
	POPs[-1].addCell()

	cell_name = list(POPs[-1].cells.keys())[0]
	POPs[-1].cell_name = cell_name
	POPs[-1].addInput(cell_name, where_synapses=['basal_dendrites', 'apical_dendrites'], record_syns=False, weight=PV_input_weight, thalamic_activations_filename=activations_filename, connecting_gids=thalamic_GIDs['to_PV'][PV_gid])

	POPs[-1].moveCell(POPs[-1].cells[POPs[-1].cell_name]['cell'], (count*350)-(100*(len(PV_models)+1)), -500, 0) 
	count += 1
	

print('Running simulation')
events = h.FInitializeHandler(putspikes)
h.tstop = 10000
t = h.Vector().record(h._ref_t)
h.run()
os.system("osascript -e \'display notification \"Simulation finished\" with title \"Simulation Finished\" sound name \"Submarine\"\'")

stim_times = cPickle.load(open('thalamocortical_Oren/SSA_spike_times/stim_times.p', 'rb'))[6666]
stim_times = [i[0] for i in stim_times]
window = 10
axes_h = None
C = iter(['skyblue', 'orange', 'crimson', 'gray', 'xkcd:magenta'])
for pop in POPs:
	temp_cell = list(pop.cells.keys())[0]
	soma_v = pop.cells[temp_cell]['soma_v']
	axes_h = plotFRs(0, stim_times, soma_v, t, tstop=h.tstop, window=window, which_cell=temp_cell[:-1], axes_h=axes_h, color=next(C))
















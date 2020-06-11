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
# from Parameter_Initialization import * # Initialize parameters before anything else!
from plotting_functions import plotThalamicResponses, Wehr_Zador_fromData, PlotSomas_fromData, plotFRs

assert os.getcwd().split('/')[-1] == 'Circuit', 'Wrong directory'
if len(sys.argv)>1:
	job_id = sys.argv[1]
else:
	job_id = '8626253'

sys.path.append('cluster_downloadables/{}/'.format(job_id))
exec('from Parameter_Initialization_{} import *'.format(job_id))

# ============================================  Define Functions & Constants  ============================================

def load_Data(job_id, load_path='cluster_downloadables'):

	Pyr_data, PV_data, SOM_data = None, None, None
	load_path = load_path + '/{}'.format(job_id)
	job_files = [i for i in os.listdir(load_path) if i.startswith(job_id)]
	assert len(job_files)<=3, 'Wrond number of cell files to load'

	if any([i for i in job_files if 'Pyr' in i]):
		Pyr_data = cPickle.load(open('{}/{}'.format(load_path, [i for i in job_files if 'Pyr' in i][0]), 'rb'))
	if any([i for i in job_files if 'PV' in i]):
		PV_data  = cPickle.load(open('{}/{}'.format(load_path, [i for i in job_files if 'PV' in i][0]), 'rb'))
	if any([i for i in job_files if 'SOM' in i]):
		SOM_data = cPickle.load(open('{}/{}'.format(load_path, [i for i in job_files if 'SOM' in i][0]), 'rb'))

	examp_job = job_files[0]
	tstop = int((examp_job.split('_tstop_')[1]).split('_')[0])
	input_filename = examp_job.split(str(tstop)+'_')[1]

	t = Pyr_data['t']

	return t, Pyr_data, PV_data, SOM_data, tstop, input_filename

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

# ============================================  Load Data from Cluster Dumps  ============================================
t, Pyr_data, PV_data, SOM_data, tstop, input_filename = load_Data(job_id)

stand_freq = freq1*(str(freq1) in input_filename) + freq2*(str(freq2) in input_filename)
dev_freq = freq1*(str(freq1) not in input_filename) + freq2*(str(freq2) not in input_filename)

stim_times = cPickle.load(open(filenames['stim_times'], 'rb'))[stand_freq]
stim_times_standard = [i[0] for i in stim_times if i[1]==stand_freq]
stim_times_deviant = [i[0] for i in stim_times if i[1]==dev_freq]

# ===============================================  Choose GIDs  ===============================================
print('\n========== Choosing GIDs from BlueBrain Database (pyramidal between {}Hz and {}Hz) =========='.format(freq1, freq2))

upload_from = 'GIDs_instantiations/pyr_72851_between_6666_9600'
chosen_GIDs, chosen_PV_n_contacts, thalamic_GIDs = get_GIDs(upload_from)

# ==============================================  Stimulus Analysis  ==============================================
# stimuli[stand_freq] = Stimulus(stand_freq, dev_freq, filenames['stim_times'], filenames['thalamic_activations_6666'], axon_gids=[i[0] for i in thalamic_GIDs['to_pyr']])

run_plot_function = False
if run_plot_function:
	stimuli[dev_freq] = Stimulus(dev_freq, stand_freq, filenames['stim_times'], filenames['thalamic_activations_9600'], axon_gids=[i[0] for i in thalamic_GIDs['to_pyr']])
	stim_ax = plotThalamicResponses(stimuli, stand_freq, dev_freq, thalamic_locations, run_function=True)

# ============================================== Plot example responses ==============================================

if PV_data:
	Wehr_Zador_fromData(PV_data, 'PV0', stim_times, dt=h.dt, tstop=tstop, t=t, exc_weight=PV_input_weight, standard_freq=stand_freq, spike_threshold=spike_threshold)
if Pyr_data:
	Wehr_Zador_fromData(Pyr_data, 'Pyr0', stim_times, dt=h.dt, tstop=tstop, t=t, exc_weight=Pyr_input_weight, standard_freq=stand_freq, spike_threshold=spike_threshold)
if SOM_data:
	Wehr_Zador_fromData(SOM_data, 'SOM0', stim_times, dt=h.dt, tstop=tstop, t=t, exc_weight=SOM_input_weight, standard_freq=stand_freq, spike_threshold=spike_threshold)
		

FR_ax = None
C = iter(['skyblue', 'orange', 'crimson'])
for data in [PV_data, SOM_data, Pyr_data]:
	if data:
		temp_cell = list(data['cells'].keys())[0]
		soma_v = data['cells'][temp_cell]['soma_v']
		temp_which_cell = temp_cell.split([i for i in temp_cell if i.isdigit()][0])[0]
		
		FR_ax = plotFRs([i[0] for i in stim_times], soma_v, t, tstop=tstop, window=6, which_cell=temp_which_cell, axes_h=FR_ax, color=next(C))

somas_dict = {}
for n, d in [['Pyr0', Pyr_data], ['PV0', PV_data], ['SOM0', SOM_data]]:
	if d:
		somas_dict[n] = d
soma_ax = PlotSomas_fromData(somas_dict, t, stim_times_standard=stim_times_standard, standard_freq=stand_freq, stim_times_deviant=stim_times_deviant, deviant_freq=dev_freq, tstop=tstop)








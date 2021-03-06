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
from plotting_functions import plotThalamicResponses, Wehr_Zador_fromData, PlotSomas_fromData, plotFRs, PlotSomaSubplots_fromData

# FRom when I didn't have the parameter record_channel
try:
	record_channel
except:
	record_channel = False

assert os.getcwd().split('/')[-1] == 'Circuit', 'Wrong directory'
if len(sys.argv)>1:
	job_id = sys.argv[1]
else:
	job_id = input('Job ID? ')

sys.path.append('cluster_downloadables/{}/'.format(job_id))
thalamic_params, Pyr_input_params, recording_dt = None, None, None
exec('from Parameter_Initialization_{} import *'.format(job_id))

# ============================================  Define Functions & Constants  ============================================
def adjustment_OldInputs(thalamic_params, Pyr_input_params, recording_dt):
	print(recording_dt)

	if not thalamic_params:
		print('No thalamic parameters given! Please input manually')
		thalamic_params = {
			'type': input('single or pair? '),
			'ITI': int(input('ITI? ')),
			'IPI': int(input('IPI? ')),
			'freq': input('Frequency? (__Hz or BroadBand): ')
		}

	if not Pyr_input_params:
		print('No Pyramidal input parameters given! Please input manually')
		Pyr_input_params = {
			'Dep': int(input('Dep? ')),
			'Fac': int(input('Fac? ')),
			'Use': int(input('Use? '))
		}

	if not recording_dt:
		recording_dt = h.dt

	return thalamic_params, Pyr_input_params, recording_dt

def get_thalamic_input_filename(thalamic_params):
	fname_check = lambda fname, fchecks: all([f in fname for f in fchecks])

	if thalamic_params['type'] == 'pairs':	
		path_ = 'thalamocortical_Oren/CreateThalamicInputs/test_times/tone_pairs'	
		activated_filename = [i for i in os.listdir(path_) if fname_check(i, [thalamic_params['freq'], '.p', 'ITI_%s'%thalamic_params['ITI'], 'IPI_%s'%thalamic_params['IPI']])]

	elif thalamic_params['type'] == 'single':
		path_ = 'thalamocortical_Oren/CreateThalamicInputs/test_times/single_tone'
		activated_filename = [i for i in os.listdir(path_) if fname_check(i, [thalamic_params['freq'], '.p', 'IPI_%s'%thalamic_params['IPI']])]

	assert len(activated_filename) == 1, ['Nonspecific thalamic times filename!', pdb.set_trace()] 
	activated_filename = '{}/{}'.format(path_, activated_filename[0])
	
	filenames['thalamic_activations'] = activated_filename

	print('Activating input file: \"{}\"'.format(activated_filename))

	return activated_filename

def get_thalamic_recordings_path(activated_filename):
	pair_or_single = 	'Pairs'				*('pair' in activated_filename and 'BroadBand' not in activated_filename) + \
						'Pairs_BroadBand'	*('BroadBand' in activated_filename) + \
						'Single'			*('single' in activated_filename)

	assert pair_or_single!='', 'Ambiguous thalamic input filename! (pair or single?)'

	thalamic_path = 'thalamic_synapse_recordings/{}/Dep_{}_Fac_{}_Use_{}/ITI_{}_IPI_{}/Pyr_{}_PV_{}_SOM_{}'.format(pair_or_single,
																					Pyr_input_params['Dep'],
																					Pyr_input_params['Fac'],
																					Pyr_input_params['Use'],
																					thalamic_params['ITI'],
																					thalamic_params['IPI'],
																					Pyr_input_weight,
																					PV_input_weight,
																					SOM_input_weight)
	return thalamic_path

def load_Data(job_id, load_path='cluster_downloadables', thalamic_path=''):

	Pyr_data, PV_data, SOM_data, axon_gsk, soma_gsk = None, None, None, None, None
	thalamic_load_path = load_path

	thalamic_files = os.listdir(thalamic_load_path + '/' + thalamic_path)
	load_path = load_path + '/{}'.format(job_id)
	job_files = [i for i in os.listdir(load_path) if i.startswith(job_id)]
	assert len(job_files)<=3, 'Wrong number of cell files to load'

	examp_job = job_files[0]
	tstop = int((examp_job.split('_tstop_')[1]).split('_')[0])
	input_filename = examp_job.split(str(tstop)+'_')[1]

	t = np.arange(0, tstop+recording_dt, recording_dt)
	

	if any([i for i in job_files if 'Pyr' in i]):
		assert len([i for i in job_files if 'Pyr' in i]) == 1, 'More than one Pyr thalamic recording!'
		Pyr_data = cPickle.load(open('{}/{}'.format(load_path, [i for i in job_files if 'Pyr' in i][0]), 'rb'))
		
		# Old version files of Popuylation dump function
		if 'cells' in Pyr_data.keys():
			if len(Pyr_data.keys())>1:
				raise Exception('Unrecognized Pyr file type')
			else:
				Pyr_data = Pyr_data['cells']

		Pyr_thalamic = cPickle.load(open('{}/{}/{}'.format(thalamic_load_path, thalamic_path, [i for i in thalamic_files if 'Pyr' in i][0]), 'rb'))
		if 'cells' in Pyr_thalamic:
			Pyr_thalamic = Pyr_thalamic['cells']

		for Pyr_cell in Pyr_data:
			Pyr_data[Pyr_cell]['inputs']['g_AMPA'] = Pyr_thalamic[Pyr_cell]['inputs']['g_AMPA']
			Pyr_data[Pyr_cell]['inputs']['g_NMDA'] = Pyr_thalamic[Pyr_cell]['inputs']['g_NMDA']
			Pyr_data[Pyr_cell]['inputs']['i_AMPA'] = Pyr_thalamic[Pyr_cell]['inputs']['i_AMPA']
			Pyr_data[Pyr_cell]['inputs']['i_NMDA'] = Pyr_thalamic[Pyr_cell]['inputs']['i_NMDA']

		if len(Pyr_data['Pyr0']['soma_v'])<len(t): t = t[:-1]

	if any([i for i in job_files if 'PV' in i]):
		assert len([i for i in job_files if 'PV' in i]) == 1, 'More than one PV thalamic recording!'
		PV_data  = cPickle.load(open('{}/{}'.format(load_path, [i for i in job_files if 'PV' in i][0]), 'rb'))

		# Old version files of Popuylation dump function
		if 'cells' in PV_data.keys():
			if len(PV_data.keys())>1:
				raise Exception('Unrecognized PV file type')
			else:
				PV_data = PV_data['cells']

		PV_thalamic = cPickle.load(open('{}/{}/{}'.format(thalamic_load_path, thalamic_path, [i for i in thalamic_files if 'PV' in i][0]), 'rb'))
		if 'cells' in PV_thalamic:
			PV_thalamic = PV_thalamic['cells']

		for PV_cell in PV_data:
			PV_data[PV_cell]['inputs']['g_AMPA'] = PV_thalamic[PV_cell]['inputs']['g_AMPA']
			PV_data[PV_cell]['inputs']['g_NMDA'] = PV_thalamic[PV_cell]['inputs']['g_NMDA']
			PV_data[PV_cell]['inputs']['i_AMPA'] = PV_thalamic[PV_cell]['inputs']['i_AMPA']
			PV_data[PV_cell]['inputs']['i_NMDA'] = PV_thalamic[PV_cell]['inputs']['i_NMDA']

		if len(PV_data['PV0']['soma_v'])<len(t): t = t[:-1]


	if any([i for i in job_files if 'SOM' in i]):
		assert len([i for i in job_files if 'SOM' in i]) == 1, 'More than one SOM thalamic recording!'
		SOM_data = cPickle.load(open('{}/{}'.format(load_path, [i for i in job_files if 'SOM' in i][0]), 'rb'))

		# Old version files of Popuylation dump function
		if 'cells' in SOM_data.keys():
			if len(SOM_data.keys())>1:
				raise Exception('Unrecognized SOM file type')
			else:
				SOM_data = SOM_data['cells']

		SOM_thalamic = cPickle.load(open('{}/{}/{}'.format(thalamic_load_path, thalamic_path, [i for i in thalamic_files if 'SOM' in i][0]), 'rb'))
		if 'cells' in SOM_thalamic:
			SOM_thalamic = SOM_thalamic['cells']
		
		for SOM_cell in SOM_data:
			SOM_data[SOM_cell]['inputs']['g_AMPA'] = SOM_thalamic[SOM_cell]['inputs']['g_AMPA']
			SOM_data[SOM_cell]['inputs']['g_NMDA'] = SOM_thalamic[SOM_cell]['inputs']['g_NMDA']
			SOM_data[SOM_cell]['inputs']['i_AMPA'] = SOM_thalamic[SOM_cell]['inputs']['i_AMPA']
			SOM_data[SOM_cell]['inputs']['i_NMDA'] = SOM_thalamic[SOM_cell]['inputs']['i_NMDA']

		if len(SOM_data['SOM0']['soma_v'])<len(t): t = t[:-1]

	if record_channel:
		axon_gsk = cPickle.load(open('{}/{}'.format(load_path, 'axon_sk.p'), 'rb'))
		soma_gsk = cPickle.load(open('{}/{}'.format(load_path, 'somatic_sk.p'), 'rb'))


	return t, Pyr_data, PV_data, SOM_data, tstop, input_filename, axon_gsk, soma_gsk

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

# ============================================  Load Data from Cluster Dumps  ============================================

thalamic_params, Pyr_input_params, recording_dt = adjustment_OldInputs(thalamic_params, Pyr_input_params, recording_dt)
activated_filename 	= get_thalamic_input_filename(thalamic_params)
thalamic_path 		= get_thalamic_recordings_path(activated_filename)

t, Pyr_data, PV_data, SOM_data, tstop, input_filename, axon_gsk, soma_gsk = load_Data(job_id, thalamic_path=thalamic_path)

stand_freq = freq1*(str(freq1) in input_filename) + freq2*(str(freq2) in input_filename)
dev_freq = freq1*(str(freq1) not in input_filename) + freq2*(str(freq2) not in input_filename)

# stim_times = cPickle.load(open(filenames['stim_times'], 'rb'))[stand_freq]
stim_times = [[i, stand_freq] for i in range(2000, tstop, thalamic_params['IPI'])]
stim_times_standard = [i[0] for i in stim_times if i[1]==stand_freq]
stim_times_deviant = [i[0] for i in stim_times if i[1]==dev_freq]
stim_times = [i for i in stim_times if i[0]<=tstop]
# ===============================================  Choose GIDs  ===============================================
print('\n========== Choosing GIDs from BlueBrain Database (pyramidal between {}Hz and {}Hz) =========='.format(freq1, freq2))

upload_from = 'GIDs_instantiations/pyr_72851_between_6666_9600'
chosen_GIDs, chosen_PV_n_contacts, thalamic_GIDs = get_GIDs(upload_from)

# ============================================== Plot example responses ==============================================
if PV_data:
	Wehr_Zador_fromData(job_id, PV_data, 'PV0', stim_times, take_after=300, dt=recording_dt, tstop=tstop, t=t, exc_weight=PV_input_weight, standard_freq=stand_freq, spike_threshold=spike_threshold)
if Pyr_data:
	Wehr_Zador_fromData(job_id, Pyr_data, 'Pyr0', stim_times, stim_params={'type':thalamic_params['type'], 'ITI':thalamic_params['ITI'], 'mark_second': True}, take_after=300, dt=recording_dt, tstop=tstop, t=t, exc_weight=Pyr_input_weight, standard_freq=stand_freq, spike_threshold=spike_threshold)
if SOM_data:
	Wehr_Zador_fromData(job_id, SOM_data, 'SOM0', stim_times, take_after=300, dt=recording_dt, tstop=tstop, t=t, exc_weight=SOM_input_weight, standard_freq=stand_freq, spike_threshold=spike_threshold)
		

FR_ax = None
C = iter(['skyblue', 'orange', 'crimson'])
# C = iter(['xkcd:fuchsia', 'xkcd:azure', 'grey'])
for data in [PV_data, SOM_data, Pyr_data]:
	if data:
		temp_cell = list(data.keys())[0]
		soma_v = data[temp_cell]['soma_v']
		temp_which_cell = temp_cell.split([i for i in temp_cell if i.isdigit()][0])[0]
		
		FR_ax = plotFRs(job_id, [i[0] for i in stim_times], soma_v, t, take_after=300, tstop=tstop, window=6, which_cell=temp_which_cell, axes_h=FR_ax, color=next(C))

somas_dict = {}
for n, d in [['Pyr0', Pyr_data], ['PV0', PV_data], ['SOM0', SOM_data]]:
	if d:
		somas_dict[n] = d
soma_ax = PlotSomas_fromData(somas_dict, t, stim_times_standard=stim_times_standard, standard_freq=stand_freq, stim_times_deviant=stim_times_deviant, deviant_freq=dev_freq, tstop=tstop)
soma_subs_ax = PlotSomaSubplots_fromData(somas_dict, t, stim_times_standard=stim_times_standard, standard_freq=stand_freq, stim_times_deviant=stim_times_deviant, deviant_freq=dev_freq, tstop=tstop)


def plot_SplitInhConductance(pre_data, pre_cell, post_datas, post_cells, stim_times, window=[0, 40], sample_interval=[10, 240], pre_input_delay=0, which_success=''):
	window_before = window[0]
	window_after = window[1]
	take_before = sample_interval[0]
	take_after = sample_interval[1]
	pre_cell_name = "".join([i for i in pre_cell if not i.isdigit()])

	cut_vec = lambda vec, start_idx, end_idx: [vec[i] for i in range(start_idx, end_idx)]
	
	pre_soma_v = pre_data[pre_cell]['soma_v']
	# split_inputs_dict = {'success': {'g_GABA': [], 'i_GABA':[]}, 'fail': {'g_GABA': [], 'i_GABA':[]}}
	split_inputs_dict = {}

	times = [i[0] for i in stim_times]

	if sample_interval[1]+times[-1]>t[-1]: times = times[:-1] # If take_after goes beyond recording, discard last stimulus
	
	for d in range(len(post_datas)):
		data = post_datas[d]
		cell = post_cells[d]
		split_inputs_dict[cell] = {'success': {'g_GABA': [], 'i_GABA':[]}, 'fail': {'g_GABA': [], 'i_GABA':[]}}
	
		for T in times:
			idx1 = int((T-window_before)/h.dt)
			idx2 = int((T+window_after)/h.dt)
			
			OK = 0
			while not OK:				
				if which_success == 'Presynaptic':
					v_vec = cut_vec(pre_soma_v, idx1, idx2)
					OK = 1

				elif which_success == 'Postsynaptic':
					v_vec = cut_vec(data[cell]['soma_v'], idx1, idx2)
					OK = 1

				else:
					which_success = input('Split by success of Presynaptic or Postsynaptic spike?')

			is_spiking = any([v_vec[i] for i in range(len(v_vec)-1) if (v_vec[i]>=spike_threshold) 
																	and (v_vec[i]>v_vec[i+1]) 
																	and (v_vec[i]>v_vec[i-1])])
			
			idx1_take = int((T-take_before)/h.dt)
			idx2_take = int((T+take_after)/h.dt)

			g_GABA_vec = cut_vec(data[cell]['inputs']['g_GABA'][pre_cell_name], idx1_take, idx2_take)
			i_GABA_vec = cut_vec(data[cell]['inputs']['i_GABA'][pre_cell_name], idx1_take, idx2_take)

			if is_spiking:
				split_inputs_dict[cell]['success']['g_GABA'].append(g_GABA_vec)
				split_inputs_dict[cell]['success']['i_GABA'].append(i_GABA_vec)
			else:
				split_inputs_dict[cell]['fail']['g_GABA'].append(g_GABA_vec)
				split_inputs_dict[cell]['fail']['i_GABA'].append(i_GABA_vec)

		mean_success_g = np.mean(split_inputs_dict[cell]['success']['g_GABA'], axis=0)
		mean_fail_g = np.mean(split_inputs_dict[cell]['fail']['g_GABA'], axis=0)
		mean_success_i = np.mean(split_inputs_dict[cell]['success']['i_GABA'], axis=0)
		mean_fail_i = np.mean(split_inputs_dict[cell]['fail']['i_GABA'], axis=0)

		t_vec = t[:len(g_GABA_vec)]

		f, ax = plt.subplots()
		f.subplots_adjust(hspace=0.34, bottom=0.08, top=0.9, left=0.1, right=0.93)  
		f.suptitle('{} Inputs to {} For {} Spike Success & Failure'.format(pre_cell_name, "".join([i for i in cell if not i.isdigit()]), which_success))


		ax.plot(t_vec, [i*1000 for i in mean_success_g], label='Success')
		ax.plot(t_vec, [i*1000 for i in mean_fail_g], label='Fail')
		ax.set_ylabel('G (nS)')
		ax.set_xlabel('T (ms)')
		ax.axvline(take_before, LineStyle='--', color='gray', LineWidth=1.5, label='Stimulus Presentation')
		ax.plot(pre_input_delay+take_before, 0, 'go', label='Input to Presynaptic')   
		ax.legend()

		x_ticks = np.append(ax.get_xticks(), pre_input_delay+take_before) 
		ax.set_xticks(x_ticks)  
		ax.set_xlim([0, sum(sample_interval)])

		plt.figure()
		for i in range(len(split_inputs_dict[cell]['success']['g_GABA'])):
			if i==0:
				plt.plot(t_vec, split_inputs_dict[cell]['success']['g_GABA'][i], 'orange', label='Success')
			else:
				plt.plot(t_vec, split_inputs_dict[cell]['success']['g_GABA'][i], 'orange')
		for i in range(len(split_inputs_dict[cell]['fail']['g_GABA'])):
			if i==0:
				plt.plot(t_vec, split_inputs_dict[cell]['fail']['g_GABA'][i], 'blue', label='Fail')
			else:
				plt.plot(t_vec, split_inputs_dict[cell]['fail']['g_GABA'][i], 'blue')

	return split_inputs_dict

# if 'g_GABA' in PV_data['PV0']['inputs']:
# 	if any([hasattr(i, '__len__') for i in PV_data['PV0']['inputs']['g_GABA'].values()]):
# 		split_inputs_dict = plot_SplitInhConductance(SOM_data, 'SOM0', [PV_data], ['PV0'], stim_times, which_success='Presynaptic', window=[0, SOM_input_delay+40], pre_input_delay=SOM_input_delay)

# if 'g_GABA' in Pyr_data['Pyr0']['inputs']:
# 	if any([hasattr(i, '__len__') for i in Pyr_data['Pyr0']['inputs']['g_GABA'].values()]):
# 		split_inputs_dict = plot_SplitInhConductance(PV_data, 'PV0', [Pyr_data], ['Pyr0'], stim_times, which_success='Postsynaptic', window=[0, PV_input_delay+40], pre_input_delay=PV_input_delay)

if record_channel:

	f, ax1 = plt.subplots(figsize=(15, 7.5))
	f.suptitle('Pyramidal Somatic Voltage and Somatic & Axonal $g_{SK}$')

	ax1.plot(t, axon_gsk, 'xkcd:aquamarine', label='Axonal $g_{SK}$')
	ax1.plot(t, soma_gsk, 'xkcd:azure', label='Somatic $g_{SK}$')
	ax1.set_xlabel('T (ms)')
	ax1.set_ylabel('G (S/$cm^2$)')
	ax1.set_ylim([-0.00008, 0.004]) 
	ax1.legend(loc='upper left')

	ax2 = ax1.twinx()
	ax2.plot(t, Pyr_data['Pyr0']['soma_v'], label='Pyr Somatic V')
	ax2.set_ylabel('V (mV)')
	ax2.set_ylim([-100, 40])
	ax2.set_xlim([0, tstop])
	ax2.legend(loc='upper right')

	stim_times = cPickle.load(open(filenames['stim_times'], 'rb'))[stand_freq]
	stim_times = [i[0] for i in stim_times]
	for T in stim_times: 
		if T <= tstop:
			ax2.axvline(T, LineStyle='--',color='gray',LineWidth=1) 









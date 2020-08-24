import pandas as pd
import numpy as np
import _pickle as cPickle
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import time as time_module
plt.ion()
from tqdm import tqdm

import pdb, os, sys
from neuron import gui, h

from Population import Population
from Stimulus import Stimulus
from Parameter_Initialization import * # Initialize parameters before anything else!
'''
Conditions:
- No inhibition
- Weak inhibition all (0.3)
- Strong inhibition all (0.8)
- Only disinhibition (SOM disconnected from Pyr)
- No disinhibition (SOM disconnected from PV)
'''

results = cPickle.load(open('quantify_disinhibition_results_1_2.p', 'rb'))
tstop 	= results['parameters']['tstop']
dt 		= results['parameters']['dt']
t 		= np.arange(0, tstop, dt)

stim_times = cPickle.load(open(filenames['stim_times'],'rb'))[6666]
stim_times = [i[0] for i in stim_times if i[0]<tstop]


def MakeHistSpikes(results, window, hists=None, as_percent=False):
	def GetHists(results, window):

		conds = [i for i in results if i!='parameters']
		hists = {}
		for C in conds:
			v = results[C]
			spike_times = [t[i] for i in range(1, len(v)-1) if (v[i]>v[i-1]) and (v[i]>v[i+1]) and (v[i]>spike_threshold)]
			num_spikes = []

			for T in stim_times:

				idx1 = int((T-window[0])/dt)
				idx2 = int((T+window[1])/dt)

				T1 = list(t)[idx1]
				T2 = list(t)[idx2]

				window_spikes = [i-T for i in spike_times if (i>=T1) and (i<=T2)]
				num_spikes.append(len(window_spikes))

				hists[C] = num_spikes

		return hists

	def PlotHists(hists, window):
		#green strong I, purple no dis, orange weak i, red only dis, blue no i
		def analyze_hists(ind_hist):
			#  std is sqrt(n*p*(p-1))

			H, B = np.histogram(ind_hist, bins=range(MAX+2))

			n = len(ind_hist)
			p = ind_hist.count(0) / n
			q = 1-p
			binomial_std = np.sqrt(n * p * q)

			return H, B, binomial_std


		MAX =  max([max(i) for i in hists.values()])
		colors = {'no_I': (0.12156862745098039, 0.4666666666666667, 0.7058823529411765), 
				  'weak_I': (1.0, 0.4980392156862745, 0.054901960784313725), 
				  'strong_I': (0.17254901960784313, 0.6274509803921569, 0.17254901960784313),
				  'only_dis': (0.8392156862745098, 0.15294117647058825, 0.1568627450980392),
				  'no_dis': (0.5803921568627451, 0.403921568627451, 0.7411764705882353)}

		f, h_ax = plt.subplots()

		SumLists = lambda l1, l2: [l1[i]+l2[i] for i in range(len(l1))]
		bottom = [0]*(MAX+1)
		conds = ['strong_I', 'no_dis', 'weak_I', 'only_dis', 'no_I']
		cols_for_cond = len(conds) + 1
		
		for c in conds:
			H, B, binomial_std = analyze_hists(hists[c])

			if as_percent:
				transformation_factor = 100/sum(H) # This should happen after binomial_std calculation
				H = [i*transformation_factor for i in H]
				binomial_std = np.sqrt((binomial_std**2) * (transformation_factor**2)) # Y = mX => Var(Y) = m^2 * Var(X)
			shifted_B = [(i*cols_for_cond + conds.index(c)) for i in B[:-1]]			

			h_ax.bar(shifted_B, H, align='edge', width=np.diff(B)[0], alpha=0.5, label=c, bottom=bottom, color=colors[c], yerr=binomial_std, ecolor=(0,0,0,0.5))

			for i in range(len(H)):

				h_ax.text(shifted_B[i], H[i], '%.1f'%H[i])

			# bottom = SumLists(bottom, H)

		labels = ['0', '1', '2']
		h_ax.set_xticks(np.arange(len(conds)/2, (cols_for_cond*len(labels))+len(conds)/2, cols_for_cond))
		h_ax.set_xticklabels(labels)
		h_ax.legend(loc='upper right')
		for sep in range(len(labels)):
			h_ax.axvline(0.5+len(conds)+sep*cols_for_cond, color='k')
		
		plt.suptitle('Number of Spikes in {} Window Around Auditory Simulus, For Different Inh. Conditions (stimulation length: {}ms)'.format(window[1]-window[0], results['parameters']['tstop']), size=13) 
		plt.title(r'Error bar = $\sqrt{n\_windows \cdot p \cdot q}$, Where p = $\frac{n\_0\_spikes}{n\_windows}}$, q = (1 - p)') 
		h_ax.set_xlabel('no. spikes')

		if as_percent:
			h_ax.set_ylabel('Percent')
			h_ax.yaxis.set_major_formatter(mtick.PercentFormatter()) 

		else:
			h_ax.set_ylabel('Count')




		return h_ax

	if not hists:
		hists = GetHists(results, window)
	h_ax = PlotHists(hists, window)


	return h_ax, hists

hists = None
h_ax, hists = MakeHistSpikes(results, [0, 100], hists, as_percent=True)

# BINS = range(20, 70, 2)

# h_ax = PlotSpikeProb(results['no_I'], 'No Inhibition', [0, 100], bins=BINS)

# h_ax = PlotSpikeProb(results['weak_I'], 'Weak Inhibition ({})'.format(results['parameters']['weak_weight']), [0, 100], h_ax2=h_ax, bins=BINS)

# h_ax = PlotSpikeProb(results['strong_I'], 'Strong Inhibition ({})'.format(results['parameters']['strong_weight']), [0, 100], h_ax2=h_ax, bins=BINS)

# h_ax = PlotSpikeProb(results['only_dis'], 'Only Dis-Inhibition', [0, 100], h_ax2=h_ax, bins=BINS)

# h_ax = PlotSpikeProb(results['no_dis'], 'No Dis-Inhibition', [0, 100], h_ax2=h_ax, bins=BINS)


def load_Data(which_jobs, job_id, load_path='cluster_downloadables', thalamic_path=''):

	Pyr_data, PV_data, SOM_data = None, None, None
	thalamic_load_path = load_path

	thalamic_files = os.listdir(thalamic_load_path + '/' + thalamic_path)
	load_path = load_path + '/{}'.format(job_id)
	job_files = [i for i in os.listdir(load_path) if i.startswith(job_id)]
	assert len(job_files)<=3, 'Wrond number of cell files to load'

	if 'Pyr' in which_jobs:
		Pyr_data = cPickle.load(open('{}/{}'.format(load_path, [i for i in job_files if 'Pyr' in i][0]), 'rb'))
		
		# Old version files of Popuylation dump function
		if 'cells' in Pyr_data.keys():
			if len(Pyr_data.keys())>1:
				raise Exception('Unrecognized Pyr file type')
			else:
				Pyr_data = Pyr_data['cells']

		Pyr_thalamic = cPickle.load(open('{}/{}/{}'.format(thalamic_load_path, thalamic_path, [i for i in thalamic_files if 'Pyr' in i][0]), 'rb'))
		
		for Pyr_cell in Pyr_data:
			Pyr_data[Pyr_cell]['inputs']['g_AMPA'] = Pyr_thalamic['cells'][Pyr_cell]['inputs']['g_AMPA']
			Pyr_data[Pyr_cell]['inputs']['g_NMDA'] = Pyr_thalamic['cells'][Pyr_cell]['inputs']['g_NMDA']
			Pyr_data[Pyr_cell]['inputs']['i_AMPA'] = Pyr_thalamic['cells'][Pyr_cell]['inputs']['i_AMPA']
			Pyr_data[Pyr_cell]['inputs']['i_NMDA'] = Pyr_thalamic['cells'][Pyr_cell]['inputs']['i_NMDA']

	if 'PV' in which_jobs:
		PV_data  = cPickle.load(open('{}/{}'.format(load_path, [i for i in job_files if 'PV' in i][0]), 'rb'))

		# Old version files of Popuylation dump function
		if 'cells' in PV_data.keys():
			if len(PV_data.keys())>1:
				raise Exception('Unrecognized PV file type')
			else:
				PV_data = PV_data['cells']

		PV_thalamic = cPickle.load(open('{}/{}/{}'.format(thalamic_load_path, thalamic_path, [i for i in thalamic_files if 'PV' in i][0]), 'rb'))

		for PV_cell in PV_data:
			PV_data[PV_cell]['inputs']['g_AMPA'] = PV_thalamic['cells'][PV_cell]['inputs']['g_AMPA']
			PV_data[PV_cell]['inputs']['g_NMDA'] = PV_thalamic['cells'][PV_cell]['inputs']['g_NMDA']
			PV_data[PV_cell]['inputs']['i_AMPA'] = PV_thalamic['cells'][PV_cell]['inputs']['i_AMPA']
			PV_data[PV_cell]['inputs']['i_NMDA'] = PV_thalamic['cells'][PV_cell]['inputs']['i_NMDA']


	if 'SOM' in which_jobs:
		SOM_data = cPickle.load(open('{}/{}'.format(load_path, [i for i in job_files if 'SOM' in i][0]), 'rb'))

		# Old version files of Popuylation dump function
		if 'cells' in SOM_data.keys():
			if len(SOM_data.keys())>1:
				raise Exception('Unrecognized SOM file type')
			else:
				SOM_data = SOM_data['cells']

		SOM_thalamic = cPickle.load(open('{}/{}/{}'.format(thalamic_load_path, thalamic_path, [i for i in thalamic_files if 'SOM' in i][0]), 'rb'))
		
		for SOM_cell in SOM_data:
			SOM_data[SOM_cell]['inputs']['g_AMPA'] = SOM_thalamic['cells'][SOM_cell]['inputs']['g_AMPA']
			SOM_data[SOM_cell]['inputs']['g_NMDA'] = SOM_thalamic['cells'][SOM_cell]['inputs']['g_NMDA']
			SOM_data[SOM_cell]['inputs']['i_AMPA'] = SOM_thalamic['cells'][SOM_cell]['inputs']['i_AMPA']
			SOM_data[SOM_cell]['inputs']['i_NMDA'] = SOM_thalamic['cells'][SOM_cell]['inputs']['i_NMDA']

	examp_job = job_files[0]
	tstop = int((examp_job.split('_tstop_')[1]).split('_')[0])
	input_filename = examp_job.split(str(tstop)+'_')[1]

	t = np.arange(0, tstop+h.dt, h.dt)

	return Pyr_data, PV_data, SOM_data, tstop


analyze_2_jobs = False
if analyze_2_jobs:
	Pyr_data_with_disinhibition_job = '9194195' #OLD '9002620'
	Pyr_data_without_disinhibition_job = '9194196' #OLD '9004757'
	def plot_G_and_I(job_with_disinhibition, job_without_disinhibition, window=[-20, 150]):
		'''
		Take data from 2 jobs, and compare the conductances and currents. Used originally for comparing PV current and conductance on Pyramidal cell, 
		for disinhibition (SOM connected to PV) and no disinhibition (SOM disconnected from PV)
		'''

		Pyr_data_with, __, __, tstop = load_Data(['Pyr'], job_with_disinhibition, thalamic_path='thalamic_synapse_recordings/Pyr_0.5_PV_0.4_SOM_0.4')
		Pyr_data_without, __, __, tstop = load_Data(['Pyr'], job_without_disinhibition, thalamic_path='thalamic_synapse_recordings/Pyr_0.5_PV_0.4_SOM_0.4')



		i_PV_with = Pyr_data_with['Pyr0']['inputs']['i_GABA']['PV'] 
		i_PV_without = Pyr_data_without['Pyr0']['inputs']['i_GABA']['PV'] 
		g_PV_with = Pyr_data_with['Pyr0']['inputs']['g_GABA']['PV'] 
		g_PV_without = Pyr_data_without['Pyr0']['inputs']['g_GABA']['PV'] 

		mean_g_with, mean_g_without, mean_i_with, mean_i_without = [], [] , [], []

		for T in stim_times: 
			idx1 = int((T+window[0])/dt) 
			idx2 = int((T+window[1])/dt) 
			mean_g_with.append(g_PV_with[idx1:idx2]) 
			mean_g_without.append(g_PV_without[idx1:idx2]) 
			mean_i_with.append(i_PV_with[idx1:idx2]) 
			mean_i_without.append(i_PV_without[idx1:idx2]) 
		
		if len(mean_g_with[0])!=len(mean_g_with[-1]): 
			mean_g_with = mean_g_with[:-1] 
			mean_g_without = mean_g_without[:-1] 
			mean_i_with = mean_i_with[:-1] 
			mean_i_without = mean_i_without[:-1] 
		mean_g_with = np.mean(mean_g_with, axis=0) 
		mean_g_without = np.mean(mean_g_without, axis=0) 
		mean_i_with = np.mean(mean_i_with, axis=0) 
		mean_i_without = np.mean(mean_i_without, axis=0) 

		f, ax = plt.subplots(2, 1)
		f.suptitle('Average PV Conductance & Current on Pyramidal (around stimulus times, PV_to_Pyr_weight = {}, SOM_to_PV_weight = {})'.format(PV_to_Pyr_weight, SOM_to_PV_weight))                                                                                             
		ax[0].set_title('Conductance', size=10)                                                        
		ax[0].plot(np.arange(-20, 150, dt), mean_g_with,label='With Disinhibition')                                                                           
		ax[0].plot(np.arange(-20, 150, dt), mean_g_without,label='Without Disinhibition')                                                                                             
		ax[0].axvline(0, LineStyle='--', color='gray', label='Stimulus Time')                                                                                                 
		ax[0].legend()                                                                                                                                                        
		ax[0].set_ylabel('G ($\mu$S)')  

		ax[1].set_title('Current', size=10)                                                        
		ax[1].plot(np.arange(-20, 150, dt), mean_i_with,label='With Disinhibition')                                                                           
		ax[1].plot(np.arange(-20, 150, dt), mean_i_without,label='Without Disinhibition')                                                                                             
		ax[1].axvline(0, LineStyle='--', color='gray', label='Stimulus Time')                                                                                                 
		ax[1].legend()                                                                                                                                                        
		ax[1].set_xlabel('T (ms)')                                                                                                                                               
		ax[1].set_ylabel('I (nA)')                                                                                                                                               
	plot_G_and_I(Pyr_data_with_disinhibition_job, Pyr_data_without_disinhibition_job)




def CompareFRs(which_cell, job_list):
	def get_SpikeTimes(t, soma_v, threshold=0):

		spike_idx = [i for i in range(1, len(soma_v)) if (soma_v[i] > threshold) and 
														 (soma_v[i] > soma_v[i-1]) and 
														 (soma_v[i] > soma_v[i+1])]
		spike_times = [t[i] for i in spike_idx]

		return spike_times

	def get_FR(stim_times, spike_times, time_unit='sec', take_before=20, take_after=100, window=6):

		bins 		= range(-take_before, take_after+window, window)
		INTERVALS 	= [[bins[i], bins[i+1]] for i in range(len(bins)-1)]
		FRs 		= {str(i): [] for i in INTERVALS}
		in_window = lambda time, inter: time > inter[0] and time <= inter[1] 

		if not stim_times:
			stim_times = cPickle.load(open(filenames['stim_times'],'rb'))[6666]
			stim_times = [i[0] for i in stim_times if i[0]<tstop]

		for time in stim_times:

			shifted_spikes = [i-time for i in spike_times]

			for interval in INTERVALS:

				spikes_in_interval = [spike for spike in shifted_spikes if in_window(spike, interval)]

				if time_unit=='sec':
					FRs[str(interval)].append(1000 * len(spikes_in_interval)/window)
				elif time_unit=='ms':
					FRs[str(interval)].append(len(spikes_in_interval)/window)
				else:
					raise Exception('Invalid time unit value!')

		return FRs, bins, stim_times

	def plot_FR(h_ax, FRs, subplots_ax):

		mean_FRs, mean_inters = [], []
		for interval_str in FRs:
			interval = [int(j) for j in interval_str.split('[')[1].split(']')[0].split(',')]
			mean_FRs.append(np.mean(FRs[interval_str]))
			mean_inters.append(np.mean(interval))

		if not h_ax:
			_, h_ax = plt.subplots()

		if color=='k':
			h_ax.plot(mean_inters, mean_FRs, label=cond_str, color=color, alpha=0.8, LineWidth=3)
		else:
			h_ax.plot(mean_inters, mean_FRs, label=cond_str, color=color, alpha=0.8)
		h_ax.set_title('Mean Firing Rate for Cell with Thalamic Input')
		h_ax.set_ylabel('Firing Rate (Hz)')

		h_ax.axvline(0, LineStyle='--', LineWidth=1, color='k')
		h_ax.legend()
		h_ax.set_xlabel('Peri-Stimulus Time')

		subplots_ax.plot(mean_inters, mean_FRs, color=color, alpha=0.8, label=cond_str)

		return h_ax

	datas = {}
	stim_times = []
	h_ax = None
	FR_window = 1
	C = iter(['k', 'xkcd:magenta', 'xkcd:crimson', 'xkcd:azure'])
	F, subplots_ax = plt.subplots(2, 2)

	reg_dis_ax 		= subplots_ax[0][0]
	un_ax 			= subplots_ax[0][1]
	dis_dis_ax 		= subplots_ax[1][0]
	dis_dis_un_ax 	= subplots_ax[1][1]

	reg_dis_ax.set_title('SOM Connected to Pyr', color='green')
	un_ax.set_title('SOM Disconnected From Pyr', color='red')
	reg_dis_ax.set_ylabel('SOM Connected to PV', color='green')
	dis_dis_ax.set_ylabel('SOM Disconnected From PV', color='red')

	ax_dict = {'Regular Disinhibition': reg_dis_ax, 'Uninhibition': un_ax, 'Dis-Disinhibition': dis_dis_ax, 'Dis-Disinhibition & Uninhibition': dis_dis_un_ax}

	for i in range(len(job_list)):		

		job = job_list[i][0]
		cond_str = job_list[i][1]

		axis = ax_dict[cond_str]

		color = next(C)

		temp1, temp2, temp3, tstop = load_Data([which_cell], job, thalamic_path='thalamic_synapse_recordings/Pyr_0.5_PV_0.4_SOM_0.4')
		t = np.arange(0, tstop, 0.025)

		assert len([i for i in [temp1, temp2, temp3] if i])==1, 'Too many data objects (expected 1, got more, or 0)'
		datas[job] = [i for i in [temp1, temp2, temp3] if i][0]

		spike_times = get_SpikeTimes(t, datas[job][which_cell+'0']['soma_v'])

		FRs, bins, stim_times = get_FR(stim_times, spike_times, window=FR_window)
		
		h_ax = plot_FR(h_ax, FRs, axis)

	plt.xlim([-20, 100])

	subplots_ax = [j for i in subplots_ax for j in i]
	YLIM_top = max([i.get_ylim()[1] for i in subplots_ax])
	YLIM_bottom = min([i.get_ylim()[0] for i in subplots_ax])
	F.suptitle('Pyramidal Mean Firing Rate (bins: %ims, simulation length: %sms)'%(FR_window, format(tstop,',d'))) 

	for i in subplots_ax:
		i.set_ylim([YLIM_bottom, YLIM_top])
		i.axvline(0, color='gray', LineStyle='--', label='Autidory Stimulus') 
		i.legend() 

CompareFRs('Pyr', [['9036475', 'Regular Disinhibition'], ['9037163', 'Dis-Disinhibition'], ['9036478', 'Uninhibition'], ['9036477', 'Dis-Disinhibition & Uninhibition']])


















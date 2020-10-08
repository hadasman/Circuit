import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from math import log
from mpl_toolkits.axes_grid.inset_locator import (inset_axes, InsetPosition, mark_inset)
import pdb, copy
from tqdm import tqdm
import _pickle as cPickle
from tqdm import tqdm
import multiprocessing
from time import time

from Parameter_Initialization import * # Initialize parameters before anything else!
from plotting_functions import plotThalamicResponses, Wehr_Zador, PlotSomas, plotFRs
thalamic_locs_filename		  	     = 'thalamocortical_Oren/thalamic_data/thalamic_axons_location_by_gid.pkl'
thal_connections_filename 	  	   	 = 'thalamocortical_Oren/thalamic_data/thalamo_cortical_connectivity.pkl'
flatten = lambda list: [j for i in list for j in i]

# ========================================== Define Stimulus to Analyze ==========================================
thalamic_activations_filenames = {6666: 'thalamocortical_Oren/CreateThalamicInputs/test_times/single_tone/input6666_106746_BroadBand_single_IPI_2000.p', 
								  9600: 'thalamocortical_Oren/CreateThalamicInputs/test_times/single_tone/input9600_109680_BroadBand_single_IPI_2000.p'}
# stim_times_filename 			     = 'thalamocortical_Oren/CreateThalamicInputs/test_times/stim_times_pairs_120.p'
stim_pairs_interval = None
inter_pair_interval = 2000

locs = pd.read_pickle(thalamic_locs_filename)
# locs = locs.set_index('gid')

f1 = 6666
f2 = 9600
min_freq = 4000
min_freq_loc = min(locs.x) 
in_window = lambda time, inter: time > inter[0] and time <= inter[1] 
window = 100
window_secs = window/1000 # Convert to seconds
PSTH_window_size = 1.5
def get_GIDs(upload_from):

	if upload_from:
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
chosen_GIDs, chosen_PV_n_contacts, thalamic_GIDs = get_GIDs('GIDs_instantiations/pyr_72851_between_6666_9600')


def worker_job(data_to_send):
	freq 					= data_to_send[0]
	activations_filename 	= data_to_send[1]
	stim_times 				= data_to_send[2]

	temp_activations = cPickle.load(open(activations_filename, 'rb'))
	# Take only times of standard frequency stimulus

	if hasattr(stim_times[0], '__len__'):
		stim_times  = [i[0] for i in stim_times if i[1]==freq] 
	else:
		stim_times  = stim_times
		print('WARNING: not distinguishing between standard and deviant stim times!')
	
	if stim_pairs_interval:
		stim_times = flatten([[i, i+stim_pairs_interval] for i in stim_times])
	intervals   = [[i, i+window] for i in stim_times]

	# Take only times that were at most <window> ms after stimulus
	temp_responses = {gid: {'times':[], 'mean_FR': None, 'count': None} for gid in temp_activations}   
	activations = {}
	gids_to_pyr = [i[0] for i in thalamic_GIDs['to_pyr']]
	for gid in temp_activations:
		activations[gid] = temp_activations[gid]
		all_times = temp_activations[gid]

		for time in all_times:
			time_in_window = any([in_window(time, inter) for inter in intervals])
			if time_in_window:
				temp_responses[gid]['times'].append(time)

	
	# Count responses and mean firing rate
	for axon in temp_responses: 
		temp_responses[axon]['count'] = len(temp_responses[axon]['times'])
		temp_responses[axon]['mean_count'] = np.mean([sum([in_window(time, inter) for time in temp_responses[axon]['times']]) for inter in intervals])
		temp_responses[axon]['mean_FR'] = (1/window_secs) * temp_responses[axon]['mean_count']
	
	return (freq, activations, temp_responses)		

def worker_PSTH(data):
	stim_time = data[0]
	start = data[1]
	end = data[2]
	activations = data[3]

	B = np.arange(start, end, PSTH_window_size)
	H = []

	interval = [stim_time+start, stim_time+end]
	gid_count = {}
	temp_PSTH = []
	for gid in activations:
		temp_gid_count = []

		all_times = activations[gid]
		for act_time in all_times:
			if in_window(act_time, interval):
				temp_PSTH.append(act_time-stim_time)
				temp_gid_count.append(act_time-stim_time)

		gid_count[gid] = temp_gid_count

		temp_H, _ = np.histogram(gid_count[gid], bins=B)
		temp_H = [1000*i/PSTH_window_size for i in temp_H]
		H.append(temp_H)
	H = np.mean(H, axis=0)

	return (stim_time, gid_count, H, B, temp_PSTH)

def plotSpontaneous(filename, spont_interval):
	activations = cPickle.load(open(filename, 'rb')) # activations = [[spike time, axon gid], ...., [spike time, axon gid]]

	interval_length = spont_interval[1] - spont_interval[0] # In ms
	act_dict = {gid: {'times':[], 'FR': None} for gid in activations}   

	for gid in activations:
		all_times = activations[gid]
		
		for time in all_times:
			if in_window(time, spont_interval):
				act_dict[gid]['times'].append(time)

	for axon in act_dict:
		act_dict[axon]['FR'] = 1000 * len(act_dict[axon]['times']) / interval_length # In Hz

	plt.figure()
	plt.hist([act_dict[a]['FR'] for a in activations], bins=9)
	plt.title('Spontaneous Firing Rate in Thalamic Axons')
	plt.xlabel(r'FR ($\frac{spikes}{sec}$)')
	plt.ylabel('No. of Axons')

def getAxonResponses(standard_freqs, activations_filename, stim_times_dict, window=window, axon_gids=None):

	responses_dict, activations = {}, {}
	
	print('Processing axon responses')

	pool = multiprocessing.Pool()
	data_to_send = [[standard_freqs[0], activations_filename[standard_freqs[0]], stim_times_dict[standard_freqs[0]]], 
					[standard_freqs[1], activations_filename[standard_freqs[1]], stim_times_dict[standard_freqs[1]]]]
	
	for i in list(tqdm(pool.imap(worker_job, data_to_send), total=len(data_to_send))):
		temp_freq = i[0]
		activations[temp_freq] = i[1]
		responses_dict[temp_freq] = i[2]

	return activations, responses_dict

def plotAxonResponses(locs, responses_dict, color='black', h_ax=None):
	if not h_ax:
		_, h_ax = plt.subplots()

	h_ax.plot([locs.loc[axon].x for axon in responses_dict], [responses_dict[axon]['mean_FR'] for axon in responses_dict], '.', color=color)                                                                                        
	
	h_ax.set_title('Thalamic Axons Firing Rate for 2 Frequencies')
	h_ax.set_xlabel('X-axis location of thalamic axons')
	h_ax.set_ylabel('Firing Rate (spikes/sec)')
	h_ax.set_xlim([65, 630])

	return h_ax

def plot_FreqLoc(h_ax, pairs, min_freq, min_freq_loc, df_dx=3.5):

	for p in pairs:
		freq = p[0]
		color = p[1]

		# Calculate location
		dOctave = log(freq / min_freq, 2) # In octaves
		d_mm =  dOctave / df_dx			  # In millimeters
		d_microne = d_mm * 1000 # In micrones
		freq_loc = min_freq_loc + d_microne

		# Plot on main axis
		y1 = h_ax.get_ylim()[0]
		y2 = 0.035
		h_ax.axvline(freq_loc, color=color, LineStyle='--', label='standard {}Hz'.format(freq))

	h_ax.legend()

def plotPSTH(activations_dict, stim_times_dict, h_ax=None, BINS=100):
	global PSTH
	if not h_ax:
		_, h_ax = plt.subplots()

	FR_fig, FR_ax = plt.subplots()

	ip = InsetPosition(h_ax, [0.1, 0.8, 0.15, 0.15])
	inset_ax = plt.axes([0, 0, 1, 1])
	inset_ax.set_axes_locator(ip)

	for freq in activations_dict:
		if hasattr(stim_times_dict[freq][0], '__len__'):
			stim_times  = [i[0] for i in stim_times_dict[freq] if i[1]==freq] 
		else:
			stim_times = stim_times_dict[freq]
		activations = activations_dict[freq]
		PSTH = {i: [] for i in stim_times}
		n_times = len(stim_times)
		start = -20
		end = 300
		print('Creating PSTH for {}Hz'.format(freq))

		FRs = {i: {gid: [] for gid in activations}  for i in stim_times}
		H = []

		pool = multiprocessing.Pool()
		data_PSTH_job = [[stim_time, start, end, activations] for stim_time in stim_times]
		for i in list(tqdm(pool.imap(worker_PSTH, data_PSTH_job), total=len(data_PSTH_job))):
			stim_time 		= i[0]
			FRs[stim_time] 	= i[1]
			H.append(i[2])
			B 				= i[3]
			PSTH[stim_time] = i[4]

		H = np.mean(H, axis=0)
		
		FR_ax.bar(B[:-1], H, width=np.diff(B)[0]*0.9, alpha=0.5, label='{}Hz'.format(freq))

		h_ax.hist([j for i in PSTH.values() for j in i], bins=BINS, label='{}Hz'.format(freq), alpha=0.5)
		inset_ax.hist([j for i in PSTH.values() for j in i], bins=BINS, alpha=0.5)

	h_ax.axvline(0, LineStyle='--', color='k', label='Stimulus Presentation')
	h_ax.set_title('PSTH of Thalamic Axon Responses to tone')                                                         
	h_ax.set_xlabel('Peri-Stimulus Time (ms)')                                                                               
	h_ax.set_ylabel('No. of Spikes')  
	h_ax.set_xlim([start, end]) 
	h_ax.legend()

	inset_ax.set_xlim([-15, -5])
	inset_ax.set_ylim([0, 500])
	inset_ax.get_xaxis().set_visible(False)
	mark_inset(h_ax, inset_ax, loc1=3, loc2=4, fc="none", ec='0.5') 
	# plt.gcf().subplots_adjust(hspace=0.34, bottom=0.08, top=0.9) 

	FR_ax.axvline(0, LineStyle='--', color='k', label='Stimulus Presentation')
	FR_ax.set_ylabel('FR (Hz)')
	FR_ax.set_xlabel('Peri-Stimulus Time (ms)')
	FR_fig.suptitle('Average Firing Rate in Bins of {}ms Around Auditory Stimulus ({})'.format(PSTH_window_size, thalamic_activations_filenames[6666]))
	FR_ax.set_title('(calculated by averaging PSTH turned into FR, for each stimulus time and gid)')
	FR_ax.legend()

	return h_ax

def removeNonResponsive(original_dict, cutoff_freq=1.8):

	uncut_gids = {}
	cut_dict = copy.copy(original_dict)
	for freq in cut_dict:
		for axon in list(cut_dict[freq].keys()):
			if cut_dict[freq][axon]['mean_FR'] < cutoff_freq:
				del cut_dict[freq][axon]

		uncut_gids[freq] = list(cut_dict[freq].keys())
	return cut_dict, uncut_gids

def FR_by_GID(chosen_pop, chosen_cell, stim_times, take_after=50, all_FRs=None, mean_all_FRs_dict=None, stim_to_thal_delay=10, compare_to_random=False):

	def get_chosen_FRs(INPUTS, stim_times, stim_to_thal_delay):
		FRs = {axon: [] for axon in INPUTS} 
		stim_times = [i[0] for i in stim_times]  
		for axon in INPUTS:   
			temp_stim_times = INPUTS[axon]['stim_times']   
			for T in stim_times:   
				temp_spikes = [i for i in temp_stim_times if (i>=T+stim_to_thal_delay) and (i<=T+stim_to_thal_delay+take_after)]   
				FRs[axon].append(1000*len(temp_spikes)/take_after) 
				chosen_axon_GIDs = [i.split('_')[-1] for i in FRs] 
		
		return FRs, chosen_axon_GIDs

	def get_all_FRs(stim_times, stim_to_thal_delay):

		if len(stim_times) != 2:
			stim_times = [i[0] for i in stim_times]

		A = cPickle.load(open(filenames['thalamic_activations_6666'],'rb')) 
		all_FRs = [] 
		mean_all_FRs_dict = {axon: [] for axon in A}
		for axon in tqdm(A): 
			for T in stim_times: 
				temp_spikes = [i for i in A[axon] if (i>=T+stim_to_thal_delay) and (i<=T+stim_to_thal_delay+take_after)] 
				temp_FR = 1000*len(temp_spikes)/take_after
				all_FRs.append(temp_FR) 		
				mean_all_FRs_dict[axon].append(temp_FR)

		for axon in A:
			mean_all_FRs_dict[axon] = np.mean(mean_all_FRs_dict[axon])


		return all_FRs, mean_all_FRs_dict

	INPUTS = chosen_pop.inputs[chosen_cell] 
	
	FRs, chosen_axon_GIDs = get_chosen_FRs(INPUTS, stim_times, stim_to_thal_delay)

	if not all_FRs or not mean_all_FRs_dict:
		all_FRs, mean_all_FRs_dict = get_all_FRs(stim_times, stim_to_thal_delay)

	shuf_GIDs = np.random.choice(list(mean_all_FRs_dict.keys()), replace=False, size=len(FRs)) 
	rand_FRs = [mean_all_FRs_dict[i] for i in shuf_GIDs] 

	mean_FRs = [np.mean(FRs[i]) for i in FRs]
	std_FRs = [np.std(FRs[i])/np.sqrt(len(FRs)) for i in FRs]

	idx_sorted = [mean_FRs.index(i) for i in sorted(mean_FRs)]
	sorted_mean_FRs = [mean_FRs[i] for i in idx_sorted]
	chosen_axon_GIDs = [chosen_axon_GIDs[i] for i in idx_sorted]
	sorted_std_FRs = [std_FRs[i] for i in idx_sorted]
	plt.bar(chosen_axon_GIDs, sorted_mean_FRs, yerr=sorted_std_FRs,color='b', label='Presynaptic Axons')  

	if compare_to_random:
		idx_sorted_rand = [rand_FRs.index(i) for i in sorted(rand_FRs)]
		sorted_rand_FRs = [rand_FRs[i] for i in idx_sorted_rand]
		plt.bar(chosen_axon_GIDs, sorted_rand_FRs, alpha=0.8, color='xkcd:azure', label='Randomly Chosen GIDs')                    

	MEAN = np.median(all_FRs)
	SE = np.std(all_FRs)/np.sqrt(len(all_FRs)) 

	plt.fill_between([i for i in chosen_axon_GIDs], MEAN-SE, MEAN+SE, color='coral', alpha=0.5,label='SE')         
	plt.axhline(MEAN, LineStyle='--', color='coral',label='Mean FR')                             
	plt.legend()                                                                                             
	plt.xlabel('Thalamic Axon GIDs')
	plt.ylabel('FR (Hz)')                                                   
	plt.suptitle('Response FR for axons on {}, Sorted by FR (taken {}-{}ms after stimulus); ({})'.format(chosen_cell, stim_to_thal_delay, take_after, thalamic_activations_filenames[6666]))              
	plt.subplots_adjust(left=0.07,right=0.99,bottom=0.11,top=0.92) 
	# plt.gca().set_xticklabels(plt.gca().get_xticklabels(), size=5) 

	return all_FRs, mean_all_FRs_dict

# all_FRs, mean_all_FRs_dict = None, None
# stim_times = cPickle.load(open(filenames['stim_times'], 'rb'))[6666]
# plt.figure()
# all_FRs, mean_all_FRs_dict = FR_by_GID(SOM_pop, 'SOM0', stim_times, all_FRs=all_FRs, mean_all_FRs_dict=mean_all_FRs_dict)
# plt.figure()
# all_FRs, mean_all_FRs_dict = FR_by_GID(PV_pop, 'PV0', stim_times, all_FRs=all_FRs, mean_all_FRs_dict=mean_all_FRs_dict)
# plt.figure()
# all_FRs, mean_all_FRs_dict = FR_by_GID(Pyr_pop, 'Pyr0', stim_times, all_FRs=all_FRs, mean_all_FRs_dict=mean_all_FRs_dict)


if __name__ == '__main__':

	# stim_times  = pd.read_pickle(stim_times_filename)	
	# stim_times_list = sorted(list(range(2000, 154000, 300)) + list(range(2120, 154000, 300)))
	stim_times_list = list(range(2000, 154000, inter_pair_interval))
	# stim_times_list = list(range(154000)); np.random.shuffle(stim_times_list); stim_times_list = stim_times_list[:500] # Control- should not give stimulus-like PSTH
	stim_times = {6666: stim_times_list, 9600: stim_times_list}    
	
	activations_dict, responses_dict = getAxonResponses([6666, 9600], thalamic_activations_filenames, stim_times)

	plotSpontaneous(thalamic_activations_filenames[6666], [0, 2000])

	F, (scatter_ax, hist_ax) = plt.subplots(2, 1, gridspec_kw = {'height_ratios':[3, 1]})
	F.subplots_adjust(hspace=0.34, bottom=0.07, top=0.91) 
	scatter_ax = plotAxonResponses(locs, responses_dict[6666], color='red', h_ax=scatter_ax)
	# scatter_ax = plotAxonResponses(locs, responses_dict[9600], color='blue', h_ax=scatter_ax)

	mean_FR_1 = np.mean([responses_dict[f1][axon]['mean_FR'] for axon in responses_dict[6666]])
	# mean_FR_2 = np.mean([responses_dict[f2][axon]['mean_FR'] for axon in responses_dict[9600]])
	plot_FreqLoc(scatter_ax, [[f1, 'red'], [f2, 'blue']], min_freq, min_freq_loc)

	hist_ax = plotPSTH(activations_dict, stim_times, h_ax=hist_ax)
	hist_ax.set_title(scatter_ax.get_title() + ' (With Non-Responsive Axons)')

	plot_cut = False
	if plot_cut:
		F_cut, (cut_scatter_ax, cut_hist_ax) = plt.subplots(2, 1, gridspec_kw = {'height_ratios':[3, 1]})
		F_cut.subplots_adjust(hspace=0.34, bottom=0.07, top=0.91) 
		cut_responses_dict, cut_gids = removeNonResponsive(responses_dict)

		cut_scatter_ax = plotAxonResponses(locs, cut_responses_dict[6666], color='red', h_ax=cut_scatter_ax)
		# cut_scatter_ax = plotAxonResponses(locs, cut_responses_dict[9600], color='blue', h_ax=cut_scatter_ax)

		mean_FR_1 = np.mean([cut_responses_dict[f1][axon]['mean_FR'] for axon in cut_responses_dict[6666]])
		# mean_FR_2 = np.mean([cut_responses_dict[f2][axon]['mean_FR'] for axon in cut_responses_dict[9600]])
		plot_FreqLoc(cut_scatter_ax, [[f1, 'red'], [f2, 'blue']], min_freq, min_freq_loc)
		cut_hist_ax = plotPSTH(activations_dict, stim_times, h_ax=cut_hist_ax)

		cut_scatter_ax.set_title(cut_scatter_ax.get_title() + ' (Without Non-Responsive Axons)')
	# all_FRs, mean_all_FRs_dict = FR_by_GID(Pyr_pop, 'Pyr0', stim_times)


'''
L1 = h_ax.plot(0,0,'r.', label='standard {}Hz'.format(f1))[0]
L2 = h_ax.plot(0,0,'b.', label='standard {}Hz'.format(f2))[0]
h_ax.legend()
L1.set_alpha(0); L2.set_alpha(0)

def plotAxonFR(filename, h_ax=None, color='black'):
	if not h_ax:
		_, h_ax = plt.subplots()

	activations = cPickle.load(open(activations_filename, 'rb'))
	
	axons = list(set([i[1] for i in activations]))                                                                                  
	act_dict = {a: {'times':[], 'FR': None} for a in axons}                                                                                            

	for a in activations: 
		gid = a[1] 
		act_dict[gid]['times'].append(a[0]) 

	for axon in act_dict: 
		temp_times = sorted(act_dict[axon]['times']) 
		FR = len(temp_times)/(max(temp_times)-min(temp_times)) 
		act_dict[axon]['FR']=FR 

		h_ax.plot(locs.loc[axon].x, act_dict[axon]['FR'], '.', color=color)
	
	return h_ax


def plot_IndFR(filename, window=2000):
	activations = cPickle.load(open(filename, 'rb'))

	axons = list(set([i[1] for i in activations]))                                                                                  
	act_dict = {a: {'times':[], 'FR': None} for a in axons}                                                                                            

	for a in activations: 
		gid = a[1] 
		act_dict[gid]['times'].append(a[0]) 

	_, ax = plt.subplots()
	fig2, ax2 = plt.subplots()
	for gid in act_dict:
		H = act_dict[gid]['times'] 
		BINS = range(0, 154*1000, window) 
 
		output = ax2.hist(H, bins=BINS) 
		h = [1000*i/window for i in output[0]]
		b = output[1]
		ax.bar([np.mean([b[i], b[i+1]]) for i in range(len(b)-1)], h, width=1000, alpha=0.5) 
	plt.close(fig2)
plot_IndFR(thalamic_activations_filename_6666, window=2000)

                                      





	intervals = [[i, i+window] for i in range(0, int(np.ceil(activations[-1][0])), window)]  

	gid=221184

	time_division = {i: [] for i in range(len(intervals))}
	for t in act_dict[gid]['times']:
		idx = np.where([in_window(t, i) for i in intervals])[0][0] 
		time_division[idx].append(t)
	plt.plot(list(time_division.keys()), [len(time_division[i])/window for i in time_division], '.')


'''










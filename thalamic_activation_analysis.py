import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from math import log
from mpl_toolkits.axes_grid.inset_locator import (inset_axes, InsetPosition, mark_inset)
import pdb, copy
from tqdm import tqdm
import _pickle as cPickle
from tqdm inmport tqdm

thalamic_locs_filename		  	     = 'thalamocortical_Oren/thalamic_data/thalamic_axons_location.pkl'
thal_connections_filename 	  	   	 = 'thalamocortical_Oren/thalamic_data/thalamo_cortical_connectivity.pkl'
thalamic_activations_filenames 		 = {}
# thalamic_activations_filenames[6666] = 'thalamocortical_Oren/SSA_spike_times/input6666_106746.dat'
thalamic_activations_filenames[6666] = 'new_activations_by_gid.p'
thalamic_activations_filenames[9600] = 'thalamocortical_Oren/SSA_spike_times/input9600_by_gid.p'

# thalamic_activations_filenames[9600] = 'thalamocortical_Oren/SSA_spike_times/input9600_109680.dat'
stim_times_filename 			     = 'thalamocortical_Oren/SSA_spike_times/stim_times.p'

locs = pd.read_pickle(thalamic_locs_filename)
locs = locs.set_index('gid')

f1 = 6666
f2 = 9600
min_freq = 4000
min_freq_loc = min(locs.x) 
in_window = lambda time, inter: time > inter[0] and time <= inter[1] 

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

def getAxonResponses(standard_freqs, activations_filename, stim_times_dict, window=100, axon_gids=None):

	responses_dict, activations = {}, {}
	window_secs = window/1000 # Convert to seconds
	print('Processing axon responses')
	for freq in tqdm(standard_freqs):
		
		activations[freq] = cPickle.load(open(activations_filename[freq], 'rb'))
		# Take only times of standard frequency stimulus
		stim_times  = [i[0] for i in stim_times_dict[freq] if i[1]==freq] 
		intervals   = [[i, i+window] for i in stim_times]

		# Take only times that were at most <window> ms after stimulus
		responses_dict[freq] = {gid: {'times':[], 'mean_FR': None, 'count': None} for gid in activations[freq]}   

		for gid in activations[freq]: 
			all_times = activations[freq][gid]

			for time in all_times:
				time_in_window = any([in_window(time, inter) for inter in intervals])
				if time_in_window:
					responses_dict[freq][gid]['times'].append(time)
		
		# Count responses and mean firing rate
		for axon in responses_dict[freq]: 
			responses_dict[freq][axon]['count'] = len(responses_dict[freq][axon]['times'])
			responses_dict[freq][axon]['mean_count'] = np.mean([sum([in_window(time, inter) for time in responses_dict[freq][axon]['times']]) for inter in intervals])
			responses_dict[freq][axon]['mean_FR'] = (1/window_secs) * responses_dict[freq][axon]['mean_count']

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

def plotPSTH(activations_dict, stim_times_dict, h_ax=None):

	if not h_ax:
		_, h_ax = plt.subplots()


	ip = InsetPosition(h_ax, [0.1, 0.8, 0.15, 0.15])
	inset_ax = plt.axes([0, 0, 1, 1])
	inset_ax.set_axes_locator(ip)

	for freq in activations_dict:
		stim_times  = [i[0] for i in stim_times_dict[freq] if i[1]==freq] 
		activations = activations_dict[freq]
		PSTH = {i: [] for i in stim_times}
		n_times = len(stim_times)
		start = -20
		end = 50
		print('Creating PSTH for {}Hz'.format(freq))
		for stim_time in tqdm(stim_times):
			interval = [stim_time+start, stim_time+end]
			for gid in activations:
				all_times = activations[gid]
				for act_time in all_times:
					if in_window(act_time, interval):
						PSTH[stim_time].append(act_time-stim_time)

		h_ax.hist([j for i in PSTH.values() for j in i], bins=100, label='{}Hz'.format(freq), alpha=0.5)
		inset_ax.hist([j for i in PSTH.values() for j in i], bins=100, alpha=0.5)

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

def FR_by_GID(chosen_pop, chosen_cell, stim_times, take_after=50, all_FRs=None, mean_all_FRs_dict=None):

	def get_chosen_FRs(INPUTS):
		FRs = {axon: [] for axon in INPUTS}   
		for axon in INPUTS:   
			temp_stim_times = INPUTS[axon]['stim_times']   
			for T in temp_stim_times:   
				temp_spikes = [i for i in temp_stim_times if (i>=T) and (i<=T+take_after)]   
				FRs[axon].append(1000*len(temp_spikes)/take_after) 
				chosen_axon_GIDs = [i.split('_')[-1] for i in FRs] 
		
		return FRs, chosen_axon_GIDs

	def get_all_FRs(stim_times):

		if len(stim_times) == 2:
			stim_times = [i[0] for i in stim_times]

		A = cPickle.load(open(filenames['thalamic_activations_6666'],'rb')) 
		all_FRs = [] 
		mean_all_FRs_dict = {axon: [] for axon in A}
		for axon in tqdm(A): 
			for T in stim_times: 
				temp_spikes = [i for i in A[axon] if (i>=T) and (i<=T+take_after)] 
				temp_FR = 1000*len(temp_spikes)/take_after
				all_FRs.append(temp_FR) 		
				mean_all_FRs_dict[axon].append(temp_FR)

		for axon in A:
			mean_all_FRs_dict[axon] = np.mean(mean_all_FRs_dict[axon])


		return all_FRs, mean_all_FRs_dict

	INPUTS = chosen_pop.inputs[chosen_cell] 
	
	FRs, chosen_axon_GIDs = get_chosen_FRs(INPUTS)

	if not all_FRs or not mean_all_FRs_dict:
		all_FRs, mean_all_FRs_dict = get_all_FRs(stim_times)

	shuf_GIDs = np.random.choice(list(mean_all_FRs_dict.keys()), replace=False, size=len(FRs)) 
	rand_FRs = [mean_all_FRs_dict[i] for i in shuf_GIDs] 

	plt.bar(chosen_axon_GIDs, [np.mean(FRs[i]) for i in FRs], yerr=[np.std(FRs[i])/np.sqrt(len(FRs)) for i in FRs],color='b')  
	plt.bar(chosen_axon_GIDs, rand_FRs, alpha=0.8, color='xkcd:azure', label='Randomly Chosen GIDs')                    

	MEAN = np.mean(all_FRs)
	SE = np.std(all_FRs)/np.sqrt(len(all_FRs)) 

	plt.fill_between([i for i in chosen_axon_GIDs], MEAN-SE, MEAN+SE, color='coral', alpha=0.5,label='SE')         
	plt.axhline(np.mean(all_FRs), LineStyle='--', color='coral',label='Mean FR')                             
	plt.legend()                                                                                             
	plt.xlabel('Thalamic Axon GIDs')
	plt.ylabel('FR (Hz)')                                                   
	plt.suptitle('Response FR for axons on {} (taken {}ms after stimulus)'.format(chosen_cell, take_after))              

	return all_FRs, mean_all_FRs_dict
all_FRs, mean_all_FRs_dict = FR_by_GID(Pyr_pop, 'Pyr0', stim_times)



if __name__ == '__main__':

	stim_times  = pd.read_pickle(stim_times_filename)	
	
	activations_dict, responses_dict = getAxonResponses([6666, 9600], thalamic_activations_filenames, stim_times)

	plotSpontaneous(thalamic_activations_filenames[6666], [0, 2000])

	_, (scatter_ax, hist_ax) = plt.subplots(2, 1, gridspec_kw = {'height_ratios':[3, 1]})

	scatter_ax = plotAxonResponses(locs, responses_dict[6666], color='red', h_ax=scatter_ax)
	scatter_ax = plotAxonResponses(locs, responses_dict[9600], color='blue', h_ax=scatter_ax)

	mean_FR_1 = np.mean([responses_dict[f1][axon]['mean_FR'] for axon in responses_dict[6666]])
	mean_FR_2 = np.mean([responses_dict[f2][axon]['mean_FR'] for axon in responses_dict[9600]])
	plot_FreqLoc(scatter_ax, [[f1, 'red'], [f2, 'blue']], min_freq, min_freq_loc)

	hist_ax = plotPSTH(activations_dict, stim_times, h_ax=hist_ax)

	_, (cut_scatter_ax, cut_hist_ax) = plt.subplots(2, 1, gridspec_kw = {'height_ratios':[3, 1]})
	cut_responses_dict, cut_gids = removeNonResponsive(responses_dict)

	cut_scatter_ax = plotAxonResponses(locs, cut_responses_dict[6666], color='red', h_ax=cut_scatter_ax)
	cut_scatter_ax = plotAxonResponses(locs, cut_responses_dict[9600], color='blue', h_ax=cut_scatter_ax)

	mean_FR_1 = np.mean([cut_responses_dict[f1][axon]['mean_FR'] for axon in cut_responses_dict[6666]])
	mean_FR_2 = np.mean([cut_responses_dict[f2][axon]['mean_FR'] for axon in cut_responses_dict[9600]])
	plot_FreqLoc(cut_scatter_ax, [[f1, 'red'], [f2, 'blue']], min_freq, min_freq_loc)

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










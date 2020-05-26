import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from math import log
from mpl_toolkits.axes_grid1.inset_locator import (inset_axes, InsetPosition, mark_inset)
import pdb
import _pickle as cPickle
from tkinter import messagebox

class Stimulus():
	def __init__(self, standard_frequency, deviant_frequency, stim_times_filename, thalamic_activations_filename, axon_gids=None):

		def get_activations(filename, axon_gids):
			temp_activations = cPickle.load(open(filename, 'rb'))			
			
			if axon_gids:
				activations = {}
				for a in axon_gids:
					activations[a] = temp_activations[a]
			else:
				activations = temp_activations

			return activations

		self.standard_frequency = standard_frequency
		self.deviant_frequency  = deviant_frequency

		self.stim_times_all 	 = pd.read_pickle(stim_times_filename)[standard_frequency]
		self.stim_times_standard = [i[0] for i in self.stim_times_all if i[1]==self.standard_frequency]
		self.stim_times_deviant  = [i[0] for i in self.stim_times_all if i[1]==self.deviant_frequency]
		
		self.thalamic_activations = get_activations(thalamic_activations_filename, axon_gids=axon_gids)

	def axonResponses(self, thalamic_locations, window=100, color='black', h_ax=None, plot_results=True):
		if not h_ax:
			_, h_ax = plt.subplots()

		# Take only times that were at most <window>ms after stimulus
		in_window = lambda time, inter: time > inter[0] and time <= inter[1] 
		intervals = [[i, i+window] for i in self.stim_times_standard]
		# axons 	  = list(set([i[1] for i in self.thalamic_activations]))                                                                                  
		axons = [i for i in self.thalamic_activations]

		responses_dict 	= {gid: {'times':[], 'mean_FR': None, 'count': None} for gid in axons}   

		# Take stimulus-responses from all spike times
		for axon in self.thalamic_activations: 
			# time = axon[0]
			# gid  = axon[1] 
			gid = axon
			times = self.thalamic_activations[gid]

			for time in times:
				time_in_window = any([in_window(time, inter) for inter in intervals])
				if time_in_window:
					responses_dict[gid]['times'].append(time)

		# Count responses and mean firing rate
		for axon in responses_dict: 
			responses_dict[axon]['count'] = len(responses_dict[axon]['times'])
			responses_dict[axon]['mean_count'] = np.mean([sum([in_window(time, inter) for time in responses_dict[axon]['times']]) for inter in intervals])
			responses_dict[axon]['mean_FR'] = (1/window) * 1000 * responses_dict[axon]['mean_count'] # Divide by window (in ms), turn to seconds

		if plot_results:
			AXONS = [thalamic_locations.loc[axon].x for axon in responses_dict]
			mean_FR = [responses_dict[axon]['mean_FR'] for axon in responses_dict]
			mean_mean_FR = np.mean([i['mean_FR'] for i in responses_dict.values()])
			h_ax.plot(AXONS, mean_FR, '.', color=color, label='{}Hz (mean FR: {:.4f})'.format(self.standard_frequency, mean_mean_FR))

			h_ax.legend()
			h_ax.set_xlabel('X-axis location of thalamic axons')
			h_ax.set_ylabel(r'Firing Rate ($\frac{spikes}{s}$)')

		self.thalamic_responses = responses_dict

		return h_ax			

	def tonotopic_location(self, thalamic_locations, min_freq=4000, df_dx=3.5, color='black', h_ax=None):

		min_freq_loc = min(thalamic_locations.x)
		
		# Calculate location
		dOctave = log(self.standard_frequency / min_freq, 2) # In octaves
		d_mm =  dOctave / df_dx			  # In millimeters
		d_microne = d_mm * 1000 # In micrones
		
		self.standard_location = min_freq_loc + d_microne

		if h_ax:
			h_ax.axvline(self.standard_location, color=color, LineStyle='--')


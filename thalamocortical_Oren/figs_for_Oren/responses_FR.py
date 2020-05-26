import matplotlib
matplotlib.use("TkAgg")
import pandas as pd
import numpy as np
from neuron import gui, h
import pdb, os, sys
import matplotlib.pyplot as plt
plt.ion()
from Stimulus import Stimulus
import _pickle as cPickle
os.chdir('../Circuit')

freq1 = 6666
freq2 = 9600
stimuli = {}
connecting_gids = []
activations_filenames = {}
chosen_pyr = 72851

stim_times_filename = 'thalamocortical_Oren/SSA_spike_times/stim_times.p'
activations_filenames[freq1] = 'thalamocortical_Oren/SSA_spike_times/input6666_106746.dat'
activations_filenames[freq2] = 'thalamocortical_Oren/SSA_spike_times/input9600_109680.dat'

thalamic_locations = pd.read_pickle('thalamocortical_Oren/thalamic_data/thalamic_axons_location_by_gid.pkl')
thal_connections   = pd.read_pickle('thalamocortical_Oren/thalamic_data/thalamo_cortical_connectivity.pkl')

# Find thalamic GIDs connecting the the pyramidal cell
for con in thal_connections.iterrows():
	if con[1].post_gid == chosen_pyr:
		connecting_gids.append(con[1].pre_gid) # [presynaptic gid, no. of contacts]

stimuli[freq1] = Stimulus(freq1, freq2, stim_times_filename, activations_filenames[freq1], axon_gids=connecting_gids)
stimuli[freq2] = Stimulus(freq2, freq1, stim_times_filename, activations_filenames[freq2], axon_gids=connecting_gids)

stim_ax = stimuli[freq1].axonResponses(thalamic_locations, color='red')
stim_ax = stimuli[freq2].axonResponses(thalamic_locations, color='blue', h_ax=stim_ax)

stimuli[freq1].tonotopic_location(thalamic_locations, color='red', h_ax=stim_ax)
stimuli[freq2].tonotopic_location(thalamic_locations, color='blue', h_ax=stim_ax)

stim_ax.set_title('Thalamic Axons (Connected to Chosen Pyramidal) Firing Rate for 2 Frequencies\nPyramidal GID: {}, Spontaneous Axon FR: {}'.format(chosen_pyr, 0.5))
stim_ax.set_xlim([65, 630])













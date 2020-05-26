import pandas as pd
import matplotlib.pyplot as plt
plt.ion()
import numpy as np

thalamic_activations_filename 	  	= {}
thalamic_activations_filename[6666] = 'thalamocortical_Oren/SSA_spike_times/input6666_106746.dat'
thalamic_activations_filename[9600] = 'thalamocortical_Oren/SSA_spike_times/input9600_109680.dat'
stim_times_filename 			   	= 'thalamocortical_Oren/SSA_spike_times/stim_times.p'

standard_stimulus = 6666
stim_times  = pd.read_pickle(stim_times_filename)[standard_stimulus]
first_stim_time = stim_times[0][0]

in_window = lambda time, inter: time > inter[0] and time <= inter[1] 

def get_activations(activations_filename):
	temp_data = [i.strip().split() for i in open(activations_filename).readlines()]
	activations = [] 
	for i in range(len(temp_data)): 
		if temp_data[i][0].replace('.', '').isdigit():
			activations.append([float(temp_data[i][0]), int(float(temp_data[i][1]))]) 
	return activations

def plotSpontaneous(filename, spont_interval):

	# Process activation data
	activations = get_activations(filename) # activations = [[spike time, axon gid], ...., [spike time, axon gid]]

	interval_length = (spont_interval[1] - spont_interval[0]) / 1000 # In second
	axons = list(set([i[1] for i in activations]))                                                                                  
	act_dict = {a: {'times':[], 'FR': None} for a in axons}   

	# Get spike times in the spontaneous interval (before stimuli start)
	for act in activations:
		time = act[0]
		gid = act[1]
		if in_window(time, spont_interval):
			act_dict[gid]['times'].append(time)

	# Calculae FR for each thalamic axon
	for axon in act_dict:
		act_dict[axon]['FR'] = len(act_dict[axon]['times']) / interval_length # In Hz

	plt.figure()
	plt.hist([act_dict[a]['FR'] for a in axons], bins=9)
	plt.title('Spontaneous Firing Rate in Thalamic Axons')
	plt.xlabel(r'FR ($\frac{spikes}{sec}$)')
	plt.ylabel('No. of Axons')

plotSpontaneous(thalamic_activations_filename[standard_stimulus], [0, first_stim_time])

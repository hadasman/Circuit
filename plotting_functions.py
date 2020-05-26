import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import pdb

def plotThalamicResponses(stimuli, freq1, freq2, thalamic_locations, run_function=False):
	if run_function:

		stim_ax = stimuli[freq1].axonResponses(thalamic_locations, color='red')
		stim_ax = stimuli[freq2].axonResponses(thalamic_locations, color='blue', h_ax=stim_ax)

		stimuli[freq1].tonotopic_location(thalamic_locations, color='red', h_ax=stim_ax)
		stimuli[freq2].tonotopic_location(thalamic_locations, color='blue', h_ax=stim_ax)

		stim_ax.set_title('Thalamic Axons (Connected to Chosen Pyramidal) Firing Rate for 2 Frequencies\nPyramidal GID: {}, Spontaneous Axon FR: {}'.format(chosen_pyr, 0.5))
		stim_ax.set_xlim([65, 630])

		return stim_ax

def Wehr_Zador(population, cell_name, stimuli, title_, exc_weight=0, inh_weight=0, standard_freq=None, input_pop_outputs=None, take_before=20, take_after=155, tstop=None, spike_threshold=None, dt=None, t=None):

	def plot_traces(h_ax, all_means, which_traces='Currents', units_='nA'):

		if which_traces == 'Currents':
			which_plot = 'i'
			unit_conversion = 1
		elif which_traces == 'Conductances':
			which_plot = 'g'
			unit_conversion = 1000

		AMPA = [i*unit_conversion for i in all_means['%s_AMPA'%which_plot]]
		NMDA = [i*unit_conversion for i in all_means['%s_NMDA'%which_plot]]			

		h_ax.axvline(take_before, LineStyle='--', color='gray')
		h_ax.plot(t_vec, [AMPA[i]+NMDA[i] for i in range(n_points)], 'purple', label='%s$_{AMPA}$ + %s$_{NMDA}$'%(which_plot, which_plot))

		if input_pop_outputs:
			GABA = [i*unit_conversion for i in all_means['%s_GABA'%which_plot]]
			if list(input_pop_outputs.values())[0]:
				h_ax.plot(t_vec, GABA, 'b', label='%s$_{GABA}$'%which_plot)
				h_ax.plot(t_vec, [GABA[i]+AMPA[i]+NMDA[i] for i in range(n_points)], label='%s$_{tot}$'%which_plot)

		h_ax.legend()
		h_ax.set_title('Mean Synaptic %s'%which_traces)
		h_ax.set_ylabel('%s (%s)'%(which_plot.upper(), units_))
		h_ax.set_xlim([0, take_before+take_after])

	fig, axes = plt.subplots(3, 1)
	fig.subplots_adjust(hspace=0.34, bottom=0.08, top=0.9) 
	times = [i[0] for i in stimuli[standard_freq].stim_times_all]
	times = [i for i in times if i<tstop]

	cut_vec = lambda vec, start_idx, end_idx: [vec[i] for i in range(start_idx, end_idx)]
	
	spike_count = 0
	all_means = {'i_AMPA': [], 'g_AMPA': [], 'i_NMDA': [], 'g_NMDA': [], 'i_GABA': [], 'g_GABA': []}

	print('Analyzing conductances and currents')
	for T in tqdm(times):
		idx1 = (np.abs([i-(T-take_before) for i in t])).argmin()
		idx2 = (np.abs([i-(T+take_after) for i in t])).argmin()
		
		t_vec = cut_vec(t, idx1, idx2)
		t_vec = [i-t_vec[0] for i in t_vec]
		v_vec = cut_vec(population.cells[cell_name]['soma_v'], idx1, idx2)
		if any([i>=spike_threshold for i in v_vec]):
			spike_count += 1

		all_sums = {'i_AMPA': [], 'g_AMPA': [], 'i_NMDA': [], 'g_NMDA': [], 'i_GABA': [], 'g_GABA': []}

		# ========== Excitation to cell ==========
		for axon in population.inputs[cell_name]:
			for i in ['i_AMPA', 'g_AMPA', 'i_NMDA', 'g_NMDA']:
				temp_vec = [cut_vec(vec, idx1, idx2) for vec in population.inputs[cell_name][axon][i]]
				all_sums[i].append(np.sum(temp_vec, axis=0))

		# ========== Inhibition to cell ==========
		if input_pop_outputs:
			for pre_PV in input_pop_outputs:
				for sec in input_pop_outputs[pre_PV]:
					for i in ['i_GABA', 'g_GABA']:
						temp_vec = [cut_vec(vec, idx1, idx2) for vec in input_pop_outputs[pre_PV][sec][i]]
						all_sums[i].append(np.sum(temp_vec, axis=0))

		# Append the TOTAL currents and conductances, over all synapses in given cell, for the current time point
		for i in all_sums:
			all_means[i].append(np.sum(all_sums[i], axis=0))

		axes[0].plot(t_vec, v_vec, 'k', LineWidth=0.7)

		if T==times[0]:
			axes[0].legend(['%s soma v'%cell_name], loc='upper right')

	# Average over time points
	for i in all_means:
		all_means[i] = np.mean(all_means[i][:-1], axis=0)

	t_vec = [i*dt for i in range(0, len(all_means['i_AMPA']))]
	n_points = len(t_vec)

	plt.suptitle('{} (GID: {}) Cell ({} spikes out of {})'.format(title_, population.name_to_gid[cell_name], spike_count, len(times)))
	axes[0].axvline(take_before, LineStyle='--', color='gray')
	axes[0].set_title('Overlay of Somatic Responses to %sHz Simulus (locked to stimulus prestntation)'%standard_freq)
	axes[0].set_ylabel('V (mV)')
	axes[0].set_xlim([0, take_before+take_after])
	
	# ========== Plot Currents ==========
	plot_traces(axes[1], all_means, which_traces='Currents', units_='nA')

	# ========== Plot Conductances ==========
	plot_traces(axes[2], all_means, which_traces='Conductances', units_='nS')
	axes[2].set_xlabel('T (ms)')

	return axes

def PlotSomas(populations, t, stimulus, tstop=None, spike_threshold=None, dt=None):
	
	_, h_ax = plt.subplots()

	all_cells, cell_names = [], []
	for cell_name in populations:
		pop = populations[cell_name]
		all_cells.append(''.join([i for i in cell_name if not i.isdigit()]))
		cell_names.append(cell_name)

		temp_soma_v = pop.cells[cell_name]['soma_v']
		if len(temp_soma_v) > len(t):
			temp_soma_v = [i for i in temp_soma_v][:len(t)]
		h_ax.plot(t, temp_soma_v, label = '{} ({})'.format(all_cells[-1], pop.name_to_gid[cell_name]))

	for stim in [['Standard', stimulus.stim_times_standard], ['Deviant', stimulus.stim_times_deviant]]:
		times = stim[1]
		for s in times:
			if s < tstop:
				if s == times[0]:
					h_ax.axvline(s, LineStyle='--', color='k', alpha=0.5, label='{} Stimulus'.format(stim[0])) 
				else:
					h_ax.axvline(s, LineStyle='--', color='k', alpha=0.5)

	h_ax.legend()

	title_string = ''
	for i in range(len(populations)):
		name_ = cell_names[i]
		if i==len(populations)-1:
			title_string = title_string + 'and {} ({})'.format(all_cells[i], populations[name_].name_to_gid[name_])
		else:
			title_string = title_string + '{} ({}), '.format(all_cells[i], populations[name_].name_to_gid[name_])

	h_ax.set_title('Example of {} Responses to 2 tones ({}Hz, {}Hz)\n(at tonotopical position between tones, standard: {})'\
					.format(title_string, min(stimulus.standard_frequency, \
							stimulus.deviant_frequency), max(stimulus.standard_frequency, \
							stimulus.deviant_frequency), stimulus.standard_frequency)) 
	h_ax.set_xlabel('T (ms)')
	h_ax.set_ylabel('V (mV)')
	h_ax.set_xlim([0, tstop])

	return h_ax

def plotFRs(stim_times, soma_v, t, tstop=0, window=0, take_before=150, take_after=150, which_cell='', axes_h=None, color=''):
	def get_SpikeTimes(t, soma_v, threshold=0):

		spike_idx = [i for i in range(1, len(soma_v)) if (soma_v[i] > threshold) and 
														 (soma_v[i] > soma_v[i-1]) and 
														 (soma_v[i] > soma_v[i+1])]
		spike_times = [t[i] for i in spike_idx]

		return spike_times
	
	def get_FR_and_PSTH(stim_times, spike_times, time_unit='sec'):

		bins 		= range(-take_before, take_after+window, window)
		INTERVALS 	= [[bins[i], bins[i+1]] for i in range(len(bins)-1)]
		FRs 		= {str(i): [] for i in INTERVALS}
		PSTH 		= []
		
		for time in stim_times:

			shifted_spikes = [i-time for i in spike_times]
			PSTH.append([spike for spike in shifted_spikes if in_window(spike, [-take_before, take_after])])

			for interval in INTERVALS:

				spikes_in_interval = [spike for spike in shifted_spikes if in_window(spike, interval)]

				if time_unit=='sec':
					FRs[str(interval)].append(1000 * len(spikes_in_interval)/window)
				elif time_unit=='ms':
					FRs[str(interval)].append(len(spikes_in_interval)/window)
				else:
					raise Exception('Invalid time unit value!')

		return FRs, PSTH, bins

	def plot_PSTH(h_ax, PSTH, bins, normalized=False):

		H, B = np.histogram([j for i in PSTH for j in i], bins=bins)

		if normalized:			
			H = [i/max(H) for i in H]
			h_ax.set_title('Normalized PSTH of Cell with Thalamic Input')
			h_ax.set_ylabel('Normalized Spike Count')
		else:			
			h_ax.set_title('PSTH of Cell with Thalamic Input')
			h_ax.set_ylabel('Spike Count')

		h_ax.bar(B[:-1], H, align='edge', width=np.diff(B)[0], alpha=0.5, label=which_cell, color=color)
		h_ax.plot([np.mean([B[i], B[i+1]]) for i in range(len(B)-1)], H, color=color)
		h_ax.axvline(0, LineStyle='--', LineWidth=1, color='k')
		h_ax.legend()
		h_ax.set_xlabel('Peri-Stimulus Time')

	def plot_FR(h_ax, FRs, normalized=False):

		mean_FRs, mean_inters = [], []
		for interval_str in FRs:
			interval = [int(j) for j in interval_str.split('[')[1].split(']')[0].split(',')]
			mean_FRs.append(np.mean(FRs[interval_str]))
			mean_inters.append(np.mean(interval))

		if normalized:
			h_ax.plot(mean_inters, [i/max(mean_FRs) for i in mean_FRs], label=which_cell, color=color)
			h_ax.set_title('Normalized Mean Firing Rate for Cell with Thalamic Input')
			h_ax.set_ylabel('Normalized Firing Rate (Hz)')

		else:
			h_ax.plot(mean_inters, mean_FRs, label=which_cell, color=color)
			h_ax.set_title('Mean Firing Rate for Cell with Thalamic Input')
			h_ax.set_ylabel('Firing Rate (Hz)')

		h_ax.axvline(0, LineStyle='--', LineWidth=1, color='k')
		h_ax.legend()
		h_ax.set_xlabel('Peri-Stimulus Time')
	
	spike_times = get_SpikeTimes(t, soma_v)
	in_window = lambda time, inter: time > inter[0] and time <= inter[1] 

	FRs, PSTH, bins = get_FR_and_PSTH(stim_times, spike_times)
	
	if not axes_h:
		_, [[PSTH_ax, PSTH_ax_norm], [FR_ax, FR_ax_norm]] = plt.subplots(2, 2)
	else:
		PSTH_ax 		= axes_h[0][0]
		PSTH_ax_norm 	= axes_h[0][1]
		FR_ax 			= axes_h[1][0]
		FR_ax_norm 		= axes_h[1][1]
	
	# PSTH
	plot_PSTH(PSTH_ax, PSTH, bins)
	plot_PSTH(PSTH_ax_norm, PSTH, bins, normalized=True)

	# FR
	plot_FR(FR_ax, FRs)
	plot_FR(FR_ax_norm, FRs, normalized=True)
	plt.gcf().subplots_adjust(hspace=0.34, bottom=0.08, top=0.9) 

	plt.suptitle('Simulation details: {}ms bins, simulation length: {}s'.format(window, int(tstop/1000)))
	
	return [[PSTH_ax, PSTH_ax_norm], [FR_ax, FR_ax_norm]]





























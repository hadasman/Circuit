import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import pdb
import math

def plotThalamicResponses(stimuli, freq1, freq2, thalamic_locations, run_function=False, axon_gids=None):
	if run_function:

		stim_ax = stimuli[freq1].axonResponses(thalamic_locations, color='red', axon_gids=axon_gids)
		stim_ax = stimuli[freq2].axonResponses(thalamic_locations, color='blue', h_ax=stim_ax, axon_gids=axon_gids)

		stimuli[freq1].tonotopic_location(thalamic_locations, color='red', h_ax=stim_ax)
		stimuli[freq2].tonotopic_location(thalamic_locations, color='blue', h_ax=stim_ax)

		stim_ax.set_title('Thalamic Axons (Connected to Chosen Pyramidal) Firing Rate for 2 Frequencies\nPyramidal GID: {}, Spontaneous Axon FR: {}'.format(chosen_pyr, 0.5))
		stim_ax.set_xlim([65, 630])

		return stim_ax

def Wehr_Zador(population, cell_name, stimulus, title_, exc_weight=0, inh_weight=0, standard_freq=None, input_pop_outputs=None, take_before=20, take_after=155, tstop=None, spike_threshold=None, dt=0.025, t=None):

	def plot_traces(h_ax, all_means, which_traces='Currents', units_='nA'):

		if which_traces == 'Currents':
			which_plot = 'i'
			unit_conversion = 1
		elif which_traces == 'Conductances':
			which_plot = 'g'
			unit_conversion = 1000

		AMPA = [i*unit_conversion for i in all_means['%s_AMPA'%which_plot]]
		NMDA = [i*unit_conversion for i in all_means['%s_NMDA'%which_plot]]			

		h_ax.axvline(0, LineStyle='--', color='gray')
		h_ax.plot(t_vec, [AMPA[i]+NMDA[i] for i in range(n_points)], 'purple', label='%s$_{AMPA}$ + %s$_{NMDA}$'%(which_plot, which_plot))

		if input_pop_outputs:
			GABA = [i*unit_conversion for i in all_means['%s_GABA'%which_plot]]
			if list(input_pop_outputs.values())[0]:
				h_ax.plot(t_vec, GABA, 'b', label='%s$_{GABA}$'%which_plot)
				h_ax.plot(t_vec, [GABA[i]+AMPA[i]+NMDA[i] for i in range(n_points)], label='%s$_{tot}$'%which_plot)

		h_ax.legend()
		h_ax.set_title('Mean Synaptic %s'%which_traces)
		h_ax.set_ylabel('%s (%s)'%(which_plot.upper(), units_))
		h_ax.set_xlim([-take_before, take_after])

	if not population:
		return

	fig, axes = plt.subplots(3, 1)
	fig.subplots_adjust(hspace=0.34, bottom=0.08, top=0.9) 
	times = [i for i in stimulus.stim_times_all if i<tstop]

	cut_vec = lambda vec, start_idx, end_idx: [vec[i] for i in range(start_idx, end_idx)]
	
	spike_count = 0
	all_means = {'i_AMPA': [], 'g_AMPA': [], 'i_NMDA': [], 'g_NMDA': [], 'i_GABA': [], 'g_GABA': []}

	print('Analyzing conductances and currents')
	for T in tqdm(times):
		# idx1 = (np.abs([i-(T-take_before) for i in t])).argmin()
		# idx2 = (np.abs([i-(T+take_after) for i in t])).argmin()
		idx1 = int((T-take_before)/dt)
		idx2 = int((T+take_after)/dt)

		if idx2>len(t):
			break

		t_vec = cut_vec(t, idx1, idx2)
		t_vec = [i-t_vec[0]-take_before for i in t_vec]
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
				for i in ['i_GABA', 'g_GABA']:
					temp_vec = [cut_vec(vec, idx1, idx2) for vec in input_pop_outputs[pre_PV][i]]
					all_sums[i].append(np.sum(temp_vec, axis=0))

		# Append the TOTAL currents and conductances, over all synapses in given cell, for the current time point
		for i in all_sums:
			all_means[i].append(np.sum(all_sums[i], axis=0))


		if T==times[0]:
			axes[0].legend(['%s soma v'%cell_name], loc='upper right')


		detect_spike = lambda vec: [i for i in range(1, len(vec)-1) if (vec[i]>vec[i-1]) and (vec[i]>vec[i+1]) and (vec[i]>spike_threshold)]
		
		idx_spont = int((take_before+30)/dt)
		idx_start_stim = int((take_before)/dt)
		if any(detect_spike(v_vec[idx_spont:])) and not any(detect_spike(v_vec[idx_start_stim:idx_spont])):
			soma_plot_color = 'r'
		else:
			soma_plot_color = 'k'
		axes[0].plot(t_vec, v_vec, color=soma_plot_color, LineWidth=0.7)

	# Average over time points
	for i in all_means:
		all_means[i] = np.mean(all_means[i][:-1], axis=0)
	t_vec = [i*dt for i in range(0, len(all_means['i_AMPA']))]
	n_points = len(t_vec)

	plt.suptitle('{} (GID: {}) Cell ({} spikes out of {})'.format(title_, population.name_to_gid[cell_name], spike_count, len(times)))
	axes[0].axvline(0, LineStyle='--', color='gray')
	axes[0].set_title('Overlay of Somatic Responses to %sHz Simulus (locked to stimulus presentation)'%standard_freq)
	axes[0].set_ylabel('V (mV)')
	axes[0].set_xlim([-take_before, take_after])
	
	# ========== Plot Currents ==========
	plot_traces(axes[1], all_means, which_traces='Currents', units_='nA')

	# ========== Plot Conductances ==========
	plot_traces(axes[2], all_means, which_traces='Conductances', units_='nS')
	axes[2].set_xlabel('T (ms)')

	return axes

def PlotSomas(populations, t, stimulus, tstop=None, spike_threshold=None, dt=0.025):
	
	if len(populations) != 3:
		return
	_, h_ax = plt.subplots(4, 1)

	all_cells, cell_names = [], []
	for cell_name in populations:
		pop_idx = list(populations.keys()).index(cell_name) + 1
		pop = populations[cell_name]
		all_cells.append(''.join([i for i in cell_name if not i.isdigit()]))
		cell_names.append(cell_name)

		temp_soma_v = pop.cells[cell_name]['soma_v']
		if len(temp_soma_v) > len(t):
			temp_soma_v = [i for i in temp_soma_v][:len(t)]
		h_ax[0].plot(t, temp_soma_v, label = '{} ({})'.format(all_cells[-1], pop.name_to_gid[cell_name]))

		h_ax[pop_idx].plot(t, temp_soma_v)
		h_ax[pop_idx].set_title('{} ({})'.format(all_cells[-1], pop.name_to_gid[cell_name]))

	for stim in [['Standard', stimulus.stim_times_standard], ['Deviant', stimulus.stim_times_deviant]]:
		times = stim[1]
		for s in times:
			if s < tstop:
				if s == times[0]:
					h_ax[0].axvline(s, LineStyle='--', color='k', alpha=0.5, label='{} Stimulus'.format(stim[0])) 
					h_ax[1].axvline(s, LineStyle='--', color='k', alpha=0.5, label='{} Stimulus'.format(stim[0]))
					h_ax[2].axvline(s, LineStyle='--', color='k', alpha=0.5, label='{} Stimulus'.format(stim[0]))
					h_ax[3].axvline(s, LineStyle='--', color='k', alpha=0.5, label='{} Stimulus'.format(stim[0]))
				else:
					h_ax[0].axvline(s, LineStyle='--', color='k', alpha=0.5)
					h_ax[1].axvline(s, LineStyle='--', color='k', alpha=0.5)
					h_ax[2].axvline(s, LineStyle='--', color='k', alpha=0.5)
					h_ax[3].axvline(s, LineStyle='--', color='k', alpha=0.5)

	h_ax[0].legend()
	h_ax[1].legend()
	h_ax[2].legend()
	h_ax[3].legend()

	title_string = ''
	for i in range(len(populations)):
		name_ = cell_names[i]
		if i==len(populations)-1:
			title_string = title_string + 'and {} ({})'.format(all_cells[i], populations[name_].name_to_gid[name_])
		else:
			title_string = title_string + '{} ({}), '.format(all_cells[i], populations[name_].name_to_gid[name_])

	h_ax[0].set_title('Example of {} Responses to 2 tones ({}Hz, {}Hz)\n(at tonotopical position between tones, standard: {})'\
					.format(title_string, min(stimulus.standard_frequency, \
							stimulus.deviant_frequency), max(stimulus.standard_frequency, \
							stimulus.deviant_frequency), stimulus.standard_frequency)) 
	
	h_ax[0].set_ylabel('V (mV)')
	h_ax[1].set_ylabel('V (mV)')
	h_ax[2].set_ylabel('V (mV)')
	h_ax[3].set_ylabel('V (mV)')
	h_ax[3].set_xlabel('T (ms)')
	h_ax[0].set_xlim([0, tstop])
	h_ax[1].set_xlim([0, tstop])
	h_ax[2].set_xlim([0, tstop])
	h_ax[3].set_xlim([0, tstop])

	return h_ax

def plotFRs(job_id, stim_times, soma_v, t, tstop=0, window=0, take_before=150, take_after=150, which_cell='', axes_h=None, color=''):
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
			# H = [i/max(H) for i in H]
			H = [i/sum(H) for i in H]
			h_ax.set_title('Normalized PSTH of Cell with Thalamic Input')
			h_ax.set_ylabel('Spike Frequency (Normalized Spike Count)')
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
			h_ax.plot(mean_inters, [i/sum(mean_FRs) for i in mean_FRs], label=which_cell, color=color)
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
		_, [[PSTH_ax, PSTH_ax_norm], [FR_ax, FR_ax_norm]] = plt.subplots(2, 2, figsize=(15, 7.5))
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

	plt.suptitle('Simulation details: {}ms bins, simulation length: {}s, job: {}'.format(window, int(tstop/1000), job_id))

	return [[PSTH_ax, PSTH_ax_norm], [FR_ax, FR_ax_norm]]




def Wehr_Zador_fromData(job_id, data, cell_name, stim_times, exc_weight=0, inh_weight=0, standard_freq=6666, take_before=20, take_after=155, tstop=None, spike_threshold=0, dt=0.025, t=None, stim_params={'type':None, 'ITI':None, 'mark_second':False}):

	def getData(data, cell_name):
		g_AMPA = data[cell_name]['inputs']['g_AMPA']
		g_NMDA = data[cell_name]['inputs']['g_NMDA']		
		i_AMPA = data[cell_name]['inputs']['i_AMPA']
		i_NMDA = data[cell_name]['inputs']['i_NMDA']
		
		soma_v = data[cell_name]['soma_v']
		gid    = data[cell_name]['gid']

		# g_GABA, i_GABA = [], []
		g_GABA, i_GABA = {}, {}
		if 'g_GABA' in data[cell_name]['inputs']:
			for pre_cell in data[cell_name]['inputs']['g_GABA']:
			
				if hasattr(data[cell_name]['inputs']['g_GABA'][pre_cell], "__len__"):

					if type(data[cell_name]['inputs']['g_GABA'][pre_cell])==list: # Old inputs were saved in individual cell keys, therefore sum was needed
						g_GABA.append(np.sum(data[cell_name]['inputs']['g_GABA'][pre_cell], axis=0))
						i_GABA.append(np.sum(data[cell_name]['inputs']['i_GABA'][pre_cell], axis=0))
					else: # This is appropriate for the current data types (separated by keys: pre_cell = 'PV', 'SOM')
						g_GABA[pre_cell] = data[cell_name]['inputs']['g_GABA'][pre_cell]
						i_GABA[pre_cell] = data[cell_name]['inputs']['i_GABA'][pre_cell]
						# g_GABA.append(data[cell_name]['inputs']['g_GABA'][pre_cell])
						# i_GABA.append(data[cell_name]['inputs']['i_GABA'][pre_cell])

			# g_GABA = np.sum(g_GABA, axis=0)
			# i_GABA = np.sum(i_GABA, axis=0)
		
		return g_AMPA, g_NMDA, i_AMPA, i_NMDA, soma_v, gid, g_GABA, i_GABA

	def plot_traces(t_vec, n_points, h_ax, all_means, which_traces='Currents', units_='nA'):

		if which_traces == 'Currents':
			which_plot = 'i'
			unit_conversion = 1
			MAXs = max_i_AMPA + max_i_NMDA
			MINs = min_i_AMPA + min_i_NMDA
		elif which_traces == 'Conductances':
			which_plot = 'g'
			unit_conversion = 1000
			MAXs = max_g_AMPA + max_g_NMDA
			MINs = min_g_AMPA + min_g_NMDA


		AMPA = [i*unit_conversion for i in all_means['%s_AMPA'%which_plot]]
		NMDA = [i*unit_conversion for i in all_means['%s_NMDA'%which_plot]]			

		minimize = lambda L: [L[i] for i in range(len(L)) if i%2==0]

		h_ax.axvline(0, LineStyle='--', color='gray')
		h_ax.plot(t_vec, [AMPA[i]+NMDA[i] for i in range(n_points)], 'purple', label='%s$_{AMPA}$ + %s$_{NMDA}$'%(which_plot, which_plot))

		if 'g_GABA' in all_means:
			GABA = {}

			alpha_ = 1
			for pre_cell in all_means['%s_GABA'%which_plot]:

				GABA[pre_cell] = [i*unit_conversion for i in all_means['%s_GABA'%which_plot][pre_cell]]
				h_ax.plot(t_vec, GABA[pre_cell], 'b', label='%s$_{GABA}$ %s'%(which_plot, pre_cell), alpha=alpha_)
				alpha_ = 0.5

			GABA_tot = np.sum([GABA[i] for i in GABA], axis=0)
			h_ax.plot(t_vec, [GABA_tot[i]+AMPA[i]+NMDA[i] for i in range(n_points)], label='%s$_{tot}$'%which_plot)			

		h_ax.legend(loc='upper right')
		h_ax.set_title('Mean Synaptic %s'%which_traces)
		h_ax.set_ylabel('%s (%s)'%(which_plot.upper(), units_))
		h_ax.set_xlim([-take_before, take_after])

	fig, axes = plt.subplots(3, 1, figsize=(9, 7.5))
	fig.subplots_adjust(hspace=0.34, bottom=0.08, top=0.9) 

	g_AMPA, g_NMDA, i_AMPA, i_NMDA, soma_v, gid, g_GABA, i_GABA = getData(data, cell_name)	

	times = [i[0] for i in stim_times if i[0]<tstop]

	cut_vec = lambda vec, start_idx, end_idx: [vec[i] for i in range(start_idx, end_idx)]
	
	spike_count = 0
	all_means = {'i_AMPA': [], 'g_AMPA': [], 'i_NMDA': [], 'g_NMDA': []}
	if hasattr(g_GABA, "__len__"):
		if len(g_GABA) > 0:
			all_means['g_GABA'] = {pre_cell: [] for pre_cell in g_GABA}
			all_means['i_GABA'] = {pre_cell: [] for pre_cell in i_GABA}

	print('Analyzing conductances and currents')

	if take_after+times[-1]>t[-1]: times = times[:-1] # If take_after goes beyond recording, discard last stimulus

	for T in tqdm(times):
		# idx1 = (np.abs([i-(T-take_before) for i in t])).argmin()
		# idx2 = (np.abs([i-(T+take_after) for i in t])).argmin()
		
		idx1 = int((T-take_before)/dt)
		idx2 = int((T+take_after)/dt)
		
		t_vec = cut_vec(t, idx1, idx2)

		t_vec = [i-t_vec[0]-take_before for i in t_vec]
		v_vec = cut_vec(soma_v, idx1, idx2)
		if any([i>=spike_threshold for i in v_vec]):
			spike_count += 1

		all_means['g_AMPA'].append(cut_vec(g_AMPA, idx1, idx2))
		all_means['g_NMDA'].append(cut_vec(g_NMDA, idx1, idx2))
		all_means['i_AMPA'].append(cut_vec(i_AMPA, idx1, idx2))
		all_means['i_NMDA'].append(cut_vec(i_NMDA, idx1, idx2))

		if 'g_GABA' in all_means:
			for pre_cell in g_GABA:				
				all_means['g_GABA'][pre_cell].append(cut_vec(g_GABA[pre_cell], idx1, idx2))
				all_means['i_GABA'][pre_cell].append(cut_vec(i_GABA[pre_cell], idx1, idx2))


		soma_plot_color = 'k'
		if stim_params['mark_second']:
			detect_spike = lambda vec: [i for i in range(1, len(vec)-1) if (vec[i]>vec[i-1]) and (vec[i]>vec[i+1]) and (vec[i]>spike_threshold)]

			if stim_params['type']=='pairs':
				try: axes[0].axvline(stim_params['ITI'], color='gray', LineStyle='--', LineWidth=0.7)
				except: print('In Wehr_Zador_fromData(): not marking 2nd stimulus on plot because no ITI given to function')

				idx_first_stim = int((take_before)/dt)
				idx_end_first_stim = int((take_before+50)/dt)
				idx_second_stim = int((take_before+stim_params['ITI'])/dt)
				idx_end_second_stim = int((take_before+stim_params['ITI']+50)/dt)
				if any(detect_spike(v_vec[idx_second_stim:idx_end_second_stim])):
					if any(detect_spike(v_vec[idx_first_stim:idx_end_first_stim])):
						soma_plot_color = 'r'
					else:
						soma_plot_color = 'g'

			elif stim_params['type']=='single':

				late_start = int((take_before+40)/dt)
				late_end = int((take_before+100)/dt)
				stim_end = int((take_before+30)/dt)
				stim_start = int(take_before/dt)
				if any(detect_spike(v_vec[late_start:late_end])) and not any(detect_spike(v_vec[stim_start:stim_end])):
					soma_plot_color = 'r'			
				elif any(detect_spike(v_vec[late_start:late_end])) and any(detect_spike(v_vec[stim_start:stim_end])):
					soma_plot_color = 'k'
				elif not any(detect_spike(v_vec[late_start:late_end])) and any(detect_spike(v_vec[stim_start:stim_end])):
					soma_plot_color = 'g'
				else:
					soma_plot_color = 'orange'
		axes[0].plot(t_vec, v_vec, color=soma_plot_color, LineWidth=0.7)
				
	# Average over time points
	max_g_AMPA = np.max(all_means['g_AMPA'][:-1], axis=0); min_g_AMPA = np.min(all_means['g_AMPA'][:-1], axis=0)
	max_g_NMDA = np.max(all_means['g_NMDA'][:-1], axis=0); min_g_NMDA = np.min(all_means['g_NMDA'][:-1], axis=0)
	max_i_AMPA = np.max(all_means['i_AMPA'][:-1], axis=0); min_i_AMPA = np.min(all_means['i_AMPA'][:-1], axis=0)
	max_i_NMDA = np.max(all_means['i_NMDA'][:-1], axis=0); min_i_NMDA = np.min(all_means['i_NMDA'][:-1], axis=0)

	for i in all_means:		
		if 'GABA' in i:
			for j in all_means[i]:
				all_means[i][j] = np.mean(all_means[i][j][:-1], axis=0)
		else:
			all_means[i] = np.mean(all_means[i][:-1], axis=0)

	t_vec_tot = [(i*dt)-take_before for i in range(0, len(all_means['i_AMPA']))]	
	n_points = len(t_vec_tot)

	title_ = cell_name.split([i for i in cell_name if i.isdigit()][0])[0]
	plt.suptitle('{} (GID: {}) Cell ({} spikes out of {}), Job {}'.format(title_, gid, spike_count, len(times), job_id))
	axes[0].axvline(0, LineStyle='--', color='gray')
	axes[0].set_title('Overlay of Somatic Responses to %sHz Simulus (locked to stimulus presentation)'%standard_freq)
	axes[0].set_ylabel('V (mV)')
	axes[0].set_xlim([-take_before, take_after])
	axes[0].plot(0, 0, 'r', label='Spike only late')
	axes[0].plot(0, 0, 'k', label='Both evoked and late spike')
	axes[0].plot(0, 0, 'g', label='Spike only evoked')
	axes[0].plot(0, 0, 'orange', label='No spikes')
	axes[0].legend()
	
	# ========== Plot Currents ==========
	plot_traces(t_vec_tot, n_points, axes[1], all_means, which_traces='Currents', units_='nA')

	# ========== Plot Conductances ==========
	plot_traces(t_vec_tot, n_points, axes[2], all_means, which_traces='Conductances', units_='nS')
	axes[2].set_xlabel('T (ms)')

	return axes

def PlotSomas_fromData(DATAs, t, stim_times_standard=None, standard_freq=None, stim_times_deviant=None, deviant_freq=None, tstop=None, spike_threshold=None, dt=0.025):
	
	fig, h_ax = plt.subplots(figsize=(15, 7.5))
	fig.subplots_adjust(hspace=0.48, bottom=0.08, top=0.91, left=0.1, right=0.95) 

	title_string = ''

	for cell_name in DATAs:
		
		data = DATAs[cell_name]
		name_ = ''.join([i for i in cell_name if not i.isdigit()])

		if title_string.count('(')==2:
			title_string = title_string + 'and {} ({})'.format(name_, DATAs[cell_name][cell_name]['gid'])
		else:
			title_string = title_string + '{} ({}), '.format(name_, DATAs[cell_name][cell_name]['gid'])

		temp_soma_v = data[cell_name]['soma_v']
		if len(temp_soma_v) > len(t):
			temp_soma_v = [i for i in temp_soma_v][:len(t)]
		h_ax.plot(t, temp_soma_v, label = '{} ({})'.format(name_, data[cell_name]['gid']))

	if stim_times_standard and stim_times_deviant:
		assert tstop is not None, 'No tstop argument!'
		for T in stim_times_standard:
			if T < tstop:
				if T == stim_times_standard[0]:
					h_ax.axvline(T, LineStyle='--', color='gray', alpha=0.5, label='{} Stimulus'.format(standard_freq)) 
				else:
					h_ax.axvline(T, LineStyle='--', color='gray', alpha=0.5)
		for T in stim_times_deviant:
			if T < tstop:
				if T == stim_times_deviant[0]:
					h_ax.axvline(T, LineStyle='--', color='green', alpha=0.5, label='{} Stimulus'.format(deviant_freq)) 
				else:
					h_ax.axvline(T, LineStyle='--', color='green', alpha=0.5)
	h_ax.legend()

	fig.suptitle('Example of {} Responses to 2 tones ({}Hz, {}Hz)'.format(title_string, min(standard_freq, deviant_freq), max(standard_freq, deviant_freq)), size=12)
	h_ax.set_title('(at tonotopical position between tones, standard: {})'.format(standard_freq)) 
	h_ax.set_xlabel('T (ms)')
	h_ax.set_ylabel('V (mV)')
	h_ax.set_xlim([0, tstop])

	return h_ax

def PlotSomaSubplots_fromData(DATAs, t, stim_times_standard=None, standard_freq=None, stim_times_deviant=None, deviant_freq=None, tstop=None, spike_threshold=None, dt=0.025):
	
	fig, h_ax = plt.subplots(4, 1, figsize=(15, 7.5))
	fig.subplots_adjust(hspace=0.48, bottom=0.08, top=0.91, left=0.1, right=0.95) 

	title_string = ''

	for cell_name in DATAs:
		
		pop_idx = list(DATAs.keys()).index(cell_name) + 1

		data = DATAs[cell_name]
		name_ = ''.join([i for i in cell_name if not i.isdigit()])

		if title_string.count('(')==2:
			title_string = title_string + 'and {} ({})'.format(name_, DATAs[cell_name][cell_name]['gid'])
		else:
			title_string = title_string + '{} ({}), '.format(name_, DATAs[cell_name][cell_name]['gid'])

		temp_soma_v = data[cell_name]['soma_v']
		if len(temp_soma_v) > len(t):
			temp_soma_v = [i for i in temp_soma_v][:len(t)]
		h_ax[0].plot(t, temp_soma_v, label = '{} ({})'.format(name_, data[cell_name]['gid']))
		
		h_ax[pop_idx].plot(t, temp_soma_v)
		h_ax[pop_idx].set_title('{} ({})'.format(name_, data[cell_name]['gid']))


	if stim_times_standard and stim_times_deviant:
		assert tstop is not None, 'No tstop argument!'

		for stim in[[standard_freq, stim_times_standard], [deviant_freq, stim_times_deviant]]:
			
			times = stim[1]
			for T in times:
				if T < tstop:
					if T  == times[0]:						
						h_ax[0].axvline(T, LineStyle='--', color='gray', alpha=0.5, label='{} Stimulus'.format(stim[0])) 
						h_ax[1].axvline(T, LineStyle='--', color='gray', alpha=0.5, label='{} Stimulus'.format(stim[0])) 
						h_ax[2].axvline(T, LineStyle='--', color='gray', alpha=0.5, label='{} Stimulus'.format(stim[0])) 
						h_ax[3].axvline(T, LineStyle='--', color='gray', alpha=0.5, label='{} Stimulus'.format(stim[0])) 
					else:
						h_ax[0].axvline(T, LineStyle='--', color='gray', alpha=0.5)
						h_ax[1].axvline(T, LineStyle='--', color='gray', alpha=0.5)
						h_ax[2].axvline(T, LineStyle='--', color='gray', alpha=0.5)
						h_ax[3].axvline(T, LineStyle='--', color='gray', alpha=0.5)

	h_ax[0].legend()
	h_ax[1].legend()
	h_ax[2].legend()
	h_ax[3].legend()

	fig.suptitle('Example of {} Responses to 2 tones ({}Hz, {}Hz)'.format(title_string, min(standard_freq, deviant_freq), max(standard_freq, deviant_freq)), size=12)
	h_ax[0].set_title('(at tonotopical position between tones, standard: {})'.format(standard_freq)) 
	h_ax[0].set_ylabel('V (mV)')
	h_ax[1].set_ylabel('V (mV)')
	h_ax[2].set_ylabel('V (mV)')
	h_ax[3].set_ylabel('V (mV)')
	h_ax[3].set_xlabel('T (ms)')
	h_ax[0].set_xlim([0, tstop])
	h_ax[1].set_xlim([0, tstop])
	h_ax[2].set_xlim([0, tstop])
	h_ax[3].set_xlim([0, tstop])

	return h_ax























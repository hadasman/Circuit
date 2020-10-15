# Tone presntentation 
# Created by Oren Amsalem oren.a4@gmail.com | The Hebrew University

from __future__ import division
from math import *
from numpy import *
from scipy.optimize import least_squares
import os, multiprocessing, time
import _pickle as cPickle
import numpy as np
from matplotlib.cbook import flatten
from tqdm import tqdm
import _pickle as cPickle

from worker_functions import *
from SSA_Stim import *

flatten = lambda L: [j for i in L for j in i]

def CreatseExpDict():
    #Read Data from fit!
    amp_to_dist_vals = {60: {'tun_width_dist': 'exp', 
                            'w_exp_scale':0.9, 
                            'w_exp_loc': 0.24, 
                            
                            'fr_dist':'gaussian', 
                            'fr_guas_mean': 440, 
                            'fr_gaus_std':175, 
                            
                            'alph_func_dist':'gaussian', 
                            'alph_tau_mean': 2.3, 
                            'alph_tau_std':0.88,
                            
                            'delay_gaus_norm_f': 6.53, 
                            'delay_gaus_std':0.672, 
                            'base_line': 20.8}}

    amp = 60
    experiment = {
                  'SSAType': 'BroadBand_pair',
                  'BroadBand': False, 
                  'first_marker_duration': 120,
                  'first_marker_minimal_FR': 40,    
                  'inter_tone_interval': 140, # marker onset-to-onset     

                  'axons_settings': [AxoGidToXY],
                  'prefered_frequency_distribution': 'tonotopic',
                  'prefered_frequency_distribution_shuffle_percent': 0,
                  'ssa_presentations': 250,
                  'ssa_standard_probability': 0.95,
                  'ssa_presentations_seed': 1,
                  'prefered_frequency_boundaries': [4000, 16000], # 2 octvaes

                  'tuning_curve_type': 'triangle', # Options: 'exp', 'triangle'
                  
                  'stim_type': 'alpha_func',
                  'duration_of_stim': 2000,
                  'duration_between_stims': 0,  #End to start
                  'responsive_axons_probability': float(0.58),
                  'spontaneous_firing_rate_type': 'constant', #When I have sontaneuos firing rate, it will be added to the osi firing rate, which means that the firing rate will be higher than the one which was set in the 
                  'spontaneous_firing_rate_value': 1.5,
                  'center_frequencies': (6666, 9600),
                  'circuit_target': "mc2_Column",
                  
                  'first_stimulus_time': 2000,
                  
                  'tunning_width_distribution': amp_to_dist_vals[amp]['tun_width_dist'],
                  'tunning_width_exp_scale': amp_to_dist_vals[amp]['w_exp_scale'],
                  'tunning_width_exp_loc': amp_to_dist_vals[amp]['w_exp_loc'],
                  'axon_stim_firing_rate_distribution': amp_to_dist_vals[amp]['fr_dist'],
                  'axon_stim_firing_rate_gaussian_mean': amp_to_dist_vals[amp]['fr_guas_mean'],
                  'axon_stim_firing_rate_gaussian_std': amp_to_dist_vals[amp]['fr_gaus_std'],
                  'alpha_func_tau_distribution': amp_to_dist_vals[amp]['alph_func_dist'],
                  'alpha_func_tau_gaussian_mean': amp_to_dist_vals[amp]['alph_tau_mean'],
                  'alpha_func_tau_gaussian_std': amp_to_dist_vals[amp]['alph_tau_std'],
                  'alpha_func_delay_distribution': 'frequency_correlated',
                  'delays_gaus_norm_factor': amp_to_dist_vals[amp]['delay_gaus_norm_f'],
                  'delays_gaus_std': amp_to_dist_vals[amp]['delay_gaus_std'],
                  'delays_gaus_base_line': float(18) ## Oren: I remove 10 ms so I will not need to fix the concept that each presentation is 30 ms!
                }

    return  experiment

def SSA_protocol_stimulations(experiment):

    StandPro                = experiment['ssa_standard_probability']
    DeviPro                 = round(1 - StandPro,3)
    Standard, Deviant = experiment['stand_dev_couple'][0], experiment['stand_dev_couple'][1]
    stimulus = SSA_Stim(experiment['SSAType'], Standard, Deviant, StandPro, DeviPro, experiment['ssa_presentations'])

    if experiment['SSAType'] in ['StandardDeviant', 'Equal']: #Two Stims
        if experiment['SSAType'] =='Equal':
            if Deviant < Standard: 
                raise Exception("Deviant should be bigger than Standard, just a def")
            elif StandPro!=0.5:
                raise Exception('If equal standPro should be 0.5')

        AudStims = stimulus.StandardDeviant()

    elif experiment['SSAType'] == 'PairedStimsOneFreq':
        AudStims = stimulus.OneFreq()

    elif 'BroadBand' in experiment['SSAType']:
      experiment['BroadBand'] = True
      AudStims = stimulus.OneFreq()

    return(AudStims)

# Keep this function definition in global workspace so multiprocessing has access to it
def worker_job(gid_seed):
    '''
    worker for the parallel 
    '''
    axon_gid, Base_Seed, responsive = gid_seed
    axon_activity_vars = {axon_gid: {}}

    axon_activity_vars[axon_gid]['prefered_freq'], axon_activity_vars[axon_gid]['shuffled_neuron'] = set_pref_frequency(axon_gid, experiment)
    
    axon_activity_vars[axon_gid]['axon_stim_firing_rate']   = get_axon_stim_firing_rate(axon_gid, experiment)

    axon_activity_vars[axon_gid]['width_value']             = get_tuning_width(axon_gid, experiment)

    axon_activity_vars[axon_gid]['duration_of_stim']        = get_duration_of_stims(experiment['duration_of_stim'])

    axon_activity_vars[axon_gid]['duration_between_stims']  = experiment['duration_between_stims']
    axon_activity_vars[axon_gid]['first_stimulus_time']     = experiment['first_stimulus_time']

    axon_activity_vars[axon_gid]['alpha_func_delay'], axon_activity_vars[axon_gid]['alpha_func_tau'] = get_alpha_params(axon_gid, experiment)

    axon_activity_vars[axon_gid]['inter_tone_interval'] = experiment['inter_tone_interval']

    if not responsive: #disable active fr of axon
        axon_activity_vars[axon_gid]['axon_stim_firing_rate'] = 1.0/9e9

    axon_activity_vars[axon_gid]['time_to_firing_rate_frequency'] =  create_time_to_spike_frequency_func(experiment, axon_activity_vars[axon_gid])
    

    axon_activity_vars[axon_gid]['spike_times'] = np.array(get_spike_times(rnd_stream = np.random.RandomState([int(axon_gid),Base_Seed]) , 
                                                                           time_to_firing_rate_frequency = axon_activity_vars[axon_gid]['time_to_firing_rate_frequency'],
                                                                           stim_end_time = experiment['simulation_end_time']))
    
    # need to delete for multiprocessing
    axon_activity_vars[axon_gid]['time_to_firing_rate_frequency']   = []
    axon_activity_vars[axon_gid]['alpha_func_delay']                = []
    axon_activity_vars[axon_gid]['duration_of_stim']                = [] 
    
    return(axon_gid, axon_activity_vars)

def create_axons_spikes(stand_freq, seed, experiment, save_path=None, save_name=None):
    '''
    save_path should be the link to the spike file to which you want to write the spike_times
    '''
    axon_activity_vars = {}
    axons_gids = experiment['axons_settings'][0].keys()

    #prepare data to send to threads
    data_to_send = []
    for axon_gid in axons_gids:
        rnd_responsive = np.random.RandomState(int(axon_gid))
        is_responsive = int(rnd_responsive.uniform() <= experiment['responsive_axons_probability'])
        data_to_send.append([axon_gid, seed, is_responsive])
 
    # send to threads and merge results
    pool = multiprocessing.Pool()
    for i in list(tqdm(pool.imap(worker_job, data_to_send), total=len(data_to_send))):
        axon_activity_vars[i[0]] = i[1][i[0]]
    
    if save_path:
        assert save_name
        print(save_path)
        dat_filename = save_path +'/' + save_name + '.dat'
        f = open(dat_filename,'w')
        f.write('/scatter\n')
        sps = [zip(*[axon_activity_vars[axon_gid]['spike_times'],[axon_gid]*len(axon_activity_vars[axon_gid]['spike_times'])]) for axon_gid in axon_activity_vars]
        sps = sorted([i for j in sps for i in j])
        [f.write('{:} \t {:} \n'.format(spike_time,axon_gid)) for spike_time,axon_gid in sps]
        f.close()

        # Save also as pickle        
        temp_data = [i.strip().split() for i in open(dat_filename).readlines()] 
        data = [] 
        for i in range(len(temp_data)): 
            if temp_data[i][0].replace('.', '').isdigit():
                data.append([float(temp_data[i][0]), int(float(temp_data[i][1]))]) 
        
        activations = {} 
        for i in data: 
            time=i[0] 
            axon=i[1] 
            if axon in activations: 
                activations[axon].append(time) 
            else: 
                activations[axon] = [time] 

        pic_filename =  save_path +'/' + save_name + '.p'
        # cPickle.dump(activations, open(pic_filename,'wb')) 

    data_to_save = {'SSAType': experiment['SSAType'],
                    'IPI': experiment['duration_of_stim'],
                    'ITI': experiment['inter_tone_interval']*('pair' in experiment['SSAType'] or 'Pair' in experiment['SSAType']), # 0 if single tone
                    'activations': activations,
                    'frequency': stand_freq}

    # return(spike_times_txt, axon_activity_vars)
    return axon_activity_vars, data_to_save


###Example Run -
if __name__ == '__main__' :

    print('Presentations changed to 250, change back to 500!')

    save_path = 'thalamocortical_Oren/CreateThalamicInputs/test_times'#directory to save the files in 

    AxoGidToXY = cPickle.load(open('thalamocortical_Oren/thalamic_data/AxoGidToXY.p','rb'), encoding='latin1') #<<--- fix this path!
    
    experiment = CreatseExpDict()
    simulation_time = "20:00:00" # real run time

    seed = 1

    experiment['simulation_end_time'] = (experiment['duration_between_stims'] + experiment['duration_of_stim'])*experiment['ssa_presentations'] + experiment['first_stimulus_time']
    simulation_duration = experiment['simulation_end_time']   # bilogical time

    def run_experiment(Standard, Deviant, BS, spike_replay_name):
        # spike_replay_name: Path to save spike times

        experiment['stand_dev_couple'] = [Standard, Deviant]
        experiment['frequencies']      = SSA_protocol_stimulations(experiment)

        axon_activity_vars, data_to_save = create_axons_spikes(Standard, BS, experiment, save_path=save_path, save_name =spike_replay_name )

        return experiment['frequencies'], data_to_save

    times = range(experiment['first_stimulus_time'], experiment['simulation_end_time'], experiment['duration_of_stim']) 
    times = flatten([[i, i+experiment['inter_tone_interval']] for i in times])

    stim_times = {}

    os.system("osascript -e \'display notification \"Input filename!\" with title \"User input required\" sound name \"Submarine\"\'")
    user_input = input("How to call the file? (inter-tone interval = {}, inter-pair interval = {}) ".format(experiment['inter_tone_interval'], experiment['duration_of_stim']))


    if 'BroadBand' in experiment['SSAType']:
        specific_filename = 'inputBroadBand_{}'.format(user_input)

        _, data_to_save = run_experiment('BroadBand', 'BroadBand', 106746, specific_filename)
        data_to_save['stim_times'] = [[i, 'BB'] for i in times]
        cPickle.dump(data_to_save, open(save_path + '/' + specific_filename + '.p', 'wb'))
    
    else:
        specific_filename = 'input6666_{}'.format(user_input)

        stim_order, data_to_save_6666 = run_experiment(6666, 9600, 106746, specific_filename)
        data_to_save_6666['stim_times'] = [[times[i], stim_order[i]] for i in range(len(times))]
        cPickle.dump(data_to_save_6666, open(save_path + '/' + specific_filename + '.p', 'wb'))

        # stim_order, data_to_save_9600 = run_experiment(9600, 6666, 109680, 'input9600_{}'.format(specific_filename))
        # data_to_save_9600['stim_times'] = [[times[i], stim_order[i]] for i in range(len(times))]
        # cPickle.dump(data_to_save_9600, open(save_path + '/' + specific_filename + '.p', 'wb'))

    
    













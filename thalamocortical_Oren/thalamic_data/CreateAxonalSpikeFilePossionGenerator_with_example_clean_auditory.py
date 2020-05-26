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

from worker_functions import *
from SSA_Stim import *

def SSA_protocol_stimulations(experiment, SSA):

    ssa_presentations_seed = experiment['ssa_presentations_seed']
    StandDeviCouples    = [experiment['stand_dev_couple']]
    SSAType             = experiment['SSAType']
    StandPro            = experiment['ssa_standard_probability']
    Presentations       = experiment['ssa_presentations']        

    if not SSA:
        stimulus = SSA_Stim(SSA)
        NoSSA(ssa_presentations_seed)
        return(AudStims)
    
    elif SSA:      
        #ISI = 300
        DeviPro  = round(1 - StandPro,3)
        for Standard, Deviant in StandDeviCouples:
            stimulus = SSA_Stim(SSAType, Standard, Deviant, StandPro, DeviPro, Presentations)

            if SSAType in ['TwoStims', 'Equal']: #Two Stims
                if SSAType =='Equal':
                    if Deviant < Standard: 
                        raise Exception("Deviant should be bigger than Standard, just a def")
                    elif StandPro!=0.5:
                        raise Exception('If equal standPro should be 0.5')

                AudStims = stimulus.TwoStims()
            
            elif SSAType == 'DeviantAlone': #Standard will be the deviant!
                AudStims = stimulus.DeviantAlone()
                raise Exception('I do not get this next step- maybe mistake and shoulf replace all standards instead of all deviants?')
                AudStims = [i if i==Standard else 900000 for i in  AudStims] #replace all standards in a frequancy that is so high it will not create any spikes.
            
            elif SSAType == 'TwoStimsPeriodic':
                AudStims = stimulus.TwoStimsPeriodic()
            
            elif SSAType == 'TwoStimsPeriodicP9_29':
                AudStims = stimulus.TwoStimsPeriodicP9_29()
            
            elif SSAType == 'DiverseNarrowT': #Diverse Narrow (equal amount of tones, spaced by 0.2 of the octave between standard and deviant; ranging from 2 under lowest and 2 above highest)
                AudStims = stimulus.DiverseNarrow_T_or_Exp(5, 2)
            
            elif SSAType =='DiverseNarrowExp'  : #Diverse Narrow  -- Real experimental option.
                AudStims = stimulus.DiverseNarrow_T_or_Exp(11, 4)

            elif SSAType in ['DiverseBroadTR', 'DiverseBroadExpR', 'DiverseBroadExp'] : #Diverse Broad  -- the most reasonable option for one column
                
                if SSAType == 'DiverseBroadTR':     DiverseBroadProbs = [0.2, 0.2, 0.1, 0.1, 0.2, 0.2] ## This option is in order to compare the Tohar simulations
                elif SSAType == 'DiverseBroadExpR': DiverseBroadProbs = [0.226, 0.224, 0.05, 0.05, 0.226, 0.224] ## This option is in order to compare to the experiments 
                elif SSAType == 'DiverseBroadExp':  DiverseBroadProbs = [0.09, 0.09, 0.09, 0.09, 0.09, 0.05, 0.05, 0.09, 0.09, 0.09, 0.09, 0.09]
                
                AudStims = stimulus.DiverseBroad(DiverseBroadProbs, SSAType)
            
            elif 'DiverseBroad_' in SSAType:

                number_of_sims = int(SSAType.split('_')[1])
                assert int(((number_of_sims-2)/2)) == ((number_of_sims-2)/2), 'check here - /jupnotepads/SSA_analysis/SSA_figures/Thalamic_input_DiverseBroad_after_eli_commens.ipynb'
                n_extend = (number_of_sims - 2) / 2

                if number_of_sims == 6 and Presentations == 500 and DeviPro==0.05:
                    DiverseBroadProbs = [0.224, 0.226, 0.05, 0.05, 0.226, 0.224]
                
                elif number_of_sims == 6 and Presentations == 100 and DeviPro==0.05:
                    DiverseBroadProbs = [0.22, 0.23, 0.05, 0.05, 0.23, 0.22]
                
                else:
                    DeviRepetitions = 2
                    other_probs = [(1-DeviPro*DeviRepetitions)/(n_extend*2)] * int(n_extend)
                    DiverseBroadProbs = other_probs + [DeviPro]*DeviRepetitions + other_probs
                
                AudStims = stimulus.DiverseBroad(DiverseBroadProbs, SSAType, n_extend=n_extend, d_octave=1)
                
                assert len(AudStims) == Presentations, 'DiverseBroadProbs[x]*Presentations is not an int!'
                assert Standard in AudStims, 'Standard ({}Hz) not in AudStims'.format(Standard)
                assert Deviant in AudStims, 'Deviant ({}Hz) not in AudStims'.format(Deviant)
                
                print(set(AudStims))

            # If stimulus should be random, shuffle
            if SSAType not in ['TwoStimsPeriodic','TwoStimsPeriodicP9_29']:
                rndd = np.random.RandomState(ssa_presentations_seed)
                rndd.shuffle(AudStims)
                rndd.shuffle(AudStims)
                rndd.shuffle(AudStims)
                rndd.shuffle(AudStims)
                        
            return(AudStims)

# Keep this function definition in global workspace so multiprocessing has access to it
def worker_job(gid_seed):
    '''
    worker for the parallel 
    '''
    axon_gid, Base_Seed, responsive = gid_seed
    axon_activity_vars = {axon_gid: {}}

    if experiment['stim_type'] in ['step', 'alpha_func']:
        axon_activity_vars[axon_gid]['prefered_freq'], \
        axon_activity_vars[axon_gid]['shuffled_neuron']         = set_pref_frequency(axon_gid, experiment)
        
        axon_activity_vars[axon_gid]['axon_stim_firing_rate']   = get_axon_stim_firing_rate(axon_gid, experiment)

        axon_activity_vars[axon_gid]['width_value']             = get_tuning_width(axon_gid, experiment)

        axon_activity_vars[axon_gid]['duration_of_stim']        = get_duration_of_stims(experiment['duration_of_stim'])

        axon_activity_vars[axon_gid]['duration_between_stims']  = experiment['duration_between_stims']
        axon_activity_vars[axon_gid]['first_stimulus_time']     = experiment['first_stimulus_time']
   
    if experiment['stim_type'] in ['alpha_func']:
        axon_activity_vars[axon_gid]['alpha_func_delay'], axon_activity_vars[axon_gid]['alpha_func_tau'] = get_alpha_params(axon_gid, experiment)
   
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

def create_axons_spikes(seed, experiment, save_path=None, save_name=None):
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
        f = open(save_path +'/' + save_name,'w')
        f.write('/scatter\n')
        sps = [zip(*[axon_activity_vars[axon_gid]['spike_times'],[axon_gid]*len(axon_activity_vars[axon_gid]['spike_times'])]) for axon_gid in axon_activity_vars]
        sps = sorted([i for j in sps for i in j])
        [f.write('{:} \t {:} \n'.format(spike_time,axon_gid)) for spike_time,axon_gid in sps]
        f.close()

        #Some new saving from more data
        # look here if you like to save the axons parameters
        #pickle.dump(experiment,open(save_path +'/exp' + save_name,'wb'))
    
    # return(spike_times_txt, axon_activity_vars)
    return axon_activity_vars




###Example Run -
if __name__ == '__main__' :

    AxoGidToXY = cPickle.load(open('thalamocortical_Oren/thalamic_data/AxoGidToXY.p','rb'), encoding='latin1') #<<--- fix this path!
    experiment = {'axons_settings': [AxoGidToXY],
                  'prefered_frequency_distribution': 'tonotopic',
                  'prefered_frequency_distribution_shuffle_percent': 0,
                  'ssa_presentations': 500,
                  'ssa_standard_probability': 0.95,
                  'ssa_presentations_seed': 1,
                  'prefered_frequency_boundaries': [4000, 16000], # 2 octvaes

                  'tuning_curve_type': 'triangle', # Options: 'exp', 'triangle'
                  
                  'stim_type': 'alpha_func',
                  'duration_of_stim': 300,
                  'duration_between_stims': 0,  #End to start
                  'responsive_axons_probability': float(0.58),
                  'spontaneous_firing_rate_type': 'constant', #When I have sontaneuos firing rate, it will be added to the osi firing rate, which means that the firing rate will be higher than the one which was set in the 
                  'spontaneous_firing_rate_value': 1.5,
                  'center_frequencies': (6666, 9600),
                  'circuit_target': "mc2_Column",
                  
                  'SSAType': 'TwoStims',
                  'first_stimulus_time': 2000
                  }

    simulation_time = "20:00:00" # real run time

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

    ### #set values for amp
    ##--------
    def set_exp_by_amp(amp):
        experiment['tunning_width_distribution']       = amp_to_dist_vals[amp]['tun_width_dist']
        experiment['tunning_width_exp_scale']          = amp_to_dist_vals[amp]['w_exp_scale']
        experiment['tunning_width_exp_loc']            = amp_to_dist_vals[amp]['w_exp_loc']

        experiment['axon_stim_firing_rate_distribution']       = amp_to_dist_vals[amp]['fr_dist']
        experiment['axon_stim_firing_rate_gaussian_mean']       = amp_to_dist_vals[amp]['fr_guas_mean']
        experiment['axon_stim_firing_rate_gaussian_std']       = amp_to_dist_vals[amp]['fr_gaus_std']

        experiment['alpha_func_tau_distribution'] = amp_to_dist_vals[amp]['alph_func_dist']
        experiment['alpha_func_tau_gaussian_mean'] = amp_to_dist_vals[amp]['alph_tau_mean']
        experiment['alpha_func_tau_gaussian_std']  = amp_to_dist_vals[amp]['alph_tau_std']

        experiment['alpha_func_delay_distribution'] = 'frequency_correlated'
        experiment['delays_gaus_norm_factor'] = amp_to_dist_vals[amp]['delay_gaus_norm_f']
        experiment['delays_gaus_std']         = amp_to_dist_vals[amp]['delay_gaus_std']
        experiment['delays_gaus_base_line'] = float(18) ## I remove 10 ms so I will not need to fix the concept that each presentation is 30 ms!
    set_exp_by_amp(60)
    ##--------

    SSA = 1
    seed = 1

    experiment['simulation_end_time'] = (experiment['duration_between_stims'] + experiment['duration_of_stim'])*experiment['ssa_presentations'] + experiment['first_stimulus_time']
    simulation_duration = experiment['simulation_end_time']   # bilogical time
    save_path = 'test_times'#directory to save the files in 

    def run_experiment(Standard, Deviant, BS, spike_replay_name):
        # spike_replay_name: Path to save spike times

        experiment['stand_dev_couple'] = [Standard, Deviant]
        experiment['frequencies']      = SSA_protocol_stimulations(experiment, SSA)
        axon_activity_vars          = create_axons_spikes(BS, experiment, save_path=save_path, save_name =spike_replay_name )

    run_experiment(6666, 9600, 106746, 'input6666_106746_3.dat')
    run_experiment(9600, 6666, 109680, 'input9600_109680_3.dat')






# Not available yet?



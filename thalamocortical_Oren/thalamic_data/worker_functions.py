import numpy as np
from math import *
from numpy import *
import pdb
            
def set_pref_frequency(axon_gid, experiment):
    '''
    Set preferred frequency for the given axon, based on axonm location (certain percent shuffled)
    '''
    min_frequency, max_frequency = experiment['prefered_frequency_boundaries']
    freq_dist                    = experiment['prefered_frequency_distribution']
    shuffle_percent              = experiment['prefered_frequency_distribution_shuffle_percent']
    
    assert shuffle_percent >= 0, ['Illegal shuffle percentage value!', pdb.set_trace()]

    if 'tonotopic' in  freq_dist:
        
        if shuffle_percent > 0:
            rnd_pref_freq = np.random.RandomState(int(axon_gid))
            
            if rnd_pref_freq.uniform(0, 1) < shuffle_percent:
                print('This calculation is weird (higher probability for low frequencies)')
                shuffled = True
                prefered_frequency = 2**rnd_pref_freq.uniform(log2(min_frequency), log2(max_frequency))

        else:
            shuffled = False
            gid_to_location   = experiment['axons_settings'][0] # Dictionary: {gid: [x_loc, y_loc]}
            min_x = min(array(list(gid_to_location.values()))[:, 0])
            max_x = max(array(list(gid_to_location.values()))[:, 0])
            axon_x = gid_to_location[axon_gid][0]

            range_doctave = log2(max_frequency)-log2(min_frequency)
            axon_doctave = (axon_x - min_x)/(max_x - min_x) * range_doctave
            
            prefered_frequency = 2**(log2(min_frequency) + axon_doctave)
    
    return prefered_frequency, shuffled # [pref_freq, is_neuron_shuffled]

def get_axon_stim_firing_rate(axon_gid, experiment):
    '''
    Draw firing rate from gaussiam distribution
    '''

    FR_dist = experiment['axon_stim_firing_rate_distribution']

    if FR_dist  == 'gaussian':
        dist_mean = experiment['axon_stim_firing_rate_gaussian_mean']
        dist_std  = experiment['axon_stim_firing_rate_gaussian_std']

        rnd_gauss = np.random.RandomState(int(axon_gid))
        width_value = -1
        while width_value<0: # to avoid minus
            width_value = rnd_gauss.normal(loc = dist_mean, scale = dist_std)
    
    return(width_value)

def get_tuning_width(axon_gid, experiment):
    '''
    Draw firing rate from gaussiam distribution
    '''

    rnd_width = np.random.RandomState(int(axon_gid))
    width_value=-1

    if experiment['tunning_width_distribution']  == 'gaussian':

        while width_value<0: # to avoid minus
            gaussian_mean = experiment['tunning_width_gaussian_mean']
            gaussian_std = experiment['tunning_width_gaussian_std']
            width_value = rnd_width.normal(loc = gaussian_mean, scale = gaussian_std)
    
    elif experiment['tunning_width_distribution'] == 'exp':
        
        while width_value<0: # to avoid minus
            exp_mean = experiment['tunning_width_exp_loc']
            exp_std = experiment['tunning_width_exp_scale']
            width_value = rnd_width.exponential(scale = exp_std) + exp_mean
    
    return(width_value)

def get_alpha_params(axon_gid, experiment):
    def get_alpha_func_delay(axon_gid, experiment):

        if experiment['alpha_func_delay_distribution']  == 'gaussian':
            
            gauss_mean = experiment['alpha_func_delay_gaussian_mean']
            gauss_std = experiment['alpha_func_delay_gaussian_std']

            rnd_gauss = np.random.RandomState(int(axon_gid))
            delay_value=-1

            while delay_value<0: # to avoid minus
                delay_value = rnd_gauss.normal(loc = gauss_mean, scale = gauss_std)
            
            def temp_func(*_):
                return delay_value
        
        elif experiment['alpha_func_delay_distribution']  == 'frequency_correlated':
            norm_factor             = experiment['delays_gaus_norm_factor']
            delays_gaus_std         = experiment['delays_gaus_std'] #Turn it to imutable 
            delays_gaus_base_line   = experiment['delays_gaus_base_line'];
            
            def temp_func(prefered_freq, stim_frequency):
                # Delay between input and thalamus is 15 ms (baseline); On top of this dependent delay- he took from experiments [auditory tuning curves]
                doctave = log2(stim_frequency)-log2(prefered_freq)
                exp_param = -(doctave**2) / (2*(delays_gaus_std**2)) # stim close to pref => exp_param closer to 0
                delay_value = delays_gaus_base_line - norm_factor*exp(exp_param) # stim close to pref => delay_value minimal
                return delay_value

        delay_value_f = temp_func
        return(delay_value_f)

    def get_alpha_func_tau(axon_gid, experiment):
        if experiment['alpha_func_tau_distribution']  == 'gaussian':
            gauss_mean = experiment['alpha_func_tau_gaussian_mean']
            gauss_std = experiment['alpha_func_tau_gaussian_std']

            rnd_gauss = np.random.RandomState(int(axon_gid))
            width_value=-1
        
            while width_value<0: # to avoid minus
                width_value = rnd_gauss.normal(loc = gauss_mean, scale = gauss_std)
        
        return width_value

    alpha_delay = get_alpha_func_delay(axon_gid, experiment)
    alpha_tau   = get_alpha_func_tau(axon_gid, experiment)

    return alpha_delay, alpha_tau

def get_duration_of_stims(duration_of_stim):
    '''
    Get duration of stimulus. Currently set to constant, so just returns the constant value
    '''

    if type(duration_of_stim)==list:
        def temp_func(stim_num):
            return(duration_of_stim[stim_num])
        duration_of_stim_f = temp_func
        
    elif type(duration_of_stim) in [int,float]:
        def temp_func(*_):
            return(duration_of_stim)
        duration_of_stim_f = temp_func

    else:
        raise Exception("duration_of_stim type is not clear")
    return(duration_of_stim_f)

def create_time_to_spike_frequency_func(experiment, axon_activity_vars):
    def get_firing_rates(frequency, axon_activity_vars, experiment):
        
        prefered_freq           = axon_activity_vars['prefered_freq']
        width_value             = axon_activity_vars['width_value']
        axon_stim_firing_rate   = axon_activity_vars['axon_stim_firing_rate']

        # set spontaneous and get freq_table
        def get_frequency_auditory(frequency, prefered_freq, width_value, axon_pref_stim_firing_rate, tunig_curve_type):
            
            if tunig_curve_type =='exp':
                doctave = log2(float(frequency)/prefered_freq)
                axon_stim_firing_rate = axon_pref_stim_firing_rate*exp(-abs(doctave)/width_value)
            
            elif tunig_curve_type=='triangle':
                    pref_log2  = log2(prefered_freq)
                    stim_log2  = log2(frequency)

                    min_tuning_edge = pref_log2 - width_value
                    max_tuning_edge = pref_log2 + width_value
                    
                    # Stimulus out of tuning curve
                    if stim_log2 < min_tuning_edge or stim_log2 >= max_tuning_edge:
                        axon_stim_firing_rate = 0.0000000001
                    
                    # Stimulus inside tuning curve
                    elif min_tuning_edge <= stim_log2 < pref_log2 or pref_log2 <= stim_log2 < max_tuning_edge:
                        doctave = log2(prefered_freq/frequency)
                        axon_stim_firing_rate = (axon_pref_stim_firing_rate*(1-(doctave)/width_value))
                                        
                    else:         
                        raise Exception('Problem')
            
            elif tunig_curve_type == 'gaussian':
                raise Exception('Not available yet')

            return axon_stim_firing_rate

        assert experiment['spontaneous_firing_rate_type']  == 'constant', 'Bad Input in create_time_to_spike_frequency_func/get_firing_rates'
        axon_spontanues_firing_rate   = experiment['spontaneous_firing_rate_value']
    
        # get stim firing rate
        tunig_curve_type  = experiment['tuning_curve_type']
        axon_stim_firing_rate = get_frequency_auditory(frequency, prefered_freq, width_value, axon_stim_firing_rate, tunig_curve_type)
       
        return(axon_stim_firing_rate, axon_spontanues_firing_rate)

    dt = 0.025

    #create stabilization time
    firing_rate_list = []
    max_firing_rate = None # I need this in order to create the spikes.
    
    ##firing_rate_list = np.ones(int(experiment['simulation_end_time'])/dt)
    for stim_num, frequency in enumerate(experiment['frequencies']):
        axon_stim_firing_rate, axon_spontanues_firing_rate =  get_firing_rates(frequency, axon_activity_vars, experiment)
        
        # HADAS: Fixed because python3 can't process comparing None to integer (originally this was inside the loop for some reason)
        if not max_firing_rate or max_firing_rate<axon_stim_firing_rate + axon_spontanues_firing_rate: 
            max_firing_rate=axon_stim_firing_rate + axon_spontanues_firing_rate

        if firing_rate_list==[]:
            firing_rate_list.append([0, axon_activity_vars['first_stimulus_time'], lambda t,axon_spontanues_firing_rate=axon_spontanues_firing_rate: axon_spontanues_firing_rate])   

        start_stim        = firing_rate_list[-1][1]
        end_stim          = start_stim + axon_activity_vars['duration_of_stim'](stim_num)
        end_between_stims = end_stim + axon_activity_vars['duration_between_stims']

        if experiment['stim_type'] == 'step':
            stim_func  = lambda t, stim_FR=axon_stim_firing_rate,spont_FR=axon_spontanues_firing_rate: stim_FR + spont_FR
            
            spont_func = lambda t, spont_FR=axon_spontanues_firing_rate: spont_FR

            firing_rate_list.append([start_stim, end_stim, stim_func])
            firing_rate_list.append([end_stim, end_between_stims, spont_func])
            
            
        elif experiment['stim_type'] == 'alpha_func':
            # axon_activity_vars['alpha_func_delay'] is a function; see def get_alpha_func_delay
            alpha_delay = start_stim + axon_activity_vars['alpha_func_delay'](axon_activity_vars['prefered_freq'], frequency)
            alpha_tau   = axon_activity_vars['alpha_func_tau']

            # The alpha function (original: t^n * exp(-t/tau)). Added parameters:
                # delay: moves the function on x axis
                # stim_FR/alpha_tau: sharpens function
                # +1: Makes maxium higher (equal to stim_FR)
                # n=1
            stim_func  = lambda t, stim_FR=axon_stim_firing_rate, spont_FR=axon_spontanues_firing_rate, alpha_delay=alpha_delay, alpha_tau=alpha_tau:\
                        (spont_FR + (stim_FR*(t-alpha_delay) / alpha_tau) * exp((-(t - alpha_delay)/alpha_tau)+1)) \
                        if t>alpha_delay else spont_FR
            
            spont_func = lambda t, spont_FR=axon_spontanues_firing_rate: spont_FR
           
            firing_rate_list.append([start_stim, end_stim, stim_func]) 
            firing_rate_list.append([end_stim, end_between_stims, spont_func])
        
        func = True
    return(time_to_firing_rate_frequency_class(firing_rate_list, max_firing_rate, dt, func))#max(firing_rate_list),dt))



##############################################



class time_to_firing_rate_frequency_class:
    def __init__(self, firing_rate_list, max_firing_rate, dt, func):
        self.firing_rate_list   = firing_rate_list        
        self.max_firing_rate    = max_firing_rate/1000.0 # Oren: I divide by 1000 to get the firing rate per ms
        self.dt                 = dt
        self.previous_j_ind     = 0

        #self.firing_rate_list_per_ms = np.array(firing_rate_list)/1000
        #self.js_list = range(0,len(self.firing_rate_list))
        if func:
            self.get_freq = self.get_freq_func
        else:
            self.get_freq = self.get_freq_list

    def get_freq_list(self, t):
        
        freq_list = self.firing_rate_list[int(t/self.dt)]/1000.0
        
        return freq_list #maybe I should do a liner interpetation 

    def get_freq_func(self,t):
        while True: # HADAS: I changed it to <=
            if self.firing_rate_list[self.previous_j_ind][0]<=t<self.firing_rate_list[self.previous_j_ind][1]:
                return self.firing_rate_list[self.previous_j_ind][2](t)/1000.0 # I divide by 1000 to get the firing rate per ms! as my time is ms
            self.previous_j_ind += 1




def get_spike_times(rnd_stream, time_to_firing_rate_frequency, stim_end_time):
    '''
    HADAS: Logic of this algorithm - CDF-Inverse Sampling:
    Based on papers by Raghu Pasupathy (Virginia Tech; pasupath@vt.edu): 
        "Generating Homogeneous Poisson Processes", "Generating Nonhomogeneous Poisson Processes"

    - We are randomly choosing a proportion of the area (denoted u) under the PDF curve and 
        returning the number in the domain such that exactly this proportion of the area under 
        the PDF occurs to the left of that number (the x for which the CDF F(x) = u.

    - Inverting a function entails getting x as a function of F(x). For example, if F(x) = 1-exp(-x), then 
        F^(-1)(x) => x = -log(1-F(x))

    - Our CDF is the CDF of ISIs, meaning an exponential random variable with parameter 1/lambda:
            - F(x) = 1 - exp(-lambda*x)

            - Draw F(x) from a uniform distribution, denote it u

            - u = 1 - exp(-lambda*x) ==> exp(-lambda*x) = 1 - u ==> x = -(1/lambda) * log(1 - u)
    '''

    spike_times = []
    t = 0 
    lamd = time_to_firing_rate_frequency.max_firing_rate #Do I need to multiply in 2????
    while (t<=stim_end_time):#{ :this Algorithm is from http://freakonometrics.hypotheses.org/724  from this paper. http://filebox.vt.edu/users/pasupath/papers/nonhompoisson_streams.pdf
        
        # Draw u
        u = rnd_stream.uniform() 
        # Update t with the drawn ISI
        t = t - np.log(u)/lamd # u is uniform(0,1) so drawing (1-u) is identical to drawing u.
        
        if t<=stim_end_time:
            if rnd_stream.uniform() <= time_to_firing_rate_frequency.get_freq(t)/lamd: # the function get_freq receives time and outputs the axon's FR at that time (based on the axon's preference and the stimuli presented)
                invl=t
                spike_times.append(invl)
            
            #This part is not good, becuase If I have an alpha function the firing rate in the start will be very low.
            #elif time_to_firing_rate_frequency.get_freq(t)<0.0001:      # if firing rate of this period is very low (if the spontanues is very low), I can just jump to the
            #     t = time_to_firing_rate_frequency.firing_rate_list[time_to_firing_rate_frequency.previous_j][1]-10  # end of the period
    return(spike_times)











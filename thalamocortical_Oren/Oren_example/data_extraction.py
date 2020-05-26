import numpy as np
import pickle
import os
from pathlib import Path
import time

start = time.time()

NeuronInfo = pickle.load(open('/Users/hadasmanor22/PycharmProjects/Orientation_Selectivity/data.p' , "rb") , encoding='latin1')
path_ = '/Users/hadasmanor22/PycharmProjects/Orientation_Selectivity/Ang_Cho_Freq_S8_OI_Cho_Net_2Hz_v1p5_spon_lin2p25'
gid_all_neurons = list(NeuronInfo['neurons_position'].keys()) # Checked- no duplicates, and sorted

def ExtTiming(filename):
    '''
    Input: filename of .dat file that includes [spike timing + neuron serial number] as strings

    Output:
        #total_timings = List of lists. Big list including small lists that specify spike timings for each neuron. Ordered by the serial numbers specified in serial_neuron
        dict_gid = dictionary of "gid": timings_list for this condition(filename)
        serial_neuron = List of neuron serial numbers.
    '''

    # ****** Extract data from file to nparray floats ******
    if Path('%s/%s/out.dat' %(path_, filename)).is_file():
        out = open('%s/%s/out.dat' %(path_, filename), 'r')
    else:
        out = open('%s/%s/Gout.dat' % (path_, filename), 'r')
    data = []
    for line in out:
        data.append(line.split())
    data = np.array(data[1:]).astype(np.float)

    # Find maximal neuron serial number
    serial_neuron = list(set(data[:, 1].astype(np.int)))
    # max_neuron = serial_neuron[-1]; min_neuron = serial_neuron[0]

    # ****** Create total list that includes a list of timings for each neuron ******
    dict_seed = {gid:[] for gid in serial_neuron}
    for i in range(len(data[:, 1])):
        dict_seed[data[i, 1]].append(data[i, 0])

    # total_timings = []
    # for i in serial_neuron:
    #     idx = []; idx = np.where(data[:, 1] == i)[0]
    #
    #     if len(idx) > 0: # if exists a neuron of this serial number (make sure)
    #         timing_list = data[idx, 0]
    #     total_timings.append(timing_list)

    return dict_seed, serial_neuron

orientation_list = list(range(0,181,20))
# orientation_list = [0]

# Go over all orientations (0,...,180)
gid_or_seed = {gid:{} for gid in gid_all_neurons}
for c in orientation_list:
    print('orientation # ', c)
    idx = np.where(orientation_list==c)[0] # place in orientation list (acts as counter)
    seed = [filename for filename in os.listdir(path_) if filename.startswith('%i_' %c)] # get all folder names corresponding to condition

    # Create dictionary of {'gid': {'orientation': timing_lists_by_seed}}
    for filename in seed:
        dict_, serial  = ExtTiming(filename) # Get dictionary dict_ = {'gid': timings_list}, given orientation[c] & seed[filename]

        gid_idx_in_seed = set(dict_.keys()) # Which neurons are recorded in this seed? (list of gid's)

        for gid in gid_all_neurons: # Go over gid's from all files
            if c not in gid_or_seed[gid]: # If orientation key doesn't exist yet, create it
                gid_or_seed[gid][c] = []
            if gid in dict_:
                gid_or_seed[gid][c].append(dict_[gid]) # If this gid exists in this seed, add its timings list
            else:
                gid_or_seed[gid][c].append([]) # If gid doesn't exist in this seed, add empty list

end = time.time()
print('time elapsed: ', end - start)

folder = '/Users/hadasmanor22/PycharmProjects/Orientation_Selectivity'
pickle.dump(gid_or_seed, open("%s/gid_or_seed.pickle" %folder,"wb"), protocol=-1)

print('finished')
import numpy as np
import pickle
import matplotlib.pyplot as plt

path_ = '/Users/hadasmanor22/PycharmProjects/Orientation_Selectivity'
NeuronInfo = pickle.load(open('%s/data.p' %path_ , "rb") , encoding='latin1')
gid_or_seed = pickle.load(open("%s/gid_or_seed.pickle" %path_,"rb")) # {'gid': {'orientation': timings_list_by_seed}}

gid_all_neurons = list(NeuronInfo['neurons_position'].keys())
orientation_list = list(range(0,181,20)); orientation_list.append('pref')

# === Create dictionary of average spike counts: {'gid': {'orientation': mean_spike_count_from_all_seeds}} ===
spike_count = {gid:{orientation: [] for orientation in orientation_list[:-1]} for gid in gid_all_neurons}
mean_spike_count = {gid:{orientation: [] for orientation in orientation_list} for gid in gid_all_neurons}
for gid in gid_all_neurons:
    for orientation in orientation_list[:-1]:
        num_seeds = len(gid_or_seed[gid][orientation])
        for i in range(num_seeds):
            spike_count[gid][orientation] += len(gid_or_seed[gid][orientation][i])

        mean_spike_count[gid][orientation] = spike_count[gid][orientation]/num_seeds

        if [mean_spike_count[gid][orientation]] > mean_spike_count[gid]['pref']: # Decide which orientation gets most firing from this neuron
            mean_spike_count[gid]['pref'] = mean_spike_count[gid][orientation]

# === Calculate orientation selectivity index ===
'''...How to calculate tuning width?....'''
OSI = np.zeros((1, gid_all_neurons))
theta_pref = np.zeros((1, gid_all_neurons))
for gid in gid_all_neurons:
    theta_pref[gid] = mean_spike_count[gid]['pref']
    theta_ortho = (theta_pref+90) % 180
    OSI[gid] = (mean_spike_count[gid][theta_pref] - mean_spike_count[gid][theta_ortho]) / (mean_spike_count[gid][theta_pref] + mean_spike_count[gid][theta_ortho])

# === Plot dependence of spike count on orientation for sample neurons ===
sample_neuron = 62800; sample_neuron1 = 62697; sample_neuron2 = 62793; sample_neuron3 = 63193; sample_neuron4 = gid_all_neurons[-10]
plt.plot(orientation_list, list(mean_spike_count[sample_neuron].values()), label='Neuron #%i' %sample_neuron)
plt.plot(orientation_list, list(mean_spike_count[sample_neuron1].values()), label='Neuron #%i' %sample_neuron1)
plt.plot(orientation_list, list(mean_spike_count[sample_neuron2].values()), label='Neuron #%i' %sample_neuron2)
plt.plot(orientation_list, list(mean_spike_count[sample_neuron3].values()), label='Neuron #%i' %sample_neuron3)
plt.plot(orientation_list, list(mean_spike_count[sample_neuron4].values()), label='Neuron #%i' %sample_neuron4)

plt.title('Mean Spike Count vs. Orientation'); plt.xlabel('Orientation'); plt.ylabel('Mean Spike Count')
plt.legend(); plt.show()

# === Plot samples responses of sample neuron ===
num = 62800
plt.plot(gid_or_seed[num][0][0], np.ones(len(gid_or_seed[num][0][0])),'b.', label='Orientation 0')
plt.plot(gid_or_seed[num][0][1], 2*np.ones(len(gid_or_seed[num][0][1])),'b.')
plt.plot(gid_or_seed[num][0][2], 3*np.ones(len(gid_or_seed[num][0][2])),'b.')

plt.plot(gid_or_seed[num][20][0], 4*np.ones(len(gid_or_seed[num][20][0])),'r.', label='Orientation 20')
plt.plot(gid_or_seed[num][20][1], 5*np.ones(len(gid_or_seed[num][20][1])),'r.')
plt.plot(gid_or_seed[num][20][2], 6*np.ones(len(gid_or_seed[num][20][2])),'r.')

plt.plot(gid_or_seed[num][40][0], 7*np.ones(len(gid_or_seed[num][40][0])),'g.', label='Orientation 40')
plt.plot(gid_or_seed[num][40][1], 8*np.ones(len(gid_or_seed[num][40][1])),'g.')
plt.plot(gid_or_seed[num][40][2], 9*np.ones(len(gid_or_seed[num][40][2])),'g.')

plt.title('Firing of Neuron #%i' %num); plt.xlabel('Time (ms)'); plt.legend(); plt.show()

# tuning curve
plt.plot(orientation_list[:-1], list(mean_spike_count[num].values())); plt.title('Tuning curve for Neuron #%i' % num)
plt.xlabel('Orientation (degrees)'); plt.ylabel('Spike Count'); plt.show

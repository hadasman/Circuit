import numpy as numpy
import _pickle as cPickle
import os
from tqdm import tqdm

# Success inputs (multiple)
success_group = cPickle.load(open('cluster_downloadables/input_groups_8852510'))['spike_success']
activations = cPickle.load(open('thalamocortical_Oren/SSA_spike_times/input6666_by_gid.p', 'rb'))

take_before=0; take_after=50 # MATCH THESE TO PARAMETERS GIVEN TO THE CREATING FUNCTION (Circuit.py => InputvsResponse())




# Repetitive one stimulus
TIME = 2000
start = -150
end = 150
standard = 6666
GIDs_to_change = cPickle.load(open('GIDs_instantiations/pyr_72851_between_6666_9600/connecting_gids_to_pyr.p','rb'))
stim_times = cPickle.load(open('thalamocortical_Oren/SSA_spike_times/stim_times.p', 'rb'))[standard]
stim_times = [i[0] for i in stim_times]

model = {a: [] for a in activations}
for a in activations: 
	model_times = [i-TIME for i in activations[a] if (i>TIME+start) and (i<TIME+end)]
	model[a] = model_times 

new_activations = {a: [] for a in activations}
for a in activations: 
	if a in GIDs_to_change:
		spont_times = [i for i in activations[a] if i<=TIME+start]
		new_activations[a] = spont_times

for temp_T in stim_times:
	for a in activations:
		if a in GIDs_to_change:
			temp_model_times = [i+temp_T for i in model[a]]
			new_activations[a] = new_activations[a] + temp_model_times
		else:
			new_activations[a] = activations[a]

filename = input('Choose filename (not including path and extension)\nTo skip saving press enter:\n')
if filename:
	cPickle.dump(new_activations, open('thalamocortical_Oren/SSA_spike_times/{}.p'.format(filename), 'wb'))

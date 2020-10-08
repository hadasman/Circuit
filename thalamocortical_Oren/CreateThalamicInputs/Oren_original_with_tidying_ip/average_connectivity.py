import numpy as np
import pdb
import _pickle as cPickle
import pandas as pd
from time import time

connectivity = cPickle.load(open('thalamocortical_Oren/post_to_pre_cortex_connectivity_num_connections.p', 'rb'))
cell_details = pd.read_pickle('thalamocortical_Oren/thalamic_data/cells_details.pkl')
cell_type_to_gids = cPickle.load(open('thalamocortical_Oren/thalamic_data/cell_type_gids.pkl','rb')) 

pyr_type = 'L4_PC'
not_PVs = ['PC', 'SP', 'SS', 'MC', 'BTC', 'L1']
SOM_types = ['MC']

pyr_gids = cell_type_to_gids[pyr_type]

PV_gids, SOM_gids = [], []
for cell_type in cell_type_to_gids:
	if all([i not in cell_type for i in not_PVs]):
		PV_gids.append(cell_type_to_gids[cell_type])

	if any([i in cell_type for i in SOM_types]):
		SOM_gids.append(cell_type_to_gids[cell_type])

PV_gids = [j for i in PV_gids for j in i]
SOM_gids = [j for i in SOM_gids for j in i]

SOM_outputs = {'to_pyr': {'contacts': {}, 'pre_neurons': {}},
			   'to_PV': {'contacts': {}, 'pre_neurons': {}}}

SOM_to_pyr_presynaptics, SOM_to_pyr_contacts = [], []
for post_gid in pyr_gids:
	all_pres = connectivity[post_gid]
	temp_pre_SOMs = [i for i in all_pres if i in SOM_gids]

	SOM_to_pyr_contacts.append([all_pres[i] for i in temp_pre_SOMs])
	SOM_to_pyr_presynaptics.append(len(temp_pre_SOMs))

SOM_outputs['to_pyr']['contacts']['mean'] 		= np.mean([j for i in SOM_to_pyr_contacts for j in i])
SOM_outputs['to_pyr']['contacts']['std'] 		= np.std([j for i in SOM_to_pyr_contacts for j in i])
SOM_outputs['to_pyr']['pre_neurons']['mean'] 	= np.mean(SOM_to_pyr_presynaptics)
SOM_outputs['to_pyr']['pre_neurons']['std'] 	= np.std(SOM_to_pyr_presynaptics)

SOM_to_PV_presynaptics, SOM_to_PV_contacts = [], []
for post_gid in PV_gids:
	all_pres = connectivity[post_gid]
	temp_pre_SOMs = [i for i in all_pres if i in SOM_gids]

	SOM_to_PV_contacts.append([all_pres[i] for i in temp_pre_SOMs])
	SOM_to_PV_presynaptics.append(len(temp_pre_SOMs))

SOM_outputs['to_PV']['contacts']['mean'] 		= np.mean([j for i in SOM_to_PV_contacts for j in i])
SOM_outputs['to_PV']['contacts']['std'] 		= np.std([j for i in SOM_to_PV_contacts for j in i])
SOM_outputs['to_PV']['pre_neurons']['mean'] 	= np.mean(SOM_to_PV_presynaptics)
SOM_outputs['to_PV']['pre_neurons']['std'] 		= np.std(SOM_to_PV_presynaptics)






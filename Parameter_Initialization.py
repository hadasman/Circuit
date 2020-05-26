import pandas as pd

# global pyr_template_path, pyr_template_name, pyr_morph_path
# global PV_template_path, PV_template_name, PV_morph_path
# global PV_input_weight, PV_to_Pyr_weight, Pyr_input_weight
# global pyr_type, not_PVs, SOM_types, stimuli, connecting_gids
# global PV_input_delay, Pyr_input_delay, PV_output_delay, freq1, freq2, simulation_time, spike_threshold
# global filenames, cell_type_gids, thal_connections, thalamic_locations

pyr_template_path 	= 'EPFL_models/L4_PC_cADpyr230_1'
pyr_template_name 	= 'cADpyr230_L4_PC_f15e35e578'
pyr_morph_path 		= '{}/morphology'.format(pyr_template_path)

PV_template_path 	= 'EPFL_models/L4_LBC_cNAC187_1'
PV_template_name 	= 'cNAC187_L4_LBC_990b7ac7df'
PV_morph_path 		= '{}/morphology/'.format(PV_template_path)

SOM_template_path = 'EPFL_models/L4_MC_cACint209_1'
SOM_template_name = 'cACint209_L4_MC_ba3c5063e4'
SOM_morph_path 	  = '{}/morphology/'.format(SOM_template_path)

PV_input_weight  = 0.4
PV_to_Pyr_weight = 0.4
Pyr_input_weight = 0.5
SOM_input_weight = 0.4
n_SOM  = 9# CHECK THIS
SOM_to_PV_weight = PV_to_Pyr_weight * n_SOM# CHECK THIS
SOM_to_Pyr_weight = PV_to_Pyr_weight * n_SOM# CHECK THIS
n_SOM_to_PV_syns = 60# CHECK THIS
n_SOM_to_Pyr_syns = 60 # CHECK THIS

pyr_type = 'L4_PC'
not_PVs = ['PC', 'SP', 'SS', 'MC', 'BTC', 'L1']
SOM_types = ['MC']

PV_input_delay 	= 0 # TEMPORARY: CHANGE THIS
Pyr_input_delay = PV_input_delay + 7 # 7 taken from Tohar paper (Fig 4- Supp. 2a)
PV_output_delay = Pyr_input_delay+5 # TEMPORARY: CHECK THIS
SOM_input_delay = PV_input_delay + 7 # 7 taken from Tohar paper (Fig 4- Supp. 2a)
SOM_output_delay = 0

freq1 = 6666
freq2 = 9600
simulation_time = 154*1000 # in ms
spike_threshold = 0


filenames = {'cell_details': 'thalamocortical_Oren/thalamic_data/cells_details.pkl', 
		 'thalamic_locs': 'thalamocortical_Oren/thalamic_data/thalamic_axons_location_by_gid.pkl',
		 'thalamic_connections': 'thalamocortical_Oren/thalamic_data/thalamo_cortical_connectivity.pkl',
		 'thalamic_activations_6666': 'thalamocortical_Oren/SSA_spike_times/input6666_success_150_before_150_after_only_to_pyr.p',
		 # 'thalamic_activations_6666': 'thalamocortical_Oren/SSA_spike_times/input6666_by_gid.p',
		 'thalamic_activations_9600': 'thalamocortical_Oren/SSA_spike_times/input9600_by_gid.p',
		 'PV_spike_times': 'thalamocortical_Oren/SSA_spike_times/PV_spike_times/PV_spike_times_tstop_10000_input6666_by_gid.p',
		 'pyr_connectivity': 'thalamocortical_Oren/pyramidal_connectivity_num_connections.p',
		 'cortex_connectivity': 'thalamocortical_Oren/cortex_connectivity_num_connections.p',       
		 'cell_type_gids': 'thalamocortical_Oren/thalamic_data/cell_type_gids.pkl',
		 'stim_times': 'thalamocortical_Oren/SSA_spike_times/stim_times.p',
		 'thalamic_activations': {6666: 'thalamocortical_Oren/SSA_spike_times/input6666_106746.dat', 9600: 'thalamocortical_Oren/SSA_spike_times/input9600_109680.dat'}}

PV_to_Pyr_source = 'voltage'
assert PV_to_Pyr_source in ['spike_times', 'voltage'], 'Invalid PV output type (choose spike_times or voltage)'

thal_connections   = pd.read_pickle(filenames['thalamic_connections'])
thalamic_locations = pd.read_pickle(filenames['thalamic_locs'])

stimuli = {}
recorded_segment = []

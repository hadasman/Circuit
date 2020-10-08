import pandas as pd

connect_interneurons=True
upload_cortical_syn_locs = True

recording_dt = 0.1

connect_PV_to_Pyr  = True
connect_SOM_to_Pyr = True
connect_SOM_to_PV  = True

record_thalamic_syns = False  # Thalamic source (presynaptic)

record_PV_syns 		 = True # PV source (presynaptic)
record_SOM_syns 	 = True # SOM source (presynaptic)

record_Pyr_syns 	 = False # Pyr source (presynaptic)
record_channel 		 = False

SOM_from = 'post'

thalamic_params = {'ITI': 120, 'IPI': 2000, 'type': 'single', 'freq': 'BroadBand'}

Pyr_input_params 	= {'Dep': 671, 'Fac': 5, 'Use': 0.86}# {'Dep': 0, 'Fac': 0, 'Use': 1}
PV_input_params 	= {'Dep': 671, 'Fac': 5, 'Use': 0.86}
SOM_input_params 	= {'Dep': 671, 'Fac': 5, 'Use': 0.86}
Pyr_IClamp_amp 		= 0.1

Pyr_input_weight = 1
PV_input_weight  = 0.6
SOM_input_weight = 0.6

PV_to_Pyr_weight = 0.5
SOM_to_Pyr_weight = 0.3 

# SOM => Pyr
n_SOMs_to_Pyr     = 11 # From BB connectivity data, based on cell types below
n_syns_SOM_to_Pyr = 11 * n_SOMs_to_Pyr # From BB connectivity data, based on cell types below

# SOM => PV
# n_SOMs_to_PV 	 = 5 # From BB connectivity data, based on cell types below
n_syns_SOM_to_PV = 200 # 11 * n_SOMs_to_PV # From BB connectivity data, based on cell types below
SOM_to_PV_weight = SOM_to_Pyr_weight # CHECK THIS

pyr_template_path 	= 'EPFL_models/L4_PC_cADpyr230_1'
pyr_template_name 	= 'cADpyr230_L4_PC_f15e35e578'
pyr_morph_path 		= '{}/morphology'.format(pyr_template_path)

PV_template_path 	= 'EPFL_models/L4_LBC_cNAC187_1'
PV_template_name 	= 'cNAC187_L4_LBC_990b7ac7df'
PV_morph_path 		= '{}/morphology/'.format(PV_template_path)

SOM_template_path = 'EPFL_models/L4_MC_cACint209_1'
SOM_template_name = 'cACint209_L4_MC_ba3c5063e4'
SOM_morph_path 	  = '{}/morphology/'.format(SOM_template_path)

pyr_type = 'L4_PC'
not_PVs = ['PC', 'SP', 'SS', 'MC', 'BTC', 'L1', 'BP', 'BPC']
SOM_types = ['MC', 'BPC', 'BP']

PV_input_delay 	= 0 # TEMPORARY: CHANGE THIS
Pyr_input_delay = PV_input_delay + 4 # taken from Tohar paper (Fig 4- Supp. 2a)
SOM_input_delay = PV_input_delay + 3 # taken from Tohar paper (Fig 4- Supp. 2a)

PV_output_delay = Pyr_input_delay + 5 # TEMPORARY: CHECK THIS
SOM_output_delay = PV_output_delay# + something??

# PV_output_params = {'Dep': 706, 'Fac': 21, 'Use': 0.25}
# SOM_output_params = {'Dep': 706, 'Fac': 21, 'Use': 0.25}

freq1 = 6666
freq2 = 9600
simulation_time = 154*1000 # in ms
spike_threshold = 0

filenames = {'cell_details': 'thalamocortical_Oren/thalamic_data/cells_details.pkl', 
		 'thalamic_locs': 'thalamocortical_Oren/thalamic_data/thalamic_axons_location_by_gid.pkl',
		 'thalamic_connections': 'thalamocortical_Oren/thalamic_data/thalamo_cortical_connectivity.pkl',

		 # 'thalamic_activations_6666': 'thalamocortical_Oren/CreateThalamicInputs/test_times/input6666_106746_pairs_ITI_120_IPI_2000.p',
		 # 'thalamic_activations_9600': 'thalamocortical_Oren/CreateThalamicInputs/test_times/input9600_109680_pairs_ITI_120_IPI_2000.p',
		 
		 'stim_times': 'thalamocortical_Oren/SSA_spike_times/stim_times.p',

		 'PV_spike_times': 'thalamocortical_Oren/SSA_spike_times/PV_spike_times/PV_spike_times_tstop_10000_input6666_by_gid.p',
		 'pyr_connectivity': 'thalamocortical_Oren/pyramidal_connectivity_num_connections.p',
		 'cortex_connectivity': 'thalamocortical_Oren/cortex_connectivity_num_connections.p',       
		 'cell_type_gids': 'thalamocortical_Oren/thalamic_data/cell_type_gids.pkl',
		 'thalamic_activations': {6666: 'thalamocortical_Oren/SSA_spike_times/input6666_106746.dat', 9600: 'thalamocortical_Oren/SSA_spike_times/input9600_109680.dat'}}

PV_to_Pyr_source = 'voltage'
assert PV_to_Pyr_source in ['spike_times', 'voltage'], 'Invalid PV output type (choose spike_times or voltage)'
assert PV_to_Pyr_source != 'spike_times', 'Still under construction!'

thal_connections   = pd.read_pickle(filenames['thalamic_connections'])
thalamic_locations = pd.read_pickle(filenames['thalamic_locs'])

recorded_segment = []

activated_frequency = thalamic_params['freq']
stand_freq = freq1*(str(freq1) == activated_frequency) + freq2*(str(freq2) == activated_frequency)
dev_freq = freq1*(str(freq1) != activated_frequency) + freq2*(str(freq2) != activated_frequency)


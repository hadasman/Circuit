import numpy as np
import _pickle as cPickle
import matplotlib.pyplot as plt
plt.ion()

import pdb, os, sys

from neuron import gui, h

from scipy.stats import ttest_ind

from Population import Population
from Stimulus import Stimulus
from Connectivity import Connectivity
from Parameter_Initialization import *

Pyr_pop = Population('Pyr', pyr_morph_path, pyr_template_path, pyr_template_name, verbose=False)
Pyr_pop.addCell()
soma = Pyr_pop.cells['Pyr0']['soma']                                                   

def exp_setup(electrode_loc):

   syn = h.ProbAMPANMDA_EMS(electrode_loc)                                                    
   syn.Dep=0                                                                           

   stim = h.IClamp(electrode_loc)                                                             
   stim.amp=1
   stim.dur=500
   stim.delay=100                                               

   netcon = h.NetCon(stim,syn)                                                            
   netcon.weight[0]=0.4   

   t = h.Vector()
   t.record(h._ref_t)                                                     
   v = h.Vector()
   v.record(electrode_loc._ref_v)                                             

   return syn, stim, netcon, t, v                                                              

electrode_loc = soma(0.5)
syn, stim, netcon, t, v = exp_setup(electrode_loc)

h.tstop = stim.delay + stim.dur + 50                                                                          

def run_with_param(sections, pairs, title_, h_ax, h_ax2):

   for prop, new_val in pairs:
      sec_name = 'axon'*any([i for i in sections if 'axon' in i.name()])+\
                 'soma'*any([i for i in sections if 'soma' in i.name()])+\
                 'apic'*any([i for i in sections if 'apic' in i.name()])+\
                 'basal'*any([i for i in sections if 'basal' in i.name()])
      print('Setting {} in {} to {}'.format(prop, sec_name, new_val))
      for sec in sections:
         exec("sec.{} = {}".format(prop, new_val))

   h.run()

   h_ax.plot(t, v)
   h_ax.set_title(title_)

   spike_times = [t[i] for i in range(len(v)-1) if (v[i] > v[i-1]) and (v[i] > v[i+1])]
   ISIs = np.diff(spike_times)
   norm_ISIs = [i/max(ISIs) for i in ISIs]
   rel_ISIs = [ISIs[i+1]/ISIs[i] for i in range(len(ISIs)-1)]

   h_ax2[0].plot(list(spike_times)[:-1], norm_ISIs, label=title_)
   h_ax2[1].plot(list(spike_times)[:-2], rel_ISIs, label=title_)

props_dict_axonal = {
               'decay_CaDynamics_E2': {'Pyr_val':179.044149, 'PV_val': 64.277990},
               'gSK_E2bar_SK_E2': {'Pyr_val': 0.097244, 'PV_val': 0.003442},
               'gCa_HVAbar_Ca_HVA': {'Pyr_val': 0.000860, 'PV_val': 0},
}

props_dict_apical = {
               'gNaTs2_tbar_NaTs2_t': {'Pyr_val': 0.022874, 'PV_val': 0.000010},
               'gSKv3_1bar_SKv3_1': {'Pyr_val': 0.039967, 'PV_val': 0.004399},
               'gImbar_Im': {'Pyr_val': 0.001000, 'PV_val': 0.000008}
}

props_dict_somatic = {
               # 'gamma_CaDynamics_E2': {'Pyr_val': 0.002253, 'PV_val': 0.000511},
               # 'gSKv3_1bar_SKv3_1': {'Pyr_val': 0.283745, 'PV_val': 0.297559},
               'gSK_E2bar_SK_E2': {'Pyr_val': 0.002971, 'PV_val': 0.019726},
               # 'gCa_HVAbar_Ca_HVA': {'Pyr_val': 0.000379, 'PV_val': 0},
               'gNaTs2_tbar_NaTs2_t': {'Pyr_val': 0.999812, 'PV_val': 0.197999},
               'gIhbar_Ih': {'Pyr_val': 0.000080, 'PV_val': 0},
               # 'decay_CaDynamics_E2': {'Pyr_val': 739.416497, 'PV_val': 731.707637},
               'gCa_LVAstbar_Ca_LVAst': {'Pyr_val': 0.006868, 'PV_val': 0.001067}
}

inspect_section = 'soma'
sections = [i for i in h.allsec() if inspect_section in i.name()]

if inspect_section=='apic':
   used_dict = props_dict_apical
elif inspect_section=='axon':
   used_dict = props_dict_axonal
elif inspect_section=='soma':
   used_dict = props_dict_somatic
else: raise Exception('error')

prop1 = list(used_dict.keys())[0]
Pyr_val1 = used_dict[prop1]['Pyr_val']
PV_val1 = used_dict[prop1]['PV_val']

prop2 = list(used_dict.keys())[1]
Pyr_val2 = used_dict[prop2]['Pyr_val']
PV_val2 = used_dict[prop2]['PV_val']

prop3 = list(used_dict.keys())[2]
Pyr_val3 = used_dict[prop3]['Pyr_val']
PV_val3 = used_dict[prop3]['PV_val']

suptitle_ = 'Changing Different Values to Eliminate Adaptation'

prop_vals1 = [Pyr_val1, PV_val1, 1e-100]
prop_vals2 = [Pyr_val2, PV_val2, 0]
prop_vals3 = [Pyr_val3, PV_val3, 0]

titles = ['Pyramidal', 'PV', '0']
fig, ax = plt.subplots(3, 2, figsize=(9, 7.5))
fig.suptitle(suptitle_, size=15)

fig1, ax1 = plt.subplots(2, 1)


# Run with original pyramidal parameters
temp_list = [[prop1, prop_vals1[0]], [prop2, prop_vals2[0]], [prop3, prop_vals3[0]]] 
run_with_param(sections, temp_list, r'Original Parameters', ax[0][0], ax1)

# Change only decay_CaDynamics_E2
temp_list = [[prop1, prop_vals1[2]], [prop2, prop_vals2[0]], [prop3, prop_vals3[0]]] 
run_with_param(sections, temp_list, r'Change Only $CaDecayDynamics$ to 0', ax[0][1], ax1)

# Change only gSK_E2bar_SK_E2
temp_list = [[prop1, prop_vals1[0]], [prop2, prop_vals2[2]], [prop3, prop_vals3[0]]] 
run_with_param(sections, temp_list, r'Change Only $g_{SK}\bar{E}_{2}$ to 0', ax[1][0], ax1)

# Change only gCa_HVAbar_Ca_HVA
temp_list = [[prop1, prop_vals1[0]], [prop2, prop_vals2[0]], [prop3, prop_vals3[2]]] 
run_with_param(sections, temp_list, r'Change Only $g_{Ca}\bar{HVA}$ to 0', ax[1][1], ax1)

# Make all 0
temp_list = [[prop1, prop_vals1[2]], [prop2, prop_vals2[2]], [prop3, prop_vals3[2]]] 
run_with_param(sections, temp_list, r'Change All to 0', ax[2][0], ax1)

# Change all
temp_list = [[prop1, prop_vals1[1]], [prop2, prop_vals2[1]], [prop3, prop_vals3[1]]] 
run_with_param(sections, temp_list, r'Change All to PV Values', ax[2][1], ax1)

fig.subplots_adjust(hspace=0.4, bottom=0.08, top=0.9, right=0.95,left=0.09)       
ax[0][0].set_ylabel('V (mV)')
ax[1][0].set_ylabel('V (mV)')
ax[2][0].set_ylabel('V (mV)')
ax[2][0].set_xlabel('T (ms)')
ax[0][1].set_ylabel('V (mV)')
ax[1][1].set_ylabel('V (mV)')
ax[1][1].set_xlabel('T (ms)')
# fig.delaxes(ax[2][1])  


fig1.subplots_adjust(hspace=0.15, bottom=0.08, top=0.89, right=0.97, left=0.11)       
fig1.suptitle(suptitle_,size=13)
ax1[0].legend()
ax1[0].set_title('Normalized Spike ISIs Throughout Simulation')
ax1[0].set_ylabel('Normalized ISI')
ax1[0].set_xlim([0, stim.dur+stim.delay])
ax1[0].set_xticks([])
ax1[1].legend()
ax1[1].set_title('Ratio of Consecutive ISIs Throughout Simulation')
ax1[1].set_xlabel('T (ms)')
ax1[1].set_ylabel('ISI Ratio')
ax1[1].axhline(1, LineStyle='--', LineWidth=1, color='gray')
ax1[1].set_xlim([0, stim.dur+stim.delay])












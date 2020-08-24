import pandas as pd
import numpy as np
import _pickle as cPickle
import matplotlib.pyplot as plt
plt.ion()

import pdb, os, sys
from Parameter_Initialization import * # Initialize parameters before anything else!
from neuron import gui, h
import matplotlib.ticker as mtick 

cell_details = pd.read_pickle('thalamocortical_Oren/thalamic_data/cells_details.pkl') 



# Process data
def process_syn_data():
     syn_data  = {}
     SOM_gids, PV_gids, pyr_gid = [], [], []

     filenames = [i for i in os.listdir('thalamocortical_Oren/cortical_synaptic_locations') if 'gid_' in i]
     GIDs      = [int(i.split('_')[1]) for i in os.listdir('thalamocortical_Oren/cortical_synaptic_locations') if 'gid_' in i]

     for f in filenames:
          gid = int(f.split('_')[1])
          syn_data.update(cPickle.load(open('thalamocortical_Oren/cortical_synaptic_locations/{}/gid_data.p'.format(f), 'rb'), encoding='latin1'))

          if any([t in cell_details.loc[gid].mtype for t in SOM_types]):
               SOM_gids.append(gid)
          elif pyr_type in cell_details.loc[gid].mtype:
               pyr_gid.append(gid)
          else:
               PV_gids.append(gid)


     new_syn_data = {gid: {} for gid in syn_data} 
     for gid in syn_data:
          for syn_id in syn_data[gid][1]:

               new_syn_data[gid][syn_id] = syn_data[gid][1][syn_id]['info_dict'].copy()
               del new_syn_data[gid][syn_id]['randseed1'], new_syn_data[gid][syn_id]['randseed2'], new_syn_data[gid][syn_id]['randseed3']

               new_syn_data[gid][syn_id].update({'post_netcon': syn_data[gid][0][syn_id]['info_dict']['post_netcon'], 
                    'post_synapse_id': syn_data[gid][0][syn_id]['info_dict']['post_synapse_id']})
     return new_syn_data, pyr_gid, PV_gids, SOM_gids

upload_processed_syn_data = True
if upload_processed_syn_data:
     pyr_gid, PV_gids, SOM_gids = [], [], []
     syn_data = cPickle.load(open('thalamocortical_Oren/cortical_synaptic_locations/processed_syn_data.p', 'rb'))
     for post_gid in syn_data:
          if any([t in cell_details.loc[post_gid].mtype for t in SOM_types]):
               SOM_gids.append(post_gid)
          elif pyr_type in cell_details.loc[post_gid].mtype:
               pyr_gid.append(post_gid)
          elif all([t not in cell_details.loc[post_gid].mtype for t in not_PVs]): 
               PV_gids.append(post_gid)
          else:
               raise Exception('Unidentified cell type: {}'.format(post_gid))
else:
     syn_data, pyr_gid, PV_gids, SOM_gids = process_syn_data()



# Plot and analyze
def plot_SynData(post_name, post_GIDs, pre_name, pre_GIDs, plot_locs=None, plot_weights=None):

     def get_SynData(post_GIDs, pre_GIDs):
          locations, weights = [], []

          weight_by_loc = {'soma': [], 'dend': [], 'apic': []}

          for post in post_GIDs:

               temp_syns = [i for i in syn_data[post] if syn_data[post][i]['pre_cell_id'] in pre_GIDs]

               temp_locs = [syn_data[post][i]['post_sec_name'].split('[')[0] for i in temp_syns]

               temp_weights = [syn_data[post][i]['post_netcon']['weight'] for i in temp_syns]

               for i in temp_syns:
                    sec = syn_data[post][i]['post_sec_name'].split('[')[0]
                    weight_by_loc[sec].append(syn_data[post][i]['post_netcon']['weight'])

               locations.append(temp_locs)

               assert len(set([syn_data[post][i]['mech_name'] for i in temp_syns[1:]]))<=1, ['Non-uniform synapse type! (problem with analyzing)', pdb.set_trace()]

          # locations = [j for i in locations for j in i]

          weights   = [j for i in weights for j in i]

          return locations, weights, weight_by_loc

     def PlotLocs(locations, post_name, pre_name, h_ax):
          h_ax.set_title('{} to {} Synapse Locations'.format(pre_name, post_name))   

          mean_soma = np.mean([locations[i].count('soma') for i in range(len(locations))])
          mean_dend = np.mean([locations[i].count('dend') for i in range(len(locations))])
          mean_apic = np.mean([locations[i].count('apic') for i in range(len(locations))])

          h_ax.bar(['soma', 'dend', 'apic'], [mean_soma, mean_dend, mean_apic], alpha=0.5, color='xkcd:azure', label='Location')
          # h_ax.bar(['soma', 'dend', 'apic'], [locations.count('soma'), locations.count('dend'), locations.count('apic')], alpha=0.5, color='xkcd:azure', label='Location')
          h_ax.set_ylabel('Count')
          h_ax.legend(loc='upper left')

     def PlotWeights(weight_by_loc, post_name, pre_name, h_ax):
          # h_ax.set_title('{} to {} Mean Synapse Weights'.format(pre_name, post_name))
          h_ax.bar(['soma', 'dend', 'apic'], [np.mean(weight_by_loc['soma']), np.mean(weight_by_loc['dend']), np.mean(weight_by_loc['apic'])], alpha=0.5, color='xkcd:magenta', label='Mean Weight')
          h_ax.set_ylabel('Mean Weight (nS)')
          # h_ax.set_ylim([0, h_ax.get_ylim()[1]*2])
          h_ax.set_ylim([0, 0.1+np.max([np.mean(weight_by_loc['soma']), np.mean(weight_by_loc['dend']), np.mean(weight_by_loc['apic'])])])
          h_ax.legend(loc='upper right')


     locations, weights, weight_by_loc = get_SynData(post_GIDs, pre_GIDs)

     if plot_locs:
          PlotLocs(locations, post_name, pre_name, plot_locs)
     if plot_weights:
          PlotWeights(weight_by_loc, post_name, pre_name, plot_weights)

     return locations, weight_by_loc

f, h_ax1 = plt.subplots(3, 1, figsize=(10, 7.5))
h_ax2 = [i.twinx() for i in h_ax1]
f.suptitle('Cortico-cortical Synapse Information')
f.subplots_adjust(hspace=0.48, bottom=0.08, top=0.91, left=0.1, right=0.9) 

# For plotting only weights, set plot_locs to None (default); opposite for plotting only locations
SOM_to_PV_locs, SOM_to_PV_weights   = plot_SynData('PV', PV_gids, 'SOM', SOM_gids, plot_locs=h_ax1[0], plot_weights=h_ax2[0])
SOM_to_Pyr_locs, SOM_to_Pyr_weights = plot_SynData('Pyr', pyr_gid, 'SOM', SOM_gids, plot_locs=h_ax1[1], plot_weights=h_ax2[1])
PV_to_Pyr_locs, PV_to_Pyr_weights   = plot_SynData('Pyr', pyr_gid, 'PV', PV_gids, plot_locs=h_ax1[2], plot_weights=h_ax2[2])

def get_LocByMorph(syn_data, all_cortical_cells, cell_details):
     loc_by_morph = {'soma': [], 'dend': [], 'apic': []}

     for i in syn_data[gid]:
          if syn_data[gid][i]['pre_cell_id'] in all_cortical_cells:
               
               pre_gid = syn_data[gid][i]['pre_cell_id']
               pre_morph = cell_details.loc[pre_gid].me_combo

               if 'soma' in syn_data[gid][i]['post_sec_name']:
                    loc_by_morph['soma'].append(pre_morph)
               elif 'dend' in syn_data[gid][i]['post_sec_name']: 
                    loc_by_morph['dend'].append(pre_morph)
               elif 'apic' in syn_data[gid][i]['post_sec_name']: 
                    loc_by_morph['apic'].append(pre_morph)
     return loc_by_morph

def plot_LocByMorph(loc_by_morph, soma_morphs, dend_morphs, apic_morphs, as_percent=False):
     f, morph_ax = plt.subplots(3, 1, figsize=(15, 7.5))
     f.subplots_adjust(hspace=0.48, bottom=0.08, top=0.91, left=0.07, right=0.99)

     if as_percent:
          f.suptitle('Histogram of Morphologies Targeting Each Area on {} cell {} (% from total synapse number)'.format(which_post, gid), size=13)
          soma_norm = len(loc_by_morph['soma'])
          dend_norm = len(loc_by_morph['dend'])
          apic_norm = len(loc_by_morph['apic'])

          morph_ax[0].yaxis.set_major_formatter(mtick.PercentFormatter(1.0)) 
          morph_ax[1].yaxis.set_major_formatter(mtick.PercentFormatter(1.0)) 
          morph_ax[2].yaxis.set_major_formatter(mtick.PercentFormatter(1.0)) 
     else:
          f.suptitle('Histogram of Morphologies Targeting Each Area on {} cell {} (Absolute numbers)'.format(which_post, gid), size=13)
          soma_norm = 1
          dend_norm = 1
          apic_norm = 1
     
     morph_ax[0].bar(soma_morphs, [loc_by_morph['soma'].count(i)/soma_norm for i in soma_morphs])
     morph_ax[1].bar(dend_morphs, [loc_by_morph['dend'].count(i)/dend_norm for i in dend_morphs])
     morph_ax[2].bar(apic_morphs, [loc_by_morph['apic'].count(i)/apic_norm for i in apic_morphs])

     morph_ax[0].set_xticklabels(soma_morphs, fontsize=7);  
     morph_ax[1].set_xticklabels(dend_morphs, fontsize=7);  
     morph_ax[2].set_xticklabels(apic_morphs, fontsize=7);  
     
     morph_ax[0].set_title('Soma')
     morph_ax[1].set_title('Basal Dendrites')
     morph_ax[2].set_title('Apical Dendrites')
     morph_ax[0].set_ylabel('Count')
     morph_ax[1].set_ylabel('Count')
     morph_ax[2].set_ylabel('Count')
     morph_ax[2].set_xlabel('Morphology')

def plot_LocByCell(loc_by_morph, as_percent=False):

     f, cell_ax = plt.subplots(3, 1, figsize=(15, 7.5))
     f.subplots_adjust(hspace=0.48, bottom=0.08, top=0.91, left=0.07, right=0.99)

     if as_percent:
          f.suptitle('Histogram of Molecular Cell Types targeting Each Area on {} cell {} (% from total synapse number)'.format(which_post, gid), size=13)

          soma_norm = len(loc_by_morph['soma'])
          dend_norm = len(loc_by_morph['dend'])
          apic_norm = len(loc_by_morph['apic'])

          cell_ax[0].yaxis.set_major_formatter(mtick.PercentFormatter(1.0)) 
          cell_ax[1].yaxis.set_major_formatter(mtick.PercentFormatter(1.0)) 
          cell_ax[2].yaxis.set_major_formatter(mtick.PercentFormatter(1.0)) 

     else:
          f.suptitle('Histogram of Molecular Cell Types targeting Each Area on {} cell {} (Absolute numbers)'.format(which_post, gid), size=13)

          soma_norm = 1
          dend_norm = 1
          apic_norm = 1

     cell_list = ['Pyr', 'PV', 'SOM', 'other']

     pyr_on_soma   = [i for i in loc_by_morph['soma'] if i == pyr_type]
     PV_on_soma    = [i for i in loc_by_morph['soma'] if all(j not in i for j in not_PVs)]
     SOM_on_soma   = [i for i in loc_by_morph['soma'] if any([j in i for j in SOM_types])]
     other_on_soma = [i for i in loc_by_morph['soma'] if (i not in pyr_on_soma) and (i not in PV_on_soma) and (i not in SOM_on_soma)]
     cell_ax[0].bar(cell_list, [len(pyr_on_soma)/soma_norm, len(PV_on_soma)/soma_norm, len(SOM_on_soma)/soma_norm, len(other_on_soma)/soma_norm])

     pyr_on_dend   = [i for i in loc_by_morph['dend'] if i == pyr_type]
     PV_on_dend    = [i for i in loc_by_morph['dend'] if all(j not in i for j in not_PVs)]
     SOM_on_dend   = [i for i in loc_by_morph['dend'] if any([j in i for j in SOM_types])]
     other_on_dend = [i for i in loc_by_morph['dend'] if (i not in pyr_on_dend) and (i not in PV_on_dend) and (i not in SOM_on_dend)]
     cell_ax[1].bar(cell_list, [len(pyr_on_dend)/dend_norm,  len(PV_on_dend)/dend_norm, len(SOM_on_dend)/dend_norm, len(other_on_dend)/dend_norm])

     pyr_on_apic   = [i for i in loc_by_morph['apic'] if i == pyr_type]
     PV_on_apic    = [i for i in loc_by_morph['apic'] if all(j not in i for j in not_PVs)]
     SOM_on_apic   = [i for i in loc_by_morph['apic'] if any([j in i for j in SOM_types])]
     other_on_apic = [i for i in loc_by_morph['apic'] if (i not in pyr_on_apic) and (i not in PV_on_apic) and (i not in SOM_on_apic)]
     cell_ax[2].bar(cell_list, [len(pyr_on_apic)/apic_norm, len(PV_on_apic)/apic_norm, len(SOM_on_apic)/apic_norm, len(other_on_apic)/apic_norm])
     
     cell_ax[0].set_title('Soma')
     cell_ax[1].set_title('Basal Dendrites')
     cell_ax[2].set_title('Apical Dendrites')

     cell_ax[0].set_ylabel('Count')
     cell_ax[1].set_ylabel('Count')
     cell_ax[2].set_ylabel('Count')
     cell_ax[2].set_xlabel('Molecular Cell Type')

which_post = 'Pyr'
all_cortical_cells=[i[0] for i in cell_details.iterrows()] 
for gid in pyr_gid:

     loc_by_morph = get_LocByMorph(syn_data, all_cortical_cells, cell_details)

     soma_morphs = list(set(loc_by_morph['soma'])) 
     dend_morphs = list(set(loc_by_morph['dend'])) 
     apic_morphs = list(set(loc_by_morph['apic'])) 

     plot_LocByMorph(loc_by_morph, soma_morphs, dend_morphs, apic_morphs)
     plot_LocByMorph(loc_by_morph, soma_morphs, dend_morphs, apic_morphs, as_percent=True)

     plot_LocByCell(loc_by_morph)
     plot_LocByCell(loc_by_morph, as_percent=True)



# Analyze Synapse locations according to GID morphology (given by Oren together with synapse data)

def upload_Morph(post_gid, morph_path):
     h.load_file('import3d.hoc')

     nl = h.Import3d_Neurolucida3() # Create .asc file reader 
     nl.quiet = 1  # Suppress nl.input() output 
     nl.input('thalamocortical_Oren/cortical_synaptic_locations/gid_{}/{}'.format(post_gid, morph_path))   # Define input to reader (morphology .asc file) 
     
     h('objref cell')
     i3d = h.Import3d_GUI(nl, 0)   # Pass loaded morphology to Import3d_GUI tool; with argument #2=0, it won't display the GUI, but will allow use of it's features 
     i3d.instantiate(h.cell)   # Insert morphology to cell template; Instantiation need objref as an argumen, or None 

     # Delete all axons
     for sec in h.allsec(): 
          if 'axon' in sec.name(): 
               h.delete_section(sec=sec) 

def plot_SynDistanceFromSoma(post_name, post_gid, pre_name, chosen_pre_gids, all_post_cell_sections, bins=40):
     D = [] 
     post_soma = [i for i in all_post_cell_sections if 'soma' in i.name()][0]

     for syn in syn_data[post_gid]: 
          if syn_data[post_gid][syn]['pre_cell_id'] in chosen_pre_gids:
               sec = [i for i in all_post_cell_sections if syn_data[post_gid][syn]['post_sec_name'] in i.name()][0] 

               seg = sec(syn_data[post_gid][syn]['post_segx']) 
               D.append(h.distance(seg, post_soma(0.5))) 

     f1, ax1 = plt.subplots()
     f1.suptitle('Histogram of Path Distances From Soma Center ({} to {})'.format(pre_name, post_name))
     ax1.set_title('Postsynaptic GID {}'.format(post_gid))
     ax1.set_xlabel('Distance ($\mu$m)')
     ax1.set_ylabel('Count')
     ax1.hist(D, bins=bins)

     f2, ax2 = plt.subplots()
     f2.suptitle('Histogram of Path Distances From Soma Center ({} to {})'.format(pre_name, post_name))
     ax2.set_title('Postsynaptic GID {}'.format(post_gid))
     ax2.set_xlabel('Distance ($\mu$m)')
     ax2.set_ylabel('Percent out of Total Synapses')
     H, B = np.histogram(D, bins=bins)
     H = [100*i/sum(H) for i in H]
     ax2.bar(B[:-1], H, align='edge', color='orange', width=np.diff(B)[0])
     vals = ax2.get_yticks() 
     ax2.set_yticklabels(['{}%'.format(x) for x in vals])

     longest_basal_D, farthest_basal = 0, None 
     for sec in all_post_cell_sections: 
          if 'dend' in sec.name(): 
               if h.distance(sec(1),post_soma(0.5))>longest_basal_D: 
                    longest_basal_D = h.distance(sec(1),post_soma(0.5)) 
                    farthest_basal = sec 

     ax1.axvline(longest_basal_D, color='k', LineStyle='--', label='Farthest on Basal Tree')
     ax2.axvline(longest_basal_D, color='k', LineStyle='--', label='Farthest on Basal Tree')
     ax2.legend()
     ax1.legend()

def plot_SynsOnMorph(post_syn_data, chosen_pre_gids, all_post_cell_sections):     

     synapses = []
     for syn in post_syn_data:
          if post_syn_data[syn]['pre_cell_id'] in chosen_pre_gids:
               sec = [i for i in all_post_cell_sections if post_syn_data[syn]['post_sec_name'] in i.name()][0]
               seg = sec(post_syn_data[syn]['post_segx'])
               synapses.append(h.ProbAMPANMDA_EMS(seg))

     S = h.Shape()
     for syn in synapses:
          S.point_mark(syn, 2)

     return synapses


post_gid = pyr_gid[0]
post_name = 'Pyr'
pre_gids = PV_gids
pre_name = 'PV'

gid_folder = [i for i in os.listdir('thalamocortical_Oren/cortical_synaptic_locations') if str(post_gid) in i]
assert len(gid_folder)==1, '2 possible folders for gid!'
gid_folder = gid_folder[0]

morph_path = [i for i in os.listdir('thalamocortical_Oren/cortical_synaptic_locations/{}'.format(gid_folder)) if 'asc' in i]
assert len(morph_path)==1, '2 possible asc files for gid!'
morph_path = morph_path[0]

upload_Morph(post_gid, morph_path)
plot_SynDistanceFromSoma(post_name, post_gid, pre_name, pre_gids, h.allsec())
synapses = plot_SynsOnMorph(syn_data[post_gid], pre_gids, h.allsec())
















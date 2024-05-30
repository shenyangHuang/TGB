""" pie charts, mrr per relation charts
"""

## imports
import numpy as np
import sys
import os
import os.path as osp
tgb_modules_path = osp.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(tgb_modules_path)
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
import stats_figures.dataset_utils as du


# specify params
names = [ 'tkgl-polecat', 'tkgl-icews', 'tkgl-wikidata', 'tkgl-smallpedia','tkgl-polecat'] #'tkgl-polecat','tkgl-smallpedia',  'tkgl-yago',  'tkgl-icews' ,'tkgl-smallpedia','thgl-myket','tkgl-yago',  'tkgl-icews','thgl-github', 'thgl-forum', 'tkgl-wikidata']
methods = ['recurrency', 'regcn', 'cen'] #'recurrency'
colortgb = '#60ab84' #tgb logo colors
colortgb2 = '#eeb641'
colortgb3 = '#dd613a'
head_tail_flag = False # if true, the head and tail of the relation are shown in the plot, otherwise just the mean across both directions

colors = [colortgb,colortgb2,colortgb3]  # from tgb logo
colors2= ['#8e0152', '#c51b7d', '#de77ae', '#f1b6da', '#fde0ef', '#f7f7f7', '#e6f5d0', '#b8e186', '#7fbc41', '#4d9221', '#276419']
# from https://colorbrewer2.org/#type=diverging&scheme=PiYG&n=11 color blind friendly 

capsize=1.5
capthick=1.5
elinewidth=1.5
occ_threshold = 5
k=10 # how many slices in the cake +1
plots_flag = True
ylimdict = {'tkgl-polecat': 0.25, 'tkgl-icews':0.6, 'tkgl-smallpedia': 1.01} # for the mrr charts the upper mrr limit

overall_min = -1 # for the correlation matrix colorbar
overall_max =1 # for the correlation matrix colorbar
num_rels_plot = 10 # how many relations to we want to plot in the mrr chart
i = 0
plot_values_list = []
plot_names_multi_line_list =[]
for dataset_name in names:
    # some directory stuff
    modified_dataset_name = dataset_name.replace('-', '_')
    current_dir = os.path.dirname(os.path.abspath(__file__))

    stats_dir = os.path.join( current_dir,dataset_name,'stats')
    tgb_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    figs_dir = os.path.join(current_dir,dataset_name,'figs_rel')
    stats_df = pd.read_csv(os.path.join(stats_dir, f"relation_statistics_{dataset_name}.csv"))

    # Create the 'figs' directory if it doesn't exist
    if not os.path.exists(figs_dir):
        os.makedirs(figs_dir)
    stats_dir = os.path.join( current_dir,dataset_name,'stats')
    if not os.path.exists(stats_dir):
        os.makedirs(stats_dir)

    ### A pie charts #plot top k relations accordung to the number of occurences plus a slice for "others"
    plot_names = list(stats_df['rel_string_word'].iloc[:k]) 
    plot_values = list(stats_df['number_total_occurences'].iloc[:k])
    all_others = np.sum(stats_df['number_total_occurences'].iloc[k:]) #slice for "others" (sum of all other relations occurences)
    plot_values.append(all_others)
    plot_names.append('Others')
    # for the pie chart labels to be more readable (i.e. force line break if words are long)
    plot_names_multi_line= []
    for name in plot_names: # add some \n to make the labels more fittable to the pie chart
        if type(name) == str:
            words = name.split()
            newname = words[0]
            if len(words) > 1:
                for i in range(len(words)-1):
                    if not '(' in words[i+1]:
                        if len(words[i]) > 3:
                            newname+='\n'
                        else:
                            newname+=' ' 
                        newname+=words[i+1]
        else:
            newname = str(name) #then only plot the int as is. 
        plot_names_multi_line.append(newname)

    num_slices = len(plot_names)
    plt.figure(figsize=(7, 7))
    wedges, texts, autotexts =plt.pie(plot_values,autopct=lambda pct: f"{pct:.0f}%" if pct > 1.5 else '', startangle=140, colors=colors2, labeldistance=2.2) #repeated_colors)
    # Increase the font size of the percentage values
    for autotext in autotexts:
        autotext.set_fontsize(15)
    plt.axis('equal')  
    # Move the percentage labels further outside
    for autotext, wedge in zip(autotexts, wedges):
        angle = (wedge.theta2 - wedge.theta1) / 2 + wedge.theta1
        x = np.cos(np.deg2rad(angle))
        y = np.sin(np.deg2rad(angle))
        distance = 0.85  # Adjust this value to move the labels further or closer to the center
        autotext.set_position((x * distance, y * distance))
    # Set the labels for each pie slice
    plt.legend(wedges, plot_names_multi_line, loc="center left", bbox_to_anchor=(1, 0, 0.5, 1), fontsize=14)
    save_path = (os.path.join(figs_dir, f"rel_pie_{dataset_name}.png"))
    plt.savefig(save_path, bbox_inches='tight')
    save_path = (os.path.join(figs_dir, f"rel_pie_{dataset_name}.pdf"))
    plt.savefig(save_path, bbox_inches='tight')

    if dataset_name == 'tkgl-wikidata': #then we do not want to plot the mrr for the relations
        continue

    ### B plot the mrr for each relation for each method, different color for different number of occurences or for different recurrency degree
    
    # prepare the dataframe: only take the top ten relations according to number of occurences and sort by recurrency degree
    # we use selected_df_sorted to plot the relations in the order of recurrency degree
    rels_sorted =  np.array(stats_df['relation'])[0:num_rels_plot]
    mask = stats_df['relation'].isin(rels_sorted)
    selected_df = stats_df[mask] #only the parts of the dataframe that contain the top ten relations according to number of occurences
    selected_df_sorted = selected_df.sort_values(by='recurrency_degree', ascending=False) # Sort selected_df by 'recurrency_degree' column in descending order
    rels_to_plot = list(selected_df_sorted['relation'])
    labels = np.array(selected_df_sorted['relation'])# only plotting the id for space reasons
    mrr_per_rel_freq = [] # list of mrr values for each relation - three lists for three methods
    mrr_per_rel_freq2 = []
    mrr_per_rel_freq3 = []
    lab = []
    lab_ht = []
    lab_rel = []
    # rel_oc_dict[rel] = count_occurrences
    count_occurrences_sorted = []
    rec_degree_sorted = []
    for index, r in enumerate(rels_to_plot):   
        if head_tail_flag:
            lab_ht.append('h')
            lab_ht.append('t')
            lab_rel.append(str(labels[index])+'    ') # add spaces to make the labels longer
        else:
            lab_rel.append(str(labels[index])+'') # add spaces to make the labels longer
        
        lab.append(labels[index])
        if head_tail_flag: # if we do head and tail separately we need the value for head and tail direction
            mrr_per_rel_freq.append(selected_df_sorted['recurrency_head'].iloc[index])
            mrr_per_rel_freq.append(selected_df_sorted['recurrency_tail'].iloc[index])
            mrr_per_rel_freq2.append(selected_df_sorted['regcn_head'].iloc[index])
            mrr_per_rel_freq2.append(selected_df_sorted['regcn_tail'].iloc[index])
            mrr_per_rel_freq3.append(selected_df_sorted['cen_head'].iloc[index])
            mrr_per_rel_freq3.append(selected_df_sorted['cen_tail'].iloc[index])
            count_occurrences_sorted.append(selected_df_sorted['number_total_occurences'].iloc[index])#append twice for head and tail
            count_occurrences_sorted.append(selected_df_sorted['number_total_occurences'].iloc[index])
            rec_degree_sorted.append(selected_df_sorted['recurrency_degree'].iloc[index]) #append twice for head and tail
            rec_degree_sorted.append(selected_df_sorted['recurrency_degree'].iloc[index])
        else:# if we do  NOT head and tail separately we need the mean value for head and tail direction
            mrr_per_rel_freq.append(np.mean([selected_df_sorted['recurrency_head'].iloc[index], selected_df_sorted['recurrency_tail'].iloc[index]]))
            mrr_per_rel_freq2.append(np.mean([selected_df_sorted['regcn_head'].iloc[index],selected_df_sorted['regcn_tail'].iloc[index]]))
            mrr_per_rel_freq3.append(np.mean([selected_df_sorted['cen_head'].iloc[index], selected_df_sorted['cen_tail'].iloc[index]]))
            count_occurrences_sorted.append(selected_df_sorted['number_total_occurences'].iloc[index])#append twice for head and tail
            rec_degree_sorted.append(selected_df_sorted['recurrency_degree'].iloc[index])

    # these are the x-values of the ticks. in case we plot head and tail separately, we need to have two ticks per relation
    x_values = []
    x_values_rel = []
    for i in range(0,num_rels_plot):
        if head_tail_flag:
            x_values.append(i*2+0.4)
            x_values.append(i*2+0.8)
        else:
            x_values.append(i*2+0.4)
        x_values_rel.append(i*2+0.4)

    lab_lines = lab_rel #labels, for now
    a = count_occurrences_sorted 

    # version 1) colors are based on the reucrrency degree
    plt.figure()
    sca = plt.scatter(x_values, mrr_per_rel_freq2,  marker='p',s=150,   c = rec_degree_sorted, alpha=1, edgecolor='grey',  cmap='jet',  norm=Normalize(vmin=0, vmax=1), label='REGCN')           # cmap='gist_rainbow',
    sca = plt.scatter(x_values, mrr_per_rel_freq3 , marker='*',s=150,   c = rec_degree_sorted, alpha=1,  edgecolor='grey', cmap='jet',  norm=Normalize(vmin=0, vmax=1), label='CEN')      
    sca = plt.scatter(x_values, mrr_per_rel_freq,   marker='o',s=60,    c = rec_degree_sorted, alpha=1,  edgecolor='grey', cmap='jet', norm=Normalize(vmin=0, vmax=1), label='Recurrency Baseline')
    plt.ylabel('MRR', fontsize=14) 
    plt.xlabel('Relation', fontsize=14) 
    plt.legend(fontsize=14)
    cbar =plt.colorbar(sca)
    plt.ylim([0,ylimdict[dataset_name]])
    cbar.ax.yaxis.label.set_color('gray')

    if head_tail_flag:
        plt.xticks(x_values, lab_ht, size=13) #, verticalalignment="center") #  ha='right', 
        plt.xticks(x_values_rel, lab_lines,  size=14, minor=True)
        plt.tick_params(axis='x', which='minor',  rotation=90,  length=0)
    else:
        plt.xticks(x_values_rel, lab_lines,  size=14)
        plt.tick_params(axis='x',  rotation=90,  length=0)
    plt.yticks(size=13)
    # Create a locator for the second set of x-ticks
    # plt.secondary_xaxis('top', x_values_rel)

    plt.grid()
    save_path = (os.path.join(figs_dir, f"rel_mrrperrel_recdeg_{dataset_name}.png"))
    plt.savefig(save_path, bbox_inches='tight')
    save_path = (os.path.join(figs_dir, f"rel_mrrperrel_recdeg_{dataset_name}.pdf"))
    plt.savefig(save_path, bbox_inches='tight')
    print('saved')


    # version 2) colors are the number of occurences
    plt.figure()
    sca = plt.scatter(x_values, mrr_per_rel_freq2, marker='p',s=150,   c = a, alpha=1, edgecolor='grey', norm=LogNorm(), cmap='jet', label='REGCN')          
    sca = plt.scatter(x_values, mrr_per_rel_freq3 , marker='*',s=150,   c = a, alpha=1, edgecolor='grey', norm=LogNorm(), cmap='jet', label='CEN')      
    sca = plt.scatter(x_values, mrr_per_rel_freq,   marker='o',s=60,    c = a, alpha=1, edgecolor='grey', norm=LogNorm(), cmap='jet', label='Recurrency Baseline')
    plt.ylabel('MRR', fontsize=14) 
    plt.xlabel('Relation', fontsize=14) 
    plt.legend(fontsize=14)
    cbar =plt.colorbar(sca)
    plt.ylim([0,ylimdict[dataset_name]])
    cbar.ax.yaxis.label.set_color('gray')

    plt.xticks(x_values, lab_ht, size=13) #, verticalalignment="center") #  ha='right', 
    plt.yticks(size=13)
    # Create a locator for the second set of x-ticks
    # plt.secondary_xaxis('top', x_values_rel)
    plt.xticks(x_values_rel, lab_lines,  size=14, minor=True)
    plt.tick_params(axis='x', which='minor',  rotation=90,  length=0)
    plt.grid()
    save_path = (os.path.join(figs_dir, f"rel_mrrperrel_occ_{dataset_name}.png"))
    plt.savefig(save_path, bbox_inches='tight')
    
    ### now we plot all sorts of correlation matrix. I specify different columns for the different plots    
    df = stats_df[['recurrency_degree', 'direct_recurrency-degree', 'recurrency_tail', 'recurrency_head', 'regcn_tail', 'regcn_head', 'cen_tail', 'cen_head']]
    corrmat= df.corr()
    f = plt.figure(figsize=(19, 15))
    plt.matshow(corrmat, fignum=f.number,  vmin=overall_min, vmax=overall_max)
    plt.xticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=16, rotation=90)
    plt.yticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=16)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=16)
    save_path = (os.path.join(figs_dir, f"corr_rec_meth_{dataset_name}.png"))
    plt.savefig(save_path, bbox_inches='tight')
    
    df = stats_df[['consecutiveness_value', 'recurrency_tail', 'recurrency_head', 'regcn_tail', 'regcn_head', 'cen_tail', 'cen_head']]
    corrmat= df.corr()
    f = plt.figure(figsize=(19, 15))
    plt.matshow(corrmat, fignum=f.number,  vmin=overall_min, vmax=overall_max)
    plt.xticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=16, rotation=90)
    plt.yticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=16)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=16)
    save_path = (os.path.join(figs_dir, f"corr_con_meth_{dataset_name}.png"))
    plt.savefig(save_path, bbox_inches='tight')
    
    df = stats_df[['recurrency_degree', 'direct_recurrency-degree', 'consecutiveness_value', 'mean_occurence_per_triple','number_total_occurences',  'recurrency_tail', 'recurrency_head', 'regcn_tail', 'regcn_head', 'cen_tail', 'cen_head']]
    corrmat= df.corr()
    f = plt.figure(figsize=(19, 15))
    plt.matshow(corrmat, fignum=f.number,  vmin=overall_min, vmax=overall_max)
    plt.xticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=16, rotation=90)
    plt.yticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=16)
    for i in range(corrmat.shape[0]):
        for j in range(corrmat.shape[1]):
            plt.text(j, i, "{:.2f}".format(corrmat.iloc[i, j]), ha='center', va='center', color='black', fontsize=16)
    cb = plt.colorbar()
    # fig.colorbar(cax, ticks=[-1,0,1], shrink=0.8)
    cb.ax.tick_params(labelsize=16)    
    # Plot the correlation matrix
    save_path = (os.path.join(figs_dir, f"corr_all_meth_{dataset_name}.png"))
    plt.savefig(save_path, bbox_inches='tight')
    plt.close('all')



print('done with creating the figs')



## imports
import numpy as np
import sys
import os
import os.path as osp
from pathlib import Path
tgb_modules_path = osp.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(tgb_modules_path)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#internal imports 
from tgb.linkproppred.dataset import LinkPropPredDataset 
from tgb_modules.tkg_utils import reformat_ts
import stats_figures.dataset_utils as du

# specify params
names = ['thgl-myket']  #'thgl-software', 'thgl-github', 'thgl-forum'
#'tkgl-polecat','tkgl-smallpedia',  'tkgl-yago',  'tkgl-icews' ,'tkgl-smallpedia','thgl-myket','tkgl-yago',  'tkgl-icews','thgl-github', 'thgl-forum', 'tkgl-wikidata']
colortgb = '#60ab84' #tgb logo colrs
colortgb2 = '#eeb641'
colortgb3 = '#dd613a'

colors = [colortgb,colortgb2,colortgb3]  # from tgb logo
capsize=1.5
capthick=1.5
elinewidth=1.5
occ_threshold = 5
k=10 # how many slices in the cake (+1 will be added for "others")
plots_flag = True
# run through each datasest
for dataset_name in names:
    ############################## LOAD DATA ##############################
    print(dataset_name)    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Navigate one folder up

    figs_dir = os.path.join( current_dir, dataset_name, 'figs')
    # Create the 'figs' directory if it doesn't exist
    if not os.path.exists(figs_dir):
        os.makedirs(figs_dir)
    stats_dir = os.path.join( current_dir, dataset_name, 'stats')
    if not os.path.exists(stats_dir):
        os.makedirs(stats_dir)

    dataset = LinkPropPredDataset(name=dataset_name, root="datasets", preprocess=True)

    relations = dataset.edge_type
    num_rels = dataset.num_rels
    if 'tkgl' in dataset_name:
        num_rels_without_inv = int(num_rels/2)
    else:
        num_rels_without_inv = num_rels

    rels = np.arange(0,num_rels)
    subjects = dataset.full_data["sources"]
    objects= dataset.full_data["destinations"]
    num_nodes = dataset.num_nodes 
    timestamps_orig = dataset.full_data["timestamps"]
    timestamps = reformat_ts(timestamps_orig, dataset_name) # stepsize:1
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_dir = os.path.join( current_dir, dataset_name)
    np.savetxt(csv_dir +"/"+dataset_name+"timestamps.csv", timestamps,fmt='%i', delimiter=",")
    all_quads = np.stack((subjects, relations, objects, timestamps, timestamps_orig), axis=1)
    train_data = all_quads[dataset.train_mask]
    val_data = all_quads[dataset.val_mask]
    test_data = all_quads[dataset.test_mask]

    # Read the CSV file into a DataFrame
    rel_type2id_dict = {}
    rel_id2type_dict = {}
    if 'wikidata' in dataset_name or 'smallpedia' in dataset_name: #otherwise I add it manually
        csv_dir = os.path.join( current_dir, dataset_name, dataset_name+'_edgelist.csv')
        df = pd.read_csv(csv_dir)

        # Create a dictionary mapping the entries in the 'relation_type' column to IDs
        rel_type2id_dict = {}
        rel_id2type_dict = {}
        unique_relation_types = df['relation_type'].unique()
        for i, relation_type in enumerate(unique_relation_types):
            rel_type2id_dict[relation_type] = i   # IDs start from 0
            rel_id2type_dict[i] = relation_type

    # create dictionaries that contain different combis of key:rel, values:[s,o,t] and so on
    timestep_range = 1+np.max(timestamps) - np.min(timestamps)
    all_possible_timestep_indices = [i for i in range(timestep_range)]
    ts_all = du.TripleSet()
    ts_all.add_triples(all_quads, num_rels_without_inv, timestep_range)
    ts_all.compute_stat()
    ts_test = du.TripleSet()
    ts_test.add_triples(test_data, num_rels_without_inv, timestep_range)
    ts_test.compute_stat()

    ############################## Compute Stats ##############################
    # compute the number of ocurances of each relation
    rels_occurences ={}
    for rel in ts_all.r_2_triple.keys():
        rels_occurences[rel] = len(ts_all.r_2_triple[rel])

    # Sort the dictionary by values in descending order
    sorted_dict = dict(sorted(rels_occurences.items(), key=lambda item: item[1], reverse=True))

    # Take the top k key-value pairs and sum up their values
    top_k = dict(list(sorted_dict.items())[:k]) # highest k relations
    bad_k = dict(list(sorted_dict.items())[-k:]) # lowest k relations
    plot_names = du.set_plot_names(top_k, sorted_dict, dataset_name, rel_id2type_dict) #names to be included in the plot  
     
    num_occurences_dict = {}
    mean_std_max_min_dict = {}
    high_occurences = {}
    low_occurences = {}
    done_dict ={}
    # compute the mean, std, max, min, median of the number of occurences of triples for each relation
    for rel in ts_all.r_2_triple.keys():
        num_occurences_dict[rel] = []
        for triple in ts_all.r_2_triple[rel]:
            (s,r,o,t) = triple[0], rel, triple[1], triple[2]
            if (s,r,o) in done_dict.keys():
                continue
            else:
                done_dict[(s,r,o)] = 1
                count_num_occurences = len(ts_all.sub_rel_2_obj[s][r][o])
                num_occurences_dict[rel].append(count_num_occurences)
        mean_std_max_min_dict[rel] = (np.mean(num_occurences_dict[rel]), np.std(num_occurences_dict[rel]), np.max(num_occurences_dict[rel]), np.min(num_occurences_dict[rel]), np.median(num_occurences_dict[rel]))
        if mean_std_max_min_dict[rel][0] < occ_threshold:
            low_occurences[rel] = rels_occurences[rel]
        else:
            high_occurences[rel] = rels_occurences[rel]

    # compute for each relation the max consecutive timesteps of each triple
    lists_per_rel = {}
    mean_per_rel = {}
    for rel in ts_all.rel_sub_obj_t.keys():
        ts_lists = ts_all.create_timestep_lists(ts_all.rel_sub_obj_t[rel])
        for list_r in ts_lists:
            max_cn = du.max_consecutive_numbers(list_r)

            if rel not in lists_per_rel.keys():
                lists_per_rel[rel] = [max_cn]
            else:
                lists_per_rel[rel].append(max_cn)
        mean_per_rel[rel] = np.mean(lists_per_rel[rel])


    #only for the most prominent relations
    statistics_dict_prominent = {}
    for rel in top_k.keys():
        if 'wiki' in dataset_name or 'small' in dataset_name:
            rel_key = rel_id2type_dict[rel]
            # print(rel_key)
            statistics_dict_prominent[rel_key] = mean_std_max_min_dict[rel]
        else:
            statistics_dict_prominent[rel] = mean_std_max_min_dict[rel]


    ## create dataframe
    # each line in the dataframe is one relation
    # columns: [relation,  recurrency degree, direct recurrency degree, consecutiveness value,number of distinct triples,
    # number of total occurences,  mean_occurence per_triple, max_occurence per_triple, min_occurence per_triple, median_occurence per_triple]
    # df = pd.DataFrame(columns=['relation', 'rel_string_id', 'rel_string_word', 'recurrency_degree', 'direct_recurrency-degree', 'consecutiveness_value', \
    #                            'number_distinct_triples', 'number_total_occurences', 'mean_occurence_per_triple', \
    #                             'max_occurence_per_triple', 'min_occurence_per_triple', 'median_occurence_per_triple'])
    # for each relation: compute stats and add line to dataframe
    ## compute recurrency degree
    new_rows = []
    for rel in ts_all.r_2_triple.keys():
        if 'wiki' in dataset_name or 'small' in dataset_name:
            rel_string_id = rel_id2type_dict[rel]
            try:
                word = du.fetch_wikidata_property_name(rel_string_id)
            except:
                word = rel_string_id
        else:
            rel_string_id = str(rel)
            word = str(rel)
        if rel in ts_test.r_2_triple.keys():
            recurrency_degree, direct_recurrency_degree = du.compute_rec_drec(ts_test.r_2_triple[rel],rel, ts_all)
        else:
            recurrency_degree = 0
            direct_recurrency_degree = 0
            consecutiveness_value = 0
        consecutiveness_value = du.compute_consecutiveness(ts_all.rel_sub_obj_t[rel],ts_all) # TODO: implement this function
        number_distinct_triples = du.compute_number_distinct_triples(ts_all.rel_sub_obj_t[rel]) # TODO: implement this function
        number_total_occurences = len(ts_all.r_2_triple[rel]) # for how many triples does the relation occur in the dataset
        mean_occurence_per_triple = mean_std_max_min_dict[rel][0]
        max_occurence_per_triple = mean_std_max_min_dict[rel][2]
        min_occurence_per_triple = mean_std_max_min_dict[rel][3]
        median_occurence_per_triple = mean_std_max_min_dict[rel][4]

        data = {'relation': rel, 'rel_string_id':rel_string_id, 'rel_string_word':word, 
                        'recurrency_degree': recurrency_degree, 'direct_recurrency-degree': direct_recurrency_degree, 
                        'consecutiveness_value': consecutiveness_value, 
                        'number_distinct_triples': number_distinct_triples,
                        'number_total_occurences': number_total_occurences, 
                        'mean_occurence_per_triple': mean_occurence_per_triple, 
                        'max_occurence_per_triple': max_occurence_per_triple, 
                        'min_occurence_per_triple': min_occurence_per_triple, 
                        'median_occurence_per_triple': median_occurence_per_triple} 
        new_rows.append(data)

    df = pd.DataFrame(new_rows)
    # df = pd.concat([df, new_df], ignore_index=True)

    df_sorted = df.sort_values(by='number_total_occurences', ascending=False).reset_index(drop=True)


    # save dataframe to csv
    df_sorted.to_csv(os.path.join(stats_dir, f"relation_statistics_{dataset_name}.csv"), index=False)
    

    # ###################### Figures ##############################  I moved them to create_relation_figures.py
    # ##PIE CHART
    # # Repeat the colors to match the number of slices
    # if plots_flag:
    #     num_slices = len(plot_names)
    #     repeated_colors = (colors * ((num_slices // len(colors)) + 1))[:num_slices]
    #     plt.figure(figsize=(8, 8))
    #     plt.pie(plot_names.values(), labels=plot_names.keys(), autopct='%1.f%%', startangle=140, 
    #     colors=repeated_colors)
    #     #plt.title(f'Pie Chart of Top {k} Relations and "Others"')
    #     plt.axis('equal')  
    #     save_path = (os.path.join(figs_dir, f"rel_pie_{dataset_name}.png"))
    #     plt.savefig(save_path, bbox_inches='tight')
        
    #     ## TRIPLES PER RELATION
    #     plt.figure()
    #     plt.bar(rels_occurences.keys(), rels_occurences.values(), color=colortgb)
    #     plt.xlabel('Relation')
    #     plt.ylabel('Number of Triples')
    #     #plt.title('Number of Triples per Relation')
    #     save_path = (os.path.join(figs_dir, f"rel_tripperrel_{dataset_name}.png"))
    #     plt.savefig(save_path, bbox_inches='tight')

    #     ## NUMBER OF OCCURENCES OF TRIPLES PER RELATION
    #     plt.figure()
    #     mins = np.array([x[3] for x in statistics_dict_prominent.values()])
    #     maxs = np.array([x[2] for x in statistics_dict_prominent.values()])
    #     mean = np.array([x[0] for x in statistics_dict_prominent.values()])
    #     std = np.array([x[1] for x in statistics_dict_prominent.values()])
    #     # plt.bar(mean_max_min_dict.keys(), [x[0] for x in mean_max_min_dict.values()], color=colortgb)
    #     plt.scatter(statistics_dict_prominent.keys(), [x[0] for x in statistics_dict_prominent.values()], label ='mean value', color=colortgb)
    #     plt.scatter(statistics_dict_prominent.keys(), [x[4] for x in statistics_dict_prominent.values()], label ='median value', color='orange')
    #     # plt.errorbar(mean_std_max_min_dict.keys(), mean,  yerr=std, fmt='none', alpha=0.9, color='grey',capsize=capsize, capthick=capthick, elinewidth=elinewidth, label='Std')
    #     plt.errorbar(statistics_dict_prominent.keys(), maxs,  yerr=[maxs-mins, maxs-maxs], fmt='none', alpha=0.9, color='grey',capsize=capsize, capthick=capthick, elinewidth=elinewidth, label='Min-Max Range')
    #     plt.xlabel('Relation')
    #     plt.ylabel('Mean Number of Occurences of [subject, object]') 
    #     #plt.title('Mean Number of Occurences of [subject, object] per Relation')
    #     plt.legend()
    #     #plt.yscale('log')
    #     save_path = (os.path.join(figs_dir, f"rel_mean_occurences_{dataset_name}.png"))
    #     plt.savefig(save_path, bbox_inches='tight')

    #     ## bar plot that shows how many relations belong to the low occurence category vs high occurence category
    #     plt.figure()
    #     plt.bar(['Low Occurence', 'High Occurence'], [len(low_occurences), len(high_occurences)], color=colortgb)
    #     plt.xlabel('Occurence Category')
    #     plt.ylabel('Number of Relations')
    #     #plt.title('Number of Relations in Low and High Occurence Categories')
    #     save_path = (os.path.join(figs_dir, f"rel_occurence_categories_{dataset_name}.png"))
    #     plt.savefig(save_path, bbox_inches='tight')

    #     ## bar plot that shows the number of triples in each occurence category
    #     plt.figure()
    #     plt.bar(['Low Occurence', 'High Occurence'], [sum([num_oc for num_oc in low_occurences.values()]), sum([num_oc for num_oc in high_occurences.values()]),], color=colortgb)
    #     plt.xlabel('Occurence Category')
    #     plt.ylabel('Number of Triples')
    #     #plt.title('Number of Triples in Low and High Occurence Categories')
    #     save_path = (os.path.join(figs_dir, f"rel_occurence_triples_categories_{dataset_name}.png"))
    #     plt.savefig(save_path, bbox_inches='tight')

    #     ## bar plot that shows the mean consecutive timesteps of each relation
    #     plt.figure()
    #     plt.bar(mean_per_rel.keys(), mean_per_rel.values(), color=colortgb)
    #     save_path = (os.path.join(figs_dir, f"rel_conperrel_{dataset_name}.png"))
    #     plt.savefig(save_path, bbox_inches='tight')

print('done')
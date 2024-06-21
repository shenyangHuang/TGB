"""
 from paper: "History repeats itself: A Baseline for Temporal Knowledge Graph Forecasting" 
Julia Gastinger, Christian Meilicke, Federico Errica, Timo Sztyler, Anett Schuelke, Heiner Stuckenschmidt (IJCAI 2024)

@inproceedings{gastinger2024baselines,
  title={History repeats itself: A Baseline for Temporal Knowledge Graph Forecasting},
  author={Gastinger, Julia and Meilicke, Christian and Errica, Federico and Sztyler, Timo and Schuelke, Anett and Stuckenschmidt, Heiner},
  booktitle={33nd International Joint Conference on Artificial Intelligence (IJCAI 2024)},
  year={2024},
  organization={International Joint Conferences on Artificial Intelligence Organization}
}

"""
import numpy as np
from collections import Counter
import ray
from modules.tkg_utils import create_scores_array

@ray.remote
def baseline_predict_remote(num_queries, test_data, all_data, window, basis_dict, num_nodes, 
                num_rels, lmbda_psi, alpha, evaluator,first_test_ts, neg_sampler, split_mode='test'):
    """
    Apply baselines psi and xi (multiprocessing possible). See baseline_predict for more details."""

    return baseline_predict(num_queries, test_data, all_data, window, basis_dict, num_nodes, 
                num_rels, lmbda_psi, alpha,  evaluator,first_test_ts, neg_sampler, split_mode)


def baseline_predict(num_queries, test_data, all_data, window, basis_dict, num_nodes, 
                num_rels, lmbda_psi, alpha, evaluator,first_test_ts, neg_sampler, split_mode='test'):
    """
    Apply baselines psi and xi and compute scores and mrr per test or valid query (multiprocessing possible).

    Parameters:
        num_queries (int): minimum number of queries for each process
        test_data (np.array): test quadruples (only used in single-step prediction, depending on window specified);
            including inverse quadruples for subject prediction
        all_data (np.array): train valid and test quadruples (test only used in single-step prediction, depending 
            on window specified); including inverse quadruples  for subject prediction
        window: int, specifying which values from the past can be used for prediction. 0: all edges before the test 
        query timestamp are included. -2: multistep. all edges from train and validation set used. as long as they are 
        < first_test_query_ts. Int n > 0, all edges within n timestamps before the test query timestamp are included.
        basis_dict (dict): keys: rel_ids; specifies the predefined rules for each relation. 
            in our case: head rel = tail rel, confidence =1 for all rels in train/valid set
        score_func_psi (method): method to use for computing time decay for psi
        num_nodes (int): number of nodes in the dataset
        num_rels (int): number of relations in the dataset
        lambda_psi (float): parameter for time decay function for baselinepsi. 0: no decay, >1 very steep decay
        alpha (float): parameter, weight to combine the scores from psi and xi. alpha*scores_psi + (1-alpha)*scores_xi
        evaluator (method): method to compute mrr and hits
        first_test_ts (int): timestamp of the first test query
        neg_sampler (NegSampler): negative sampler
        split_mode (str): 'test' or 'valid'
    Returns:
        performance_list and hits_list (one entry per query)
    """
    num_this_queries = len(test_data)
    cur_ts = test_data[0][3]
    first_test_query_ts = first_test_ts #test_data[0][3]
    edges, all_data_ts = get_window_edges(all_data, cur_ts, window, first_test_query_ts) # get for the current 
                            # timestep all previous quadruples per relation that fullfill time constraints

    rel_obj_dist_cur_ts = update_distributions(edges, num_rels)
    if len(all_data_ts) >0:
        sum_delta_t = update_delta_t(np.min(all_data_ts[:,3]), np.max(all_data_ts[:,3]), cur_ts, lmbda_psi)

    predictions_xi=np.zeros(num_nodes) 
    predictions_psi=np.zeros(num_nodes)
    # if num_queries != len(test_queries_idx):
        # print('num_queries not equal to len(test_queries_idx)')
        
    hits_list = [0] * num_this_queries #len(test_queries_idx)
    perf_list = [0] * num_this_queries #* len(test_queries_idx)
    for j in range(num_this_queries):   
        neg_sample_el =  neg_sampler.query_batch(np.expand_dims(np.array(test_data[j,0]), axis=0), 
                                                np.expand_dims(np.array(test_data[j,2]), axis=0), 
                                                np.expand_dims(np.array(test_data[j,4]), axis=0), 
                                                np.expand_dims(np.array(test_data[j,1]), axis=0), 
                                                split_mode)[0]
        pos_sample_el =  test_data[j,2]      
        test_query = test_data[j]
        assert(pos_sample_el == test_query[2])
        cands_dict = dict() 
        cands_dict_psi = dict() 
        # 1) update timestep and known triples
        if test_query[3] != cur_ts: # if we have a new timestep
            cur_ts = test_query[3]
            edges, all_data_ts = get_window_edges(all_data, cur_ts, window, first_test_query_ts) # get for the current 
            # timestep all previous quadruples per relation that fullfill time constraints
            # update the object and rel-object distritbutions to take into account what timesteps to use
            if window > -1: #otherwise: multistep, we do not need to update
                rel_obj_dist_cur_ts = update_distributions( edges, num_rels)

            if len(all_data_ts) >0:
                if window > -1: #otherwise: multistep, we do not need to update
                    sum_delta_t = update_delta_t(np.min(all_data_ts[:,3]), np.max(all_data_ts[:,3]), cur_ts, lmbda_psi)
                        
        #### BASELINE  PSI
        # 2) apply rules for relation of interest, if we have any
        if str(test_query[1]) in basis_dict: # do we have rules for the given relation?                
            walk_edges = match_body_relations(basis_dict[str(test_query[1])][0], edges, test_query[0]) 
                                # Find quadruples that match the rule (starting from the test query subject)
                                # Find edges whose subject match the query subject and the relation matches
                                # the relation in the rule body. np array with [[sub, obj, ts]]
            if 0 not in [len(x) for x in walk_edges]: # if we found at least one potential rule                        
                cands_dict_psi = get_candidates_psi(walk_edges[0][:,1:3], cur_ts, cands_dict, lmbda_psi, sum_delta_t)
                if len(cands_dict_psi)>0:                
                    # predictions_psi = create_scores_tensor(cands_dict_psi, num_nodes)
                    predictions_psi = create_scores_array(cands_dict_psi, num_nodes)

        #### BASELINE XI      
        predictions_xi = create_scores_array(rel_obj_dist_cur_ts[test_query[1]], num_nodes)
        # predictions_xi = create_scores_tensor(rel_obj_dist_cur_ts[test_query[1]], num_nodes)

        #### Combine Both
        predictions_all = 1000*alpha*predictions_psi + 1000*(1-alpha)*predictions_xi           
        # predictions_of_interest_pos = predimctions_all[pos_sample_el].unsqueeze(0)
        predictions_of_interest_pos = np.array(predictions_all[pos_sample_el])
        predictions_of_interest_neg = predictions_all[neg_sample_el]
        input_dict = {
            "y_pred_pos": predictions_of_interest_pos,
            "y_pred_neg": predictions_of_interest_neg,
            "eval_metric": ['mrr'], 
        }

        predictions = evaluator.eval(input_dict)
        perf_list[j] = float(predictions['mrr'])
        hits_list[j] = float(predictions['hits@10'])
        

    return perf_list, hits_list


def match_body_relations(rule, edges, test_query_sub):
    """
    for rules of length 1
    Find quadruples that match the rule (starting from the test query subject)
    Find edges whose subject match the query subject and the relation matches
    the relation in the rule body. 
    Memory-efficient implementation.

    modified from Tlogic rule_application.py https://github.com/liu-yushan/TLogic/blob/main/mycode/rule_application.py
    shortened because we only have rules of length one 

    Parameters:
        rule (dict): rule from rules_dict
        edges (dict): edges for rule application
        test_query_sub (int): test query subject
    Returns:
        walk_edges (list of np.ndarrays): edges that could constitute rule walks
    """

    rels = rule["body_rels"]
    # Match query subject and first body relation
    try:
        rel_edges = edges[rels[0]]
        mask = rel_edges[:, 0] == test_query_sub
        new_edges = rel_edges[mask]
        walk_edges = [np.hstack((new_edges[:, 0:1], new_edges[:, 2:4]))]  # [sub, obj, ts]

    except KeyError:
        walk_edges = [[]]
    return walk_edges #subject object timestamp

def score_delta(cands_ts, test_query_ts, lmbda):
    """ deta function to score a given candidate based on its distance to current timestep and based on param lambda
    Parameters:
        cands_ts (int): timestep of candidate(s)
        test_query_ts (int): timestep of current test quadruple
        lmbda (float): param to specify how steep decay is
    Returns:
        score (float): score for a given candicate
    """
    score = pow(2, lmbda * (cands_ts - test_query_ts))
    return score

def get_window_edges(all_data, test_query_ts, window=-2, first_test_query_ts=0): 
    """
    modified from Tlogic rule_application.py https://github.com/liu-yushan/TLogic/blob/main/mycode/rule_application.py
    introduce window -2 

    Get the edges in the data (for rule application) that occur in the specified time window.
    If window is 0, all edges before the test query timestamp are included.
    If window is -2, all edges from train and validation set are used. as long as they are < first_test_query_ts
    If window is an integer n > 0, all edges within n timestamps before the test query
    timestamp are included.

    Parameters:
        all_data (np.ndarray): complete dataset (train/valid/test)
        test_query_ts (np.ndarray): test query timestamp
        window (int): time window used for rule application
        first_test_query_ts (int): smallest timestamp from test set (eval_paper_authors)

    Returns:
        window_edges (dict): edges in the window for rule application
    """

    if window > 0:
        mask = (all_data[:, 3] < test_query_ts) * (
            all_data[:, 3] >= test_query_ts - window 
        )
        window_edges = quads_per_rel(all_data[mask]) # quadruples per relation that fullfill the time constraints 
    elif window == 0:
        mask = all_data[:, 3] < test_query_ts #!!! 
        window_edges = quads_per_rel(all_data[mask]) 
    elif window == -2: #modified eval_paper_authors: added this option
        mask = all_data[:, 3] < first_test_query_ts # all edges at timestep smaller then the test queries. meaning all from train and valid set
        window_edges = quads_per_rel(all_data[mask])  
    elif window == -200: #modified eval_paper_authors: added this option
        abswindow = 200
        mask = (all_data[:, 3] < first_test_query_ts) * (
            all_data[:, 3] >= first_test_query_ts - abswindow  # all edges at timestep smaller than the test queries - 200
        )
        window_edges = quads_per_rel(all_data[mask])
    all_data_ts = all_data[mask]
    return window_edges, all_data_ts


def quads_per_rel(quads):
    """
    modified from Tlogic rule_application.py https://github.com/liu-yushan/TLogic/blob/main/mycode/rule_application.py
    Store all edges for each relation.

    Parameters:
        quads (np.ndarray): indices of quadruples

    Returns:
        edges (dict): edges for each relation
    """

    edges = dict()
    relations = list(set(quads[:, 1]))
    for rel in relations:
        edges[rel] = quads[quads[:, 1] == rel]
    return edges

def get_candidates_psi(rule_walks, test_query_ts, cands_dict,lmbda, sum_delta_t):
    """
    Get answer candidates from the walks that follow the rule.
    Add the confidence of the rule that leads to these candidates.
    originally from TLogic https://github.com/liu-yushan/TLogic/blob/main/mycode/apply.py but heavily modified

    Parameters:
        rule_walks (np.array): rule walks np array with [[sub, obj]]
        test_query_ts (int): test query timestamp
        cands_dict (dict): candidates along with the confidences of the rules that generated these candidates
        score_func (function): function for calculating the candidate score
        lmbda (float): parameter to describe decay of the scoring function
        sum_delta_t: to be used in denominator of scoring fct
    Returns:
        cands_dict (dict): keys: candidates, values: score for the candidates  """

    cands = set(rule_walks[:,0]) 

    for cand in cands:
        cands_walks = rule_walks[rule_walks[:,0] == cand] 
        score = score_psi(cands_walks, test_query_ts, lmbda, sum_delta_t).astype(np.float64)
        cands_dict[cand] = score

    return cands_dict

def update_delta_t(min_ts, max_ts, cur_ts, lmbda):
    """ compute denominator for scoring function psi_delta
    Patameters:
        min_ts (int): minimum available timestep
        max_ts (int): maximum available timestep
        cur_ts (int): current timestep
        lmbda (float): time decay parameter
    Returns:
        delta_all (float): sum(delta_t for all available timesteps between min_ts and max_ts)
    """
    timesteps = np.arange(min_ts, max_ts)
    now = np.ones(len(timesteps))*cur_ts
    delta_all = score_delta(timesteps, now, lmbda)
    delta_all = np.sum(delta_all)
    return delta_all

def score_psi(cands_walks, test_query_ts, lmbda, sum_delta_t):
    """
    Calculate candidate score depending on the time difference.

    Parameters:
        cands_walks (np.array): rule walks np array with [[sub, obj]]
        test_query_ts (int): test query timestamp
        lmbda (float): rate of exponential distribution

    Returns:
        score (float): candidate score
    """

    all_cands_ts = cands_walks[:,1] #cands_walks["timestamp_0"].reset_index()["timestamp_0"]
    ts_series = np.ones(len(all_cands_ts))*test_query_ts 
    scores =  score_delta(all_cands_ts, ts_series, lmbda) # Score depending on time difference
    if sum_delta_t == 0:
        print(scores, "sum_delta_t is zero")
        print(all_cands_ts)
        score = np.sum(scores)
        # print(score)
    else:            
        score = np.sum(scores)/sum_delta_t

    return score   

def update_distributions(ts_edges,num_rels):
    """ update the distributions with more recent infos, if there is a more recent timestep available, depending on window parameter
    take into account scaling factor
    """
    rel_obj_dist_cur_ts= calculate_obj_distribution(ts_edges, num_rels) #, lmbda, cur_ts)
    return  rel_obj_dist_cur_ts

def calculate_obj_distribution(edges, num_rels):
    """
    Calculate the overall object distribution and the object distribution for each relation in the data.

    Parameters:
        edges (dict): edges from the data on which the rules should be learned

    Returns:
        rel_obj_dist (dict): object distribution for each relation
    """
    rel_obj_dist_scaled = dict()
    for rel in range(num_rels):
        rel_obj_dist_scaled[rel] = {}
    
    for rel in edges:
        objects = edges[rel][:, 2]
        dist = Counter(objects)
        for obj in dist:
            dist[obj] /= len(objects)
        rel_obj_dist_scaled[rel] = {k: v for k, v in dist.items()}

    return rel_obj_dist_scaled

def update_delta_t(min_ts, max_ts, cur_ts, lmbda):
    """ compute denominator for scoring function psi_delta
    Patameters:
        min_ts (int): minimum available timestep
        max_ts (int): maximum available timestep
        cur_ts (int): current timestep
        lmbda (float): time decay parameter
    Returns:
        delta_all (float): sum(delta_t for all available timesteps between min_ts and max_ts)
    """
    timesteps = np.arange(min_ts, max_ts)
    now = np.ones(len(timesteps))*cur_ts
    delta_all = score_delta(timesteps, now, lmbda)
    delta_all = np.sum(delta_all)
    return delta_all
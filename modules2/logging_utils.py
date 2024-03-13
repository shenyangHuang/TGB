# TO BE REPLACED AS SOON AS WE HAVE UNIFIED TKG EVALUATION

import numpy as np
import torch


def query_name_from_quadruple(quad, num_rels, plus_one_flag=False):
    """ get the query namefrom the given quadruple. if they do reverse prediction with nr*rel+rel_id then we undo it here
    :param quad: numpy array, len 4: [sub, rel, ob, ts]; if rel>num_rels-1: this means inverse prediction
    :param num_rels: [int] number of relations
    :param plus_one_flag: [Bool] if the number of relations for inverse predictions is one higher than expected - the case for timetraveler:self.quadruples.append([ex[2], ex[1]+num_r+1, ex[0], ex[3]]
    :return: 
    query_name [str]: name of the query, with xxx showing the entity of interest. e.g.'30_13_xxx_334' for 
        object prediction or 'xxx_13_18_334' for subject prediction
    test_query_ids [np array]: sub, rel, ob, ts (original rel id)
    """
    rel = quad[1]
    ts = quad[3]
    if rel > (num_rels-1): #wrong direction
        
        ob_pred = False
        if plus_one_flag == False:
            rel = rel - (num_rels) 
        else:
            rel = rel - (num_rels) -1 
        sub = quad[2]
        ob = quad[0]
    else:
        ob_pred = True
        sub = quad[0]
        ob = quad[2]      
    
    if ob_pred == True:
        query_name = str(sub) + '_' + str(rel) + '_' + 'xxx'+ str(ob) +'_' + str(ts)
    else:
        query_name = 'xxx'+ str(sub)+ '_' + str(rel) + '_' + str(ob) + '_'  + str(ts)
    
    test_query_ids = np.array([sub, rel, ob, ts])
    return query_name, test_query_ids

def create_scores_tensor(predictions_dict, num_nodes, device=None):
    """ for given dict with key: node id, and value: score -> create a tensor with num_nodes entries, where the score 
    from dict is enetered at respective place, and all others are zeros.

    :returns: predictions  tensor with predicted scores, one per node; e.g. tensor([ 5.3042,  6....='cuda:0') torch.Size([23033])
    """
    predictions = torch.zeros(num_nodes, device=device)
    predictions.scatter_(0, torch.tensor(list(predictions_dict.keys())).long(), torch.tensor(list(predictions_dict.values())).float())
    return predictions
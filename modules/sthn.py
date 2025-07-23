import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import numpy as np
from torch import Tensor


from tqdm import tqdm
from sampler_core import ParallelSampler
import torch_sparse


import time
import copy
import random
from torch_sparse import SparseTensor
from torchmetrics.classification import MulticlassAUROC, MulticlassAveragePrecision
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision
from sklearn.preprocessing import MinMaxScaler
import os
import pickle


"""
Source: STHN: utils.py
URL: https://github.com/celi52/STHN/blob/main/utils.py
"""

# utility function
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def row_norm(adj_t):
    if isinstance(adj_t, torch_sparse.SparseTensor):
        # adj_t = torch_sparse.fill_diag(adj, 1)
        deg = torch_sparse.sum(adj_t, dim=1)
        deg_inv = 1. / deg
        deg_inv.masked_fill_(deg_inv == float('inf'), 0.)
        adj_t = torch_sparse.mul(adj_t, deg_inv.view(-1, 1))
        return adj_t


"""
Source: STHN: construct_subgraph.py
URL: https://github.com/celi52/STHN/blob/main/construct_subgraph.py

Notes: The NegLinkSampler is only used for STHN internal sampling and not for TGB
"""


##############################################################################
##############################################################################
##############################################################################


# get sampler
class NegLinkSampler:
    """
    From https://github.com/amazon-research/tgl/blob/main/sampler.py
    """
    def __init__(self, num_nodes):
        self.num_nodes = num_nodes

    def sample(self, n):
        return np.random.randint(self.num_nodes, size=n)
    
def get_parallel_sampler(g, num_neighbors=10):
    """
    Function wrapper of the C++ sampler (https://github.com/amazon-research/tgl/blob/main/sampler_core.cpp)
    Sample the 1-hop most recent neighbors of each node
    """

    configs = [
        g['indptr'],       # indptr --> fixed: data info
        g['indices'],      # indices --> fixed: data info
        g['eid'],          # eid --> fixed: data info
        g['ts'],           # ts --> fixed: data info
        32, # num_thread_per_worker --> change this based on machine's setup
        1,  # num_workers --> change this based on machine's setup
        1,  # num_layers --> change this based on machine's setup
        [num_neighbors],   # num_neighbors --> hyper-parameters. Reddit 10, WIKI 30
        True,  # recent --> fixed: never touch
        False, # prop_time --> never touch
        1,     # num_history --> fixed: never touch
        0      # window_duration --> fixed: never touch
    ]
    
    sampler = ParallelSampler(*configs)       
    neg_link_sampler = NegLinkSampler(g['indptr'].shape[0] - 1)
    return sampler, neg_link_sampler
    
##############################################################################
##############################################################################
##############################################################################
# sampling

def get_mini_batch(sampler, root_nodes, ts, num_hops): # neg_samples is not used
    """
    Call function fetch_subgraph()
    Return: Subgraph of each node. 
    """
    all_graphs = []
    
    for root_node, root_time in zip(root_nodes, ts):
        all_graphs.append(fetch_subgraph(sampler, root_node, root_time, num_hops))

    return all_graphs

def fetch_subgraph(sampler, root_node, root_time, num_hops):
    """
    Sample a subgraph for each node or node pair
    """
    all_row_col_times_nodes_eid = []

    # suppose sampling for both a single node and a node pair (two side of a link)
    if isinstance(root_node, list):
        nodes, ts = [i for i in root_node], [root_time for i in root_node]
    else:
        nodes, ts = [root_node], [root_time]
    
    # fetch all nodes+edges
    for _ in range(num_hops):
        sampler.sample(nodes, ts)
        ret = sampler.get_ret() # 1-hop recent neighbors
        row, col, eid = ret[0].row(), ret[0].col(), ret[0].eid()
        nodes, ts = ret[0].nodes(), ret[0].ts().astype(np.float32)
             
        row_col_times_nodes_eid = np.stack([ts[row], nodes[row], ts[col], nodes[col], eid]).T
        all_row_col_times_nodes_eid.append(row_col_times_nodes_eid)
    all_row_col_times_nodes_eid = np.concatenate(all_row_col_times_nodes_eid, axis=0)

    # remove duplicate edges and sort according to the root node time (descending)
    all_row_col_times_nodes_eid = np.unique(all_row_col_times_nodes_eid, axis=0)[::-1]
    all_row_col_times_nodes = all_row_col_times_nodes_eid[:, :-1]
    eid = all_row_col_times_nodes_eid[:, -1]

    # remove duplicate (node+time) and sorted by time decending order
    all_row_col_times_nodes = np.array_split(all_row_col_times_nodes, 2, axis=1)
    times_nodes = np.concatenate(all_row_col_times_nodes, axis=0)
    times_nodes = np.unique(times_nodes, axis=0)[::-1]
    
    # each (node, time) pair identifies a node
    node_2_ind = dict()
    for ind, (time, node) in enumerate(times_nodes):
        node_2_ind[(time, node)] = ind

    # translate the nodes into new index
    row = np.zeros(len(eid), dtype=np.int32)
    col = np.zeros(len(eid), dtype=np.int32)
    for i, ((t1, n1), (t2, n2)) in enumerate(zip(*all_row_col_times_nodes)):
        row[i] = node_2_ind[(t1, n1)]
        col[i] = node_2_ind[(t2, n2)]
        
    # fetch get time + node information
    eid = eid.astype(np.int32)
    ts = times_nodes[:,0].astype(np.float32)
    nodes = times_nodes[:,1].astype(np.int32)
    dts = root_time - ts # make sure the root node time is 0
    
    return {
        # edge info: sorted with descending row (src) node temporal order
        'row': row, # src
        'col': col, # dst
        'eid': eid, 
        # node info
        'nodes': nodes , # sorted by the ascending order of node's dts (root_node's dts = 0)
        'dts': dts,
        # graph info
        'num_nodes': len(nodes),
        'num_edges': len(eid),
        # root info
        'root_node': root_node,
        'root_time': root_time,
    }


def construct_mini_batch_giant_graph(all_graphs, max_num_edges):
    """
    Take the subgraph computed by fetch_subgraph() and combine it into a giant graph
    Return: the new indices of the graph
    """
    
    all_rows, all_cols, all_eids, all_nodes, all_dts = [], [], [], [], []
    
    cumsum_edges = 0
    all_edge_indptr = [0]
    
    cumsum_nodes = 0
    all_node_indptr = [0]
    
    all_root_nodes = []
    all_root_times = []
    for all_graph in all_graphs:
        # record inds
        num_nodes = all_graph['num_nodes']
        num_edges = min(all_graph['num_edges'], max_num_edges)

        # add graph information
        all_rows.append(all_graph['row'][:num_edges] + cumsum_nodes)
        all_cols.append(all_graph['col'][:num_edges] + cumsum_nodes)
        all_eids.append(all_graph['eid'][:num_edges])
        
        all_nodes.append(all_graph['nodes'])
        all_dts.append(all_graph['dts'])

        # update cumsum
        cumsum_nodes += num_nodes
        all_node_indptr.append(cumsum_nodes)
        
        cumsum_edges += num_edges
        all_edge_indptr.append(cumsum_edges)
        
        # add root nodes
        all_root_nodes.append(all_graph['root_node'])
        all_root_times.append(all_graph['root_time'])
    # for each edges
    all_rows = np.concatenate(all_rows).astype(np.int32)
    all_cols = np.concatenate(all_cols).astype(np.int32)
    all_eids = np.concatenate(all_eids).astype(np.int32)
    all_edge_indptr = np.array(all_edge_indptr).astype(np.int32)
    
    # for each nodes
    all_nodes = np.concatenate(all_nodes).astype(np.int32)
    all_dts = np.concatenate(all_dts).astype(np.float32)
    all_node_indptr = np.array(all_node_indptr).astype(np.int32)
        
    return {
        # for edges
        'row': all_rows, 
        'col': all_cols, 
        'eid': all_eids, 
        'edts': all_dts[all_cols] - all_dts[all_rows],
        # number of subgraphs + 1
        'all_node_indptr': all_node_indptr,
        'all_edge_indptr': all_edge_indptr,
        # for nodes
        'nodes': all_nodes, 
        'dts': all_dts, 
        # general information
        'all_num_nodes': cumsum_nodes,
        'all_num_edges': cumsum_edges,
        # root nodes
        'root_nodes': np.array(all_root_nodes, dtype=np.int32), 
        'root_times': np.array(all_root_times, dtype=np.float32), 
    }

##############################################################################
##############################################################################
##############################################################################

def print_subgraph_data(subgraph_data):
    """
    Used to double check see if the sampled graph is as expected
    """
    for key, vals in subgraph_data.items():
        if isinstance(vals, np.ndarray):
            print(key, vals.shape)
        else:
            print(key, vals)


"""
Source: STHN data_process_utils.py
URL: https://github.com/celi52/STHN/blob/main/data_process_utils.py

Note:
Currently only using pre_compute_subgraphs because use_cached_subgraph is True
get_subgraph_sampler needs to be modified if use_cached_subgraph is False

The function get_all_inds is new to handle TGB evaluation
"""


class SubgraphSampler:
    def __init__(self, all_root_nodes, all_ts, sampler, args):
        self.all_root_nodes = all_root_nodes
        self.all_ts = all_ts
        self.sampler = sampler
        self.sampled_num_hops = args.sampled_num_hops

    def mini_batch(self, ind, mini_batch_inds):
        root_nodes = self.all_root_nodes[ind][mini_batch_inds]
        ts = self.all_ts[ind][mini_batch_inds]
        return get_mini_batch(self.sampler, root_nodes, ts, self.sampled_num_hops)

def get_subgraph_sampler(args, g, df, mode):
    ###################################################
    # get cached file_name
    if mode == 'train':
        extra_neg_samples = args.extra_neg_samples
    else:
        extra_neg_samples = 1

    ###################################################
    # for each node, sample its neighbors with the most recent neighbors (sorted) 
    print('Sample subgraphs ... for %s mode'%mode)
    sampler, neg_link_sampler = get_parallel_sampler(g, args.num_neighbors)

    ###################################################
    # setup modes
    if mode == 'train':
        cur_df = df[args.train_mask]

    elif mode == 'valid':
        cur_df = df[args.val_mask]

    elif mode == 'test':
        cur_df = df[args.test_mask]

    loader = cur_df.groupby(cur_df.index // args.batch_size)
    print(cur_df.index, cur_df.index // args.batch_size)
    pbar = tqdm(total=len(loader))
    pbar.set_description('Pre-sampling: %s mode with negative sampleds %s ...'%(mode, extra_neg_samples))

    all_root_nodes = []
    all_ts = []
    for _, rows in loader:

        root_nodes = np.concatenate(
            [rows.src.values, 
            rows.dst.values, 
            neg_link_sampler.sample(len(rows) * extra_neg_samples)]
        ).astype(np.int32)
        all_root_nodes.append(root_nodes)

        # time-stamp for node = edge time-stamp
        ts = np.tile(rows.time.values, extra_neg_samples + 2).astype(np.float32)
        all_ts.append(ts)

        pbar.update(1)
    pbar.close()
    return SubgraphSampler(all_root_nodes, all_ts, sampler, args)

######################################################################################################
######################################################################################################
######################################################################################################
# for small dataset, we can cache each graph
def pre_compute_subgraphs(args, g, df, mode, negative_sampler=None, split_mode='test', cache=False):
    ###################################################
    # get cached file_name
    if mode == 'train':
        extra_neg_samples = args.extra_neg_samples
    else:
        extra_neg_samples = 1

    fn = os.path.join(os.getcwd(), 'DATA', args.data, 
                        '%s_neg_sample_neg%d_bs%d_hops%d_neighbors%d.pickle'%(mode, 
                                                                            extra_neg_samples, 
                                                                            args.batch_size, 
                                                                            args.sampled_num_hops, 
                                                                          args.num_neighbors))
    ###################################################

    # # try:
    if os.path.exists(fn):
        subgraph_elabel = pickle.load(open(fn, 'rb'))
        # print('load ', fn)

    else:
        ##################################################
        # for each node, sample its neighbors with the most recent neighbors (sorted) 
        print('Sample subgraphs ... for %s mode'%mode)
        sampler, neg_link_sampler = get_parallel_sampler(g, args.num_neighbors)

        ###################################################
        # setup modes
        if mode == 'train':
            cur_df = df[args.train_mask]

        elif mode == 'valid':
            cur_df = df[args.val_mask]

        elif mode == 'test':
            cur_df = df[args.test_mask]

        loader = cur_df.groupby(cur_df.index // args.batch_size)
        pbar = tqdm(total=len(loader))
        pbar.set_description('Pre-sampling: %s mode'%(mode,))

        ###################################################
        all_subgraphs = []
        all_elabel = []
        sampler.reset()
        for _, rows in loader:
            
            if negative_sampler is not None:
                neg_batch_list = negative_sampler.query_batch(
                    rows.src.values,
                    rows.dst.values,
                    rows.time.values,
                    rows.label.values,
                    split_mode=split_mode
                )
                neg_batch_list = np.concatenate(neg_batch_list)
                extra_neg_samples = neg_batch_list.shape[0] // len(rows)
            else:
                neg_batch_list = neg_link_sampler.sample(len(rows) * extra_neg_samples)

            root_nodes = np.concatenate(
                [rows.src.values, 
                    rows.dst.values, 
                    neg_batch_list]
            ).astype(np.int32)

            # time-stamp for node = edge time-stamp
            ts = np.tile(rows.time.values, extra_neg_samples + 2).astype(np.float32)
            all_elabel.append(rows.label.values)
            all_subgraphs.append(get_mini_batch(sampler, root_nodes, ts, args.sampled_num_hops))
            
            pbar.update(1)
        pbar.close()
        subgraph_elabel = (all_subgraphs, all_elabel)

        if cache:
            try:
                pickle.dump(subgraph_elabel, open(fn, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
            except:
                print('For some shit reason pickle cannot save ... but anyway ...')
        
        ##################################################
        
    return subgraph_elabel


def get_random_inds(num_subgraph, cached_neg_samples, neg_samples):
    ###################################################
    batch_size = num_subgraph // (2+cached_neg_samples)
    pos_src_inds = np.arange(batch_size)
    pos_dst_inds = np.arange(batch_size) + batch_size
    neg_dst_inds = np.random.randint(low=2, high=2+cached_neg_samples, size=batch_size*neg_samples)
    neg_dst_inds = batch_size * neg_dst_inds + np.arange(batch_size)
    mini_batch_inds = np.concatenate([pos_src_inds, pos_dst_inds, neg_dst_inds]).astype(np.int32)
    ###################################################

    return mini_batch_inds


def get_all_inds(num_subgraph, neg_samples):
    ###################################################
    batch_size = num_subgraph // (2+neg_samples)
    pos_src_inds = np.arange(batch_size)
    pos_dst_inds = np.arange(batch_size) + batch_size
    neg_dst_inds = batch_size * 2 + np.arange(batch_size * neg_samples)
    mini_batch_inds = np.concatenate([pos_src_inds, pos_dst_inds, neg_dst_inds]).astype(np.int32)
    ###################################################

    return mini_batch_inds


def check_data_leakage(args, g, df):
    """
    This is a function to double if the sampled graph has eid greater than the positive node pairs eid (if no then no data leakage)
    """
    for mode in ['train', 'valid', 'test']:

        if mode == 'train':
            cur_df = df[:args.train_edge_end]
        elif mode == 'valid':
            cur_df = df[args.train_edge_end:args.val_edge_end]
        elif mode == 'test':
            cur_df = df[args.val_edge_end:]

        loader = cur_df.groupby(cur_df.index // args.batch_size)
        subgraphs = pre_compute_subgraphs(args, g, df, mode)

        for i, (_, rows) in enumerate(loader):
            root_nodes = np.concatenate([rows.src.values, rows.dst.values]).astype(np.int32)
            eids = np.tile(rows.index.values, 2)
            cur_subgraphs = subgraphs[i][:args.batch_size*2]

            for eid, cur_subgraph in zip(eids, cur_subgraphs):
                all_eids_in_subgraph = cur_subgraph['eid']
                if len(all_eids_in_subgraph) == 0:
                    continue
                # all edges in the sampled graph has eid smaller than the target edge's eid, i.e,. sampled links never seen before
                assert sum(all_eids_in_subgraph < eid) == len(all_eids_in_subgraph)
                
    print('Does not detect information leakage ...')


"""
Source: STHN link_pred_train_utils.py
URL: https://github.com/celi52/STHN/blob/main/link_pred_train_utils.py

Notes: I created a separate function for get_inputs_for_ind so that we can use it for TGB evaluation as well
"""

def get_inputs_for_ind(subgraphs, mode, cached_neg_samples, neg_samples, node_feats, edge_feats, cur_df, cur_inds, ind, args):
    subgraphs, elabel = subgraphs
    scaler = MinMaxScaler()
    if args.use_cached_subgraph == False and mode == 'train':
        subgraph_data_list = subgraphs.all_root_nodes[ind]
        mini_batch_inds = get_random_inds(len(subgraph_data_list), cached_neg_samples, neg_samples)
        subgraph_data = subgraphs.mini_batch(ind, mini_batch_inds)
    elif mode in ['test', 'tgb-val']:
        assert cached_neg_samples == neg_samples
        subgraph_data_list = subgraphs[ind]
        mini_batch_inds = get_all_inds(len(subgraph_data_list), cached_neg_samples)
        subgraph_data = [subgraph_data_list[i] for i in mini_batch_inds]      
    else: # sthn valid
        subgraph_data_list = subgraphs[ind]
        mini_batch_inds = get_random_inds(len(subgraph_data_list), cached_neg_samples, neg_samples)
        subgraph_data = [subgraph_data_list[i] for i in mini_batch_inds]
    subgraph_data = construct_mini_batch_giant_graph(subgraph_data, args.max_edges)

    # raw edge feats 
    subgraph_edge_feats = edge_feats[subgraph_data['eid']]
    subgraph_edts = torch.from_numpy(subgraph_data['edts']).float()
    if args.use_graph_structure and node_feats:
        num_of_df_links = len(subgraph_data_list) //  (cached_neg_samples+2)   
        # subgraph_node_feats = compute_sign_feats(node_feats, df, cur_inds, num_of_df_links, subgraph_data['root_nodes'], args)
        # Erfan: change this part to use masked version
        subgraph_node_feats = compute_sign_feats(node_feats, cur_df, cur_inds, num_of_df_links, subgraph_data['root_nodes'], args)
        cur_inds += num_of_df_links
    else:
        subgraph_node_feats = None
    # scale
    scaler.fit(subgraph_edts.reshape(-1,1))
    subgraph_edts = scaler.transform(subgraph_edts.reshape(-1,1)).ravel().astype(np.float32) * 1000
    subgraph_edts = torch.from_numpy(subgraph_edts)
    
    # get mini-batch inds
    all_inds, has_temporal_neighbors = [], []

    # ignore an edge pair if (src_node, dst_node) does not have temporal neighbors
    all_edge_indptr = subgraph_data['all_edge_indptr']
    
    for i in range(len(all_edge_indptr)-1):
        num_edges = all_edge_indptr[i+1] - all_edge_indptr[i]
        all_inds.extend([(args.max_edges * i + j) for j in range(num_edges)])
        has_temporal_neighbors.append(num_edges>0)
        
    if not args.predict_class:
        inputs = [
            subgraph_edge_feats.to(args.device), 
            subgraph_edts.to(args.device), 
            len(has_temporal_neighbors), 
            torch.tensor(all_inds).long() 
        ]
    else:
        subgraph_edge_type = elabel[ind]
        inputs = [
            subgraph_edge_feats.to(args.device), 
            subgraph_edts.to(args.device), 
            len(has_temporal_neighbors), 
            torch.tensor(all_inds).long(),  
            torch.from_numpy(subgraph_edge_type).to(args.device)
        ]
    return inputs, subgraph_node_feats, cur_inds

def run(model, optimizer, args, subgraphs, df, node_feats, edge_feats, MLAUROC, MLAUPRC, mode):
    time_epoch = 0
    ###################################################
    # setup modes
    cur_inds = 0
    if mode == 'train':
        model.train()
        cur_df = df[args.train_mask]
        neg_samples = args.neg_samples
        cached_neg_samples = args.extra_neg_samples

    elif mode == 'valid':
        model.eval()
        cur_df = df[args.val_mask]
        neg_samples = 1
        cached_neg_samples = 1

    elif mode == 'test':
        ## Erfan: remove this part use TGB evaluation
        raise('Use TGB evaluation')
        # model.eval()
        # cur_df = df[args.test_mask]
        # neg_samples = 1
        # cached_neg_samples = 1
        # cur_inds = args.val_edge_end

    train_loader = cur_df.groupby(cur_df.index // args.batch_size)
    pbar = tqdm(total=len(train_loader))
    pbar.set_description('%s mode with negative samples %d ...'%(mode, neg_samples))        
        
    ###################################################
    # compute + training + fetch all scores
    loss_lst = []
    MLAUROC.reset()
    MLAUPRC.reset()
    
    for ind in range(len(train_loader)):
        ###################################################
        inputs, subgraph_node_feats, cur_inds = get_inputs_for_ind(subgraphs, mode, cached_neg_samples, neg_samples, node_feats, edge_feats, cur_df, cur_inds, ind, args)
        
        start_time = time.time()
        loss, pred, edge_label = model(inputs, neg_samples, subgraph_node_feats)
        if mode == 'train' and optimizer != None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        time_epoch += (time.time() - start_time)
        
        batch_auroc = MLAUROC.update(pred, edge_label)
        batch_auprc = MLAUPRC.update(pred, edge_label)
        loss_lst.append(float(loss.detach()))

        pbar.update(1)
    pbar.close()    
    total_auroc = MLAUROC.compute()
    total_auprc = MLAUPRC.compute()
    print('%s mode with time %.4f, AUROC %.4f, AUPRC %.4f, loss %.4f'%(mode, time_epoch, total_auroc, total_auprc, loss.item()))
    return_loss = np.mean(loss_lst)
    return total_auroc, total_auprc, return_loss, time_epoch


def link_pred_train(model, args, g, df, node_feats, edge_feats):
    
    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ###################################################
    # get cached data
    if args.use_cached_subgraph:
        train_subgraphs = pre_compute_subgraphs(args, g, df, mode='train')
    else:
        train_subgraphs = get_subgraph_sampler(args, g, df, mode='train')
    
    valid_subgraphs = pre_compute_subgraphs(args, g, df, mode='valid')
    # test_subgraphs  = pre_compute_subgraphs(args, g, df, mode='test' )
          
    ###################################################
    all_results = {
        'train_ap': [],
        'valid_ap': [],
        # 'test_ap' : [],
        'train_auc': [],
        'valid_auc': [],
        # 'test_auc' : [],
        'train_loss': [],
        'valid_loss': [],
        # 'test_loss': [],
    }

    low_loss = 100000
    user_train_total_time = 0
    user_epoch_num = 0
    if args.predict_class:
        num_classes = args.num_edgeType+1
        train_AUROC = MulticlassAUROC(num_classes, average="macro", thresholds=None)
        valid_AUROC = MulticlassAUROC(num_classes, average="macro", thresholds=None)
        train_AUPRC = MulticlassAveragePrecision(num_classes, average="macro", thresholds=None)
        valid_AUPRC = MulticlassAveragePrecision(num_classes, average="macro", thresholds=None)
    else:
        train_AUROC = BinaryAUROC(thresholds=None)
        valid_AUROC = BinaryAUROC(thresholds=None)
        train_AUPRC = BinaryAveragePrecision(thresholds=None)
        valid_AUPRC = BinaryAveragePrecision(thresholds=None)
        
    for epoch in range(args.epochs):
        print('>>> Epoch ', epoch+1)
        train_auc, train_ap, train_loss, time_train = run(model, optimizer, args, train_subgraphs, df, 
                                              node_feats, edge_feats, train_AUROC, train_AUPRC, mode='train')
        with torch.no_grad():
            # second variable (optimizer) is only required for training
            valid_auc, valid_ap, valid_loss, time_valid = run(copy.deepcopy(model), None, args, valid_subgraphs, df, 
                                                  node_feats, edge_feats, valid_AUROC, valid_AUPRC, mode='valid')
        #     # second variable (optimizer) is only required for training
        #     test_auc,  test_ap,  test_loss, time_test = run(copy.deepcopy(model), None, args, test_subgraphs,  df, 
        #                                           node_feats, edge_feats, test_AUROC, test_AUPRC, mode='test')  

        if valid_loss < low_loss:
            best_auc_model = copy.deepcopy(model).cpu() 
            best_auc = valid_auc
            low_loss = valid_loss
            best_epoch = epoch

        user_train_total_time += time_train + time_valid
        user_epoch_num += 1
        if epoch > best_epoch + 20:
            break
        
        all_results['train_ap'].append(train_ap)
        all_results['valid_ap'].append(valid_ap)
        # all_results['test_ap'].append(test_ap)
        
        all_results['valid_auc'].append(valid_auc)
        all_results['train_auc'].append(train_auc)
        # all_results['test_auc'].append(test_auc)
        
        all_results['train_loss'].append(train_loss)
        all_results['valid_loss'].append(valid_loss)
        # all_results['test_loss'].append(test_loss)        
        
    print('best epoch %d, auc score %.4f'%(best_epoch, best_auc))     
    return best_auc_model


def compute_sign_feats(node_feats, df, start_i, num_links, root_nodes, args):
    num_duplicate = len(root_nodes) // num_links 
    num_nodes = args.num_nodes

    root_inds = torch.arange(len(root_nodes)).view(num_duplicate, -1)
    root_inds = [arr.flatten() for arr in root_inds.chunk(1, dim=1)]

    output_feats = torch.zeros((len(root_nodes), node_feats.size(1))).to(args.device)
    i = start_i

    for _root_ind in root_inds:

        if i == 0 or args.structure_hops == 0:
            sign_feats = node_feats.clone()
        else:
            prev_i = max(0, i - args.structure_time_gap)
            cur_df = df[prev_i: i] # get adj's row, col indices (as undirected)
            src = torch.from_numpy(cur_df.src.values)
            dst = torch.from_numpy(cur_df.dst.values)
            edge_index = torch.stack([
                torch.cat([src, dst]), 
                torch.cat([dst, src])
            ])
            edge_index, edge_cnt = torch.unique(edge_index, dim=1, return_counts=True) 
            mask = edge_index[0]!=edge_index[1] # ignore self-loops
            adj = SparseTensor(
                value = torch.ones_like(edge_cnt[mask]).float(),
                row = edge_index[0][mask].long(),
                col = edge_index[1][mask].long(),
                sparse_sizes=(num_nodes, num_nodes)
            )
            adj_norm = row_norm(adj).to(args.device)
            sign_feats = [node_feats]
            for _ in range(args.structure_hops):
                sign_feats.append(adj_norm@sign_feats[-1])
            sign_feats = torch.sum(torch.stack(sign_feats), dim=0)

        output_feats[_root_ind] = sign_feats[root_nodes[_root_ind]]

        i += len(_root_ind) // num_duplicate

    return output_feats



################################################################################################
################################################################################################
################################################################################################

"""
Source: STHN torch_encodings
URL: https://github.com/celi52/STHN/blob/main/torch_encodings.py
"""

def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)


class PositionalEncoding1D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding1D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 2) * 2)
        self.channels = channels
        inv_freq = 1.0 / (1000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.cached_penc = None

    def forward(self, tensor):
        """
        :param tensor: A 3d tensor of size (batch_size, x, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, ch)
        """
        if len(tensor.shape) != 3:
            raise RuntimeError("The input tensor has to be 3d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        batch_size, x, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        emb_x = get_emb(sin_inp_x)
        emb = torch.zeros((x, self.channels), device=tensor.device).type(tensor.type())
        emb[:, : self.channels] = emb_x

        self.cached_penc = emb[None, :, :orig_ch].repeat(batch_size, 1, 1)
        return self.cached_penc


class PositionalEncodingPermute1D(nn.Module):
    def __init__(self, channels):
        """
        Accepts (batchsize, ch, x) instead of (batchsize, x, ch)
        """
        super(PositionalEncodingPermute1D, self).__init__()
        self.penc = PositionalEncoding1D(channels)

    def forward(self, tensor):
        tensor = tensor.permute(0, 2, 1)
        enc = self.penc(tensor)
        return enc.permute(0, 2, 1)

    @property
    def org_channels(self):
        return self.penc.org_channels


class PositionalEncoding2D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding2D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 4) * 2)
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.cached_penc = None

    def forward(self, tensor):
        """
        :param tensor: A 4d tensor of size (batch_size, x, y, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, ch)
        """
        if len(tensor.shape) != 4:
            raise RuntimeError("The input tensor has to be 4d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        batch_size, x, y, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(y, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        emb_x = get_emb(sin_inp_x).unsqueeze(1)
        emb_y = get_emb(sin_inp_y)
        emb = torch.zeros((x, y, self.channels * 2), device=tensor.device).type(
            tensor.type()
        )
        emb[:, :, : self.channels] = emb_x
        emb[:, :, self.channels : 2 * self.channels] = emb_y

        self.cached_penc = emb[None, :, :, :orig_ch].repeat(tensor.shape[0], 1, 1, 1)
        return self.cached_penc


class PositionalEncodingPermute2D(nn.Module):
    def __init__(self, channels):
        """
        Accepts (batchsize, ch, x, y) instead of (batchsize, x, y, ch)
        """
        super(PositionalEncodingPermute2D, self).__init__()
        self.penc = PositionalEncoding2D(channels)

    def forward(self, tensor):
        tensor = tensor.permute(0, 2, 3, 1)
        enc = self.penc(tensor)
        return enc.permute(0, 3, 1, 2)

    @property
    def org_channels(self):
        return self.penc.org_channels


class PositionalEncoding3D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding3D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 6) * 2)
        if channels % 2:
            channels += 1
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.cached_penc = None

    def forward(self, tensor):
        """
        :param tensor: A 5d tensor of size (batch_size, x, y, z, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, z, ch)
        """
        if len(tensor.shape) != 5:
            raise RuntimeError("The input tensor has to be 5d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        batch_size, x, y, z, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(y, device=tensor.device).type(self.inv_freq.type())
        pos_z = torch.arange(z, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        sin_inp_z = torch.einsum("i,j->ij", pos_z, self.inv_freq)
        emb_x = get_emb(sin_inp_x).unsqueeze(1).unsqueeze(1)
        emb_y = get_emb(sin_inp_y).unsqueeze(1)
        emb_z = get_emb(sin_inp_z)
        emb = torch.zeros((x, y, z, self.channels * 3), device=tensor.device).type(
            tensor.type()
        )
        emb[:, :, :, : self.channels] = emb_x
        emb[:, :, :, self.channels : 2 * self.channels] = emb_y
        emb[:, :, :, 2 * self.channels :] = emb_z

        self.cached_penc = emb[None, :, :, :, :orig_ch].repeat(batch_size, 1, 1, 1, 1)
        return self.cached_penc


class PositionalEncodingPermute3D(nn.Module):
    def __init__(self, channels):
        """
        Accepts (batchsize, ch, x, y, z) instead of (batchsize, x, y, z, ch)
        """
        super(PositionalEncodingPermute3D, self).__init__()
        self.penc = PositionalEncoding3D(channels)

    def forward(self, tensor):
        tensor = tensor.permute(0, 2, 3, 4, 1)
        enc = self.penc(tensor)
        return enc.permute(0, 4, 1, 2, 3)

    @property
    def org_channels(self):
        return self.penc.org_channels


class Summer(nn.Module):
    def __init__(self, penc):
        """
        :param model: The type of positional encoding to run the summer on.
        """
        super(Summer, self).__init__()
        self.penc = penc

    def forward(self, tensor):
        """
        :param tensor: A 3, 4 or 5d tensor that matches the model output size
        :return: Positional Encoding Matrix summed to the original tensor
        """
        penc = self.penc(tensor)
        assert (
            tensor.size() == penc.size()
        ), "The original tensor size {} and the positional encoding tensor size {} must match!".format(
            tensor.size(), penc.size()
        )
        return tensor + penc


"""
Source: STHN model.py
URL: https://github.com/celi52/STHN/blob/main/model.py
"""


"""
Module: Time-encoder
"""

class TimeEncode(nn.Module):
    """
    out = linear(time_scatter): 1-->time_dims
    out = cos(out)
    """
    def __init__(self, dim):
        super(TimeEncode, self).__init__()
        self.dim = dim
        self.w = nn.Linear(1, dim)
        self.reset_parameters()
    
    def reset_parameters(self, ):
        self.w.weight = nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, self.dim, dtype=np.float32))).reshape(self.dim, -1))
        self.w.bias = nn.Parameter(torch.zeros(self.dim))

        self.w.weight.requires_grad = False
        self.w.bias.requires_grad = False
    
    @torch.no_grad()
    def forward(self, t):
        output = torch.cos(self.w(t.reshape((-1, 1))))
        return output



################################################################################################
################################################################################################
################################################################################################
"""
Module: STHN
"""

class FeedForward(nn.Module):
    """
    2-layer MLP with GeLU (fancy version of ReLU) as activation
    """
    def __init__(self, dims, expansion_factor, dropout=0, use_single_layer=False):
        super().__init__()

        self.dims = dims
        self.use_single_layer = use_single_layer
        
        self.expansion_factor = expansion_factor
        self.dropout = dropout

        if use_single_layer:
            self.linear_0 = nn.Linear(dims, dims)
        else:
            self.linear_0 = nn.Linear(dims, int(expansion_factor * dims))
            self.linear_1 = nn.Linear(int(expansion_factor * dims), dims)

        self.reset_parameters()

    def reset_parameters(self):
        self.linear_0.reset_parameters()
        if self.use_single_layer==False:
            self.linear_1.reset_parameters()

    def forward(self, x):
        x = self.linear_0(x)
        x = F.gelu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        if self.use_single_layer==False:
            x = self.linear_1(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class TransformerBlock(nn.Module):
    """
    out = X.T + MLP_Layernorm(X.T)     # apply token mixing
    out = out.T + MLP_Layernorm(out.T) # apply channel mixing
    """
    def __init__(self, dims, 
                 channel_expansion_factor=4, 
                 dropout=0.2, 
                 module_spec=None, use_single_layer=False):
        super().__init__()
        
        if module_spec == None:
            self.module_spec = ['token', 'channel']
        else:
            self.module_spec = module_spec.split('+')

        self.dims = dims
        if 'token' in self.module_spec:
            self.transformer_encoder = _MultiheadAttention(d_model=dims, 
                                                           n_heads=2,
                                                           d_k=None,
                                                           d_v=None,
                                                           attn_dropout=dropout)
        if 'channel' in self.module_spec:
            self.channel_layernorm = nn.LayerNorm(dims)
            self.channel_forward = FeedForward(dims, channel_expansion_factor, dropout, use_single_layer)
        
    def reset_parameters(self):
        if 'token' in self.module_spec:
            self.transformer_encoder.reset_parameters()
        if 'channel' in self.module_spec:
            self.channel_layernorm.reset_parameters()
            self.channel_forward.reset_parameters()
        
    def token_mixer(self, x):
        x = self.transformer_encoder(x, x, x)
        return x
    
    def channel_mixer(self, x):
        x = self.channel_layernorm(x)
        x = self.channel_forward(x)
        return x

    def forward(self, x):
        if 'token' in self.module_spec:
            x = x + self.token_mixer(x)
        if 'channel' in self.module_spec:
            x = x + self.channel_mixer(x)
        return x


class _MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, res_attention=False, attn_dropout=0., proj_dropout=0., qkv_bias=True, lsa=False):
        """Multi Head Attention Layer
        Input shape:
            Q:       [batch_size (bs) x max_q_len x d_model]
            K, V:    [batch_size (bs) x q_len x d_model]
            mask:    [q_len x q_len]
        """
        super().__init__()
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)

        # Scaled Dot-Product Attention (multiple heads)
        self.res_attention = res_attention
        self.sdp_attn = _ScaledDotProductAttention(d_model, n_heads, attn_dropout=attn_dropout, res_attention=self.res_attention, lsa=lsa)

        # Poject output
        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, d_model), nn.Dropout(attn_dropout))

    def reset_parameters(self):
        self.to_out[0].reset_parameters()
        self.W_Q.reset_parameters()
        self.W_K.reset_parameters()
        self.W_V.reset_parameters()

    def forward(self, Q:Tensor, K:Optional[Tensor]=None, V:Optional[Tensor]=None, prev:Optional[Tensor]=None,
                key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):

        bs = Q.size(0)
        if K is None: K = Q
        if V is None: V = Q

        # Linear (+ split in multiple heads)
        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1,2)       # q_s    : [bs x n_heads x max_q_len x d_k]
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0,2,3,1)     # k_s    : [bs x n_heads x d_k x q_len] - transpose(1,2) + transpose(2,3)
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1,2)       # v_s    : [bs x n_heads x q_len x d_v]

        # Apply Scaled Dot-Product Attention (multiple heads)
        output, attn_weights = self.sdp_attn(q_s, k_s, v_s, prev=prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)

        # back to the original inputs dimensions
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v) # output: [bs x q_len x n_heads * d_v]
        output = self.to_out(output)

        return output


class _ScaledDotProductAttention(nn.Module):
    r"""Scaled Dot-Product Attention module (Attention is all you need by Vaswani et al., 2017) with optional residual attention from previous layer
    (Realformer: Transformer likes residual attention by He et al, 2020) and locality self sttention (Vision Transformer for Small-Size Datasets
    by Lee et al, 2021)"""

    def __init__(self, d_model, n_heads, attn_dropout=0., res_attention=False, lsa=False):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=lsa)
        self.lsa = lsa

    def forward(self, q:Tensor, k:Tensor, v:Tensor, prev:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
        '''
        Input shape:
            q               : [bs x n_heads x max_q_len x d_k]
            k               : [bs x n_heads x d_k x seq_len]
            v               : [bs x n_heads x seq_len x d_v]
            prev            : [bs x n_heads x q_len x seq_len]
            key_padding_mask: [bs x seq_len]
            attn_mask       : [1 x seq_len x seq_len]
        Output shape:
            output:  [bs x n_heads x q_len x d_v]
            attn   : [bs x n_heads x q_len x seq_len]
            scores : [bs x n_heads x q_len x seq_len]
        '''

        # Scaled MatMul (q, k) - similarity scores for all pairs of positions in an input sequence
        attn_scores = torch.matmul(q, k) * self.scale      # attn_scores : [bs x n_heads x max_q_len x q_len]

        # Add pre-softmax attention scores from the previous layer (optional)
        if prev is not None: attn_scores = attn_scores + prev

        # Attention mask (optional)
        if attn_mask is not None:                                     # attn_mask with shape [q_len x seq_len] - only used when q_len == seq_len
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask

        # Key padding mask (optional)
        if key_padding_mask is not None:                              # mask with shape [bs x q_len] (only when max_w_len == q_len)
            attn_scores.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf)

        # normalize the attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)                 # attn_weights   : [bs x n_heads x max_q_len x q_len]
        attn_weights = self.attn_dropout(attn_weights)

        # compute the new values given the attention weights
        output = torch.matmul(attn_weights, v)                        # output: [bs x n_heads x max_q_len x d_v]

        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights


    
class FeatEncode(nn.Module):
    """
    Return [raw_edge_feat | TimeEncode(edge_time_stamp)]
    """
    def __init__(self, time_dims, feat_dims, out_dims):
        super().__init__()
        
        self.time_encoder = TimeEncode(time_dims)
        self.feat_encoder = nn.Linear(time_dims + feat_dims, out_dims) 
        self.reset_parameters()

    def reset_parameters(self):
        self.time_encoder.reset_parameters()
        self.feat_encoder.reset_parameters()
        
    def forward(self, edge_feats, edge_ts):
        edge_time_feats = self.time_encoder(edge_ts)
        x = torch.cat([edge_feats, edge_time_feats], dim=1)
        return self.feat_encoder(x)

class Patch_Encoding(nn.Module):
    """
    Input : [ batch_size, graph_size, edge_dims+time_dims]
    Output: [ batch_size, graph_size, output_dims]
    """
    def __init__(self, per_graph_size, time_channels,
                 input_channels, hidden_channels, out_channels,
                 num_layers, dropout,
                 channel_expansion_factor,
                 window_size,
                 module_spec=None, 
                 use_single_layer=False
                ):
        super().__init__()
        self.per_graph_size = per_graph_size
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        
        # input & output classifer
        self.feat_encoder = FeatEncode(time_channels, input_channels, hidden_channels)
        self.layernorm = nn.LayerNorm(hidden_channels)
        self.mlp_head = nn.Linear(hidden_channels, out_channels)
        
        # inner layers
        self.mixer_blocks = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.mixer_blocks.append(
                TransformerBlock(hidden_channels, 
                                 channel_expansion_factor, 
                                 dropout, 
                                 module_spec=None, 
                                 use_single_layer=use_single_layer)
            )
        # padding
        self.stride = window_size
        self.window_size = window_size
        self.pad_projector = nn.Linear(window_size*hidden_channels, hidden_channels)
        self.p_enc_1d_model_sum = Summer(PositionalEncoding1D(hidden_channels))

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.mixer_blocks:
            layer.reset_parameters()
        self.feat_encoder.reset_parameters()
        self.layernorm.reset_parameters()
        self.mlp_head.reset_parameters()

    def forward(self, edge_feats, edge_ts, batch_size, inds):
        # x : [ batch_size, graph_size, edge_dims+time_dims]
        edge_time_feats = self.feat_encoder(edge_feats, edge_ts)
        x = torch.zeros((batch_size * self.per_graph_size, 
                         edge_time_feats.size(1)), device=edge_feats.device)
        x[inds] = x[inds] + edge_time_feats         
        x = x. view(-1, self.per_graph_size//self.window_size, self.window_size*x.shape[-1])
        x = self.pad_projector(x)
        x = self.p_enc_1d_model_sum(x) 
        for i in range(self.num_layers):
            # apply to channel + feat dim
            x = self.mixer_blocks[i](x)    
        x = self.layernorm(x)
        x = torch.mean(x, dim=1)
        x = self.mlp_head(x)
        return x
    
################################################################################################
################################################################################################
################################################################################################

"""
Edge predictor
"""

class EdgePredictor_per_node(torch.nn.Module):
    """
    out = linear(src_node_feats) + linear(dst_node_feats)
    out = ReLU(out)
    """
    def __init__(self, dim_in_time, dim_in_node, predict_class):
        super().__init__()

        self.dim_in_time = dim_in_time
        self.dim_in_node = dim_in_node

        # dim_in_time + dim_in_node
        self.src_fc = torch.nn.Linear(dim_in_time+dim_in_node, 100)
        self.dst_fc = torch.nn.Linear(dim_in_time+dim_in_node, 100)
    
        self.out_fc = torch.nn.Linear(100, predict_class)
        self.reset_parameters()
        
    def reset_parameters(self, ):
        self.src_fc.reset_parameters()
        self.dst_fc.reset_parameters()
        self.out_fc.reset_parameters()

    def forward(self, h, neg_samples=1):
        num_edge = h.shape[0]//(neg_samples + 2)
        h_src = self.src_fc(h[:num_edge])
        h_pos_dst = self.dst_fc(h[num_edge:2 * num_edge])
        h_neg_dst = self.dst_fc(h[2 * num_edge:])
        
        h_pos_edge = torch.nn.functional.relu(h_src + h_pos_dst)
        h_neg_edge = torch.nn.functional.relu(h_src.tile(neg_samples, 1) + h_neg_dst)
        
        return self.out_fc(h_pos_edge), self.out_fc(h_neg_edge)
    
    
class STHN_Interface(nn.Module):
    def __init__(self, mlp_mixer_configs, edge_predictor_configs):
        super(STHN_Interface, self).__init__()

        self.time_feats_dim = edge_predictor_configs['dim_in_time']
        self.node_feats_dim = edge_predictor_configs['dim_in_node']

        if self.time_feats_dim > 0:
            self.base_model = Patch_Encoding(**mlp_mixer_configs)

        self.edge_predictor = EdgePredictor_per_node(**edge_predictor_configs)        
        self.creterion = nn.BCEWithLogitsLoss(reduction='none') 
        self.reset_parameters()            

    def reset_parameters(self):
        if self.time_feats_dim > 0:
            self.base_model.reset_parameters()
        self.edge_predictor.reset_parameters()
        
    def forward(self, model_inputs, neg_samples, node_feats):        
        pred_pos, pred_neg = self.predict(model_inputs, neg_samples, node_feats)
        all_pred = torch.cat((pred_pos, pred_neg), dim=0)
        all_edge_label = torch.cat((torch.ones_like(pred_pos), 
                                    torch.zeros_like(pred_neg)), dim=0)
        loss = self.creterion(all_pred, all_edge_label).mean()
        return loss, all_pred, all_edge_label
    
    def predict(self, model_inputs, neg_samples, node_feats):
        if self.time_feats_dim > 0 and self.node_feats_dim == 0:
            x = self.base_model(*model_inputs)
        elif self.time_feats_dim > 0 and self.node_feats_dim > 0:
            x = self.base_model(*model_inputs)
            x = torch.cat([x, node_feats], dim=1)
        elif self.time_feats_dim == 0 and self.node_feats_dim > 0:
            x = node_feats
        else:
            print('Either time_feats_dim or node_feats_dim must larger than 0!')
        
        pred_pos, pred_neg = self.edge_predictor(x, neg_samples=neg_samples)
        return pred_pos, pred_neg

class Multiclass_Interface(nn.Module):
    def __init__(self, mlp_mixer_configs, edge_predictor_configs):
        super(Multiclass_Interface, self).__init__()

        self.time_feats_dim = edge_predictor_configs['dim_in_time']
        self.node_feats_dim = edge_predictor_configs['dim_in_node']

        if self.time_feats_dim > 0:
            self.base_model = Patch_Encoding(**mlp_mixer_configs)

        self.edge_predictor = EdgePredictor_per_node(**edge_predictor_configs)        
        self.creterion = nn.CrossEntropyLoss(reduction='none')
        self.reset_parameters()            

    def reset_parameters(self):
        if self.time_feats_dim > 0:
            self.base_model.reset_parameters()
        self.edge_predictor.reset_parameters()
        
    def forward(self, model_inputs, neg_samples, node_feats):        
        pos_edge_label = model_inputs[-1].view(-1,1)
        model_inputs = model_inputs[:-1]
        pred_pos, pred_neg = self.predict(model_inputs, neg_samples, node_feats)
        
        all_pred = torch.cat((pred_pos, pred_neg), dim=0)
        all_edge_label = torch.squeeze(torch.cat((pos_edge_label, torch.zeros_like(pos_edge_label)), dim=0))
        loss = self.creterion(all_pred, all_edge_label).mean()
            
        return loss, all_pred, all_edge_label
    
    def predict(self, model_inputs, neg_samples, node_feats):
        if self.time_feats_dim > 0 and self.node_feats_dim == 0:
            x = self.base_model(*model_inputs)
        elif self.time_feats_dim > 0 and self.node_feats_dim > 0:
            x = self.base_model(*model_inputs)
            x = torch.cat([x, node_feats], dim=1)
        elif self.time_feats_dim == 0 and self.node_feats_dim > 0:
            x = node_feats
        else:
            print('Either time_feats_dim or node_feats_dim must larger than 0!')
        
        pred_pos, pred_neg = self.edge_predictor(x, neg_samples=neg_samples)
        return pred_pos, pred_neg

    
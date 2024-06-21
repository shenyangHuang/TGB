import timeit
import numpy as np
from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset
from tgb.linkproppred.evaluate import Evaluator

import argparse
from modules.sthn import set_seed, pre_compute_subgraphs, get_inputs_for_ind, check_data_leakage
import torch
import pandas as pd
import itertools
from tqdm import tqdm
import math
import os
import os.path as osp
from pathlib import Path
from tgb.utils.utils import set_random_seed, save_results


# Start...
start_overall = timeit.default_timer()

DATA = "thgl-myket"

MODEL_NAME = 'STHN'

# data loading
dataset = PyGLinkPropPredDataset(name=DATA, root="datasets")
train_mask = dataset.train_mask
val_mask = dataset.val_mask
test_mask = dataset.test_mask
data = dataset.get_TemporalData()
metric = dataset.eval_metric

print ("there are {} nodes and {} edges".format(dataset.num_nodes, dataset.num_edges))
print ("there are {} relation types".format(dataset.num_rels))


timestamp = data.t
head = data.src
tail = data.dst
edge_type = data.edge_type
neg_sampler = dataset.negative_sampler

print(data)
print(timestamp)
print(head)
print(tail)

train_data = data[train_mask]
val_data = data[val_mask]
test_data = data[test_mask]


metric = dataset.eval_metric
evaluator = Evaluator(name=DATA)
neg_sampler = dataset.negative_sampler

# for saving the results...
results_path = f'{osp.dirname(osp.abspath(__file__))}/saved_results'
if not osp.exists(results_path):
    os.mkdir(results_path)
    print('INFO: Create directory {}'.format(results_path))
Path(results_path).mkdir(parents=True, exist_ok=True)
results_filename = f'{results_path}/{MODEL_NAME}_{DATA}_results.json'


####################################################################
####################################################################
####################################################################


def print_model_info(model):
    print(model)
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters])
    print('Trainable Parameters: %d' % parameters)

def get_args():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='movie')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=600)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--max_edges', type=int, default=50)
    parser.add_argument('--num_edgeType', type=int, default=0, help='num of edgeType')
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--predict_class', action='store_true')
    
    # model
    parser.add_argument('--window_size', type=int, default=5)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--model', type=str, default='sthn') 
    parser.add_argument('--neg_samples', type=int, default=1)
    parser.add_argument('--extra_neg_samples', type=int, default=5)
    parser.add_argument('--num_neighbors', type=int, default=50)
    parser.add_argument('--channel_expansion_factor', type=int, default=2)
    parser.add_argument('--sampled_num_hops', type=int, default=1)
    parser.add_argument('--time_dims', type=int, default=100)
    parser.add_argument('--hidden_dims', type=int, default=100)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--check_data_leakage', action='store_true')
    
    parser.add_argument('--ignore_node_feats', action='store_true')
    parser.add_argument('--node_feats_as_edge_feats', action='store_true')
    parser.add_argument('--ignore_edge_feats', action='store_true')
    parser.add_argument('--use_onehot_node_feats', action='store_true')
    parser.add_argument('--use_type_feats', action='store_true')

    parser.add_argument('--use_graph_structure', action='store_true')
    parser.add_argument('--structure_time_gap', type=int, default=2000)
    parser.add_argument('--structure_hops', type=int, default=1) 

    parser.add_argument('--use_node_cls', action='store_true')
    parser.add_argument('--use_cached_subgraph', action='store_true')
    
    parser.add_argument('--seed', type=int, help='Random seed', default=1)
    parser.add_argument('--num_run', type=int, help='Number of iteration runs', default=5)
    return parser.parse_args()


def load_model(args):
    # get model
    edge_predictor_configs = {
        'dim_in_time': args.time_dims,
        'dim_in_node': args.node_feat_dims,
        'predict_class': 1 if not args.predict_class else args.num_edgeType+1,
    }
    if args.model == 'sthn':
        if args.predict_class:
            from modules.sthn import Multiclass_Interface as STHN_Interface
        else:
            from modules.sthn import STHN_Interface
        from modules.sthn import link_pred_train

        mixer_configs = {
            'per_graph_size'  : args.max_edges,
            'time_channels'   : args.time_dims, 
            'input_channels'  : args.edge_feat_dims, 
            'hidden_channels' : args.hidden_dims, 
            'out_channels'    : args.hidden_dims,
            'num_layers'      : args.num_layers,
            'dropout'         : args.dropout,
            'channel_expansion_factor': args.channel_expansion_factor,
            'window_size'     : args.window_size,
            'use_single_layer' : False
        }  
        
    else:
        NotImplementedError()

    model = STHN_Interface(mixer_configs, edge_predictor_configs)
    for k, v in model.named_parameters():
        print(k, v.requires_grad)

    print_model_info(model)

    return model, args, link_pred_train

def load_graph(data):
    df = pd.DataFrame({
        'idx': np.arange(len(data.t)),
        'src': data.src,
        'dst': data.dst,
        'time': data.t,
        'label': data.edge_type,
    })

    num_nodes = max(int(df['src'].max()), int(df['dst'].max())) + 1 

    ext_full_indptr = np.zeros(num_nodes + 1, dtype=np.int32)
    ext_full_indices = [[] for _ in range(num_nodes)]
    ext_full_ts = [[] for _ in range(num_nodes)]
    ext_full_eid = [[] for _ in range(num_nodes)]

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        src = int(row['src'])
        dst = int(row['dst'])
        
        ext_full_indices[src].append(dst)
        ext_full_ts[src].append(row['time'])
        ext_full_eid[src].append(idx)
        
    for i in tqdm(range(num_nodes)):
        ext_full_indptr[i + 1] = ext_full_indptr[i] + len(ext_full_indices[i])

    ext_full_indices = np.array(list(itertools.chain(*ext_full_indices)))
    ext_full_ts = np.array(list(itertools.chain(*ext_full_ts)))
    ext_full_eid = np.array(list(itertools.chain(*ext_full_eid)))

    print('Sorting...')

    def tsort(i, indptr, indices, t, eid):
        beg = indptr[i]
        end = indptr[i + 1]
        sidx = np.argsort(t[beg:end])
        indices[beg:end] = indices[beg:end][sidx]
        t[beg:end] = t[beg:end][sidx]
        eid[beg:end] = eid[beg:end][sidx] 

    for i in tqdm(range(ext_full_indptr.shape[0] - 1)):
        tsort(i, ext_full_indptr, ext_full_indices, ext_full_ts, ext_full_eid)

    print('saving...')

    np.savez('/tmp/ext_full.npz', indptr=ext_full_indptr,
            indices=ext_full_indices, ts=ext_full_ts, eid=ext_full_eid)
    g = np.load('/tmp/ext_full.npz')
    return g, df

def load_all_data(args):

    # load graph
    g, df = load_graph(data)

    args.train_mask = train_mask.numpy()
    args.val_mask   = val_mask.numpy()
    args.test_mask = test_mask.numpy()
    args.num_edges = len(df)

    print('Train %d, Valid %d, Test %d'%(sum(args.train_mask), 
                                         sum(args.val_mask),
                                         sum(test_mask)))
    
    args.num_nodes = max(int(df['src'].max()), int(df['dst'].max())) + 1
    args.num_edges = len(df)

    print('Num nodes %d, num edges %d'%(args.num_nodes, args.num_edges))

    # load feats 
    node_feats, edge_feats = dataset.node_feat, dataset.edge_feat
    node_feat_dims = 0 if node_feats is None else node_feats.shape[1]
    edge_feat_dims = 0 if edge_feats is None else edge_feats.shape[1]

    # feature pre-processing
    if args.use_onehot_node_feats:
        print('>>> Use one-hot node features')
        node_feats = torch.eye(args.num_nodes)
        node_feat_dims = node_feats.size(1)

    if args.ignore_node_feats:
        print('>>> Ignore node features')
        node_feats = None
        node_feat_dims = 0

    if args.use_type_feats:
        edge_type = df.label.values
        print(edge_type)
        print(edge_type.sum())
        args.num_edgeType = len(set(edge_type.tolist()))
        edge_feats = torch.nn.functional.one_hot(torch.from_numpy(edge_type), 
                                                 num_classes=args.num_edgeType)
        edge_feat_dims = edge_feats.size(1)
        
    print('Node feature dim %d, edge feature dim %d'%(node_feat_dims, edge_feat_dims))
    
    # double check (if data leakage then cannot continue the code)
    if args.check_data_leakage:
        check_data_leakage(args, g, df)

    args.node_feat_dims = node_feat_dims
    args.edge_feat_dims = edge_feat_dims
    
    if node_feats != None:
        node_feats = node_feats.to(args.device)
    if edge_feats != None:
        edge_feats = edge_feats.to(args.device)
    
    return node_feats, edge_feats, g, df, args

####################################################################
####################################################################
####################################################################

@torch.no_grad()
def test(data, test_mask, model, neg_sampler, split_mode):
    r"""
    Evaluated the dynamic link prediction
    Evaluation happens as 'one vs. many', meaning that each positive edge is evaluated against many negative edges

    Parameters:
        data: a dataset object
        test_mask: required masks to load the test set edges
        neg_sampler: an object that gives the negative edges corresponding to each positive edge
        split_mode: specifies whether it is the 'val' or 'test' set to correctly load the negatives
    Returns:
        perf_metric: the result of the performance evaluation
    """
    test_subgraphs  = pre_compute_subgraphs(args, g, df, mode='test' if split_mode == 'test' else 'valid', negative_sampler=neg_sampler, split_mode=split_mode)
    perf_list = []
    
    if split_mode == 'test':
        cur_df = df[args.test_mask]
    elif split_mode == 'val':
        cur_df = df[args.val_mask]
    neg_samples = 20
    cached_neg_samples = 20

    test_loader = cur_df.groupby(cur_df.index // args.batch_size)
    pbar = tqdm(total=len(test_loader))
    pbar.set_description('%s mode with negative samples %d ...'%(split_mode, neg_samples))        
    
    ###################################################
    # compute + training + fetch all scores
    cur_inds = 0

    for ind in range(len(test_loader)):
        ###################################################
        inputs, subgraph_node_feats, cur_inds = get_inputs_for_ind(test_subgraphs, 'test' if split_mode == 'test' else 'tgb-val', cached_neg_samples, neg_samples, node_feats, edge_feats, cur_df, cur_inds, ind, args)
        
        loss, pred, edge_label = model(inputs, neg_samples, subgraph_node_feats)
        # print(ind, [l for l in inputs], pred.shape)

        input_dict = {
            "y_pred_pos": np.array([pred.cpu()[0]]),
            "y_pred_neg": np.array(pred.cpu()[1:]),
            "eval_metric": [metric],
        }
        perf_list.append(evaluator.eval(input_dict)[metric])

    perf_metrics_mean = float(np.mean(perf_list))
    perf_metrics_std = float(np.std(perf_list))

    return perf_metrics_mean, perf_metrics_std, perf_list


args = get_args()

args.use_graph_structure = True
args.use_onehot_node_feats = False
args.ignore_node_feats = False # we only use graph structure
args.use_type_feats = True # type encoding
args.use_cached_subgraph = True

print(args)

args.device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
args.device = torch.device(args.device)
SEED = args.seed
BATCH_SIZE = args.batch_size
NUM_RUNS = args.num_run
set_seed(SEED)


###################################################
# load feats + graph
node_feats, edge_feats, g, df, args = load_all_data(args)

###################################################
# get model 
model, args, link_pred_train = load_model(args)

###################################################

print("==========================================================")
print(f"=================*** {MODEL_NAME}: LinkPropPred: {DATA} ***=============")
print("==========================================================")


# for saving the results...
results_path = f'{osp.dirname(osp.abspath(__file__))}/saved_results'
if not osp.exists(results_path):
    os.mkdir(results_path)
    print('INFO: Create directory {}'.format(results_path))
Path(results_path).mkdir(parents=True, exist_ok=True)
results_filename = f'{results_path}/{MODEL_NAME}_{DATA}_results.json'



for run_idx in range(NUM_RUNS):
    print('-------------------------------------------------------------------------------')
    print(f"INFO: >>>>> Run: {run_idx} <<<<<")
    start_run = timeit.default_timer()

    # set the seed for deterministic results...
    torch.manual_seed(run_idx + SEED)
    set_random_seed(run_idx + SEED)

    # define an early stopper
    save_model_dir = f'{osp.dirname(osp.abspath(__file__))}/saved_models/'
    save_model_id = f'{MODEL_NAME}_{DATA}_{SEED}_{run_idx}'
    # early_stopper = EarlyStopMonitor(save_model_dir=save_model_dir, save_model_id=save_model_id, 
    #                                 tolerance=TOLERANCE, patience=PATIENCE)

    # ==================================================== Train & Validation
    # loading the validation negative samples

    # Link prediction
    start_val = timeit.default_timer()
    print('Train link prediction task from scratch ...')
    model = link_pred_train(model.to(args.device), args, g, df, node_feats, edge_feats)

    dataset.load_val_ns()

    # Validation ...
    
    perf_metrics_val_mean, perf_metrics_val_std, perf_list_val = test(data.to(args.device), test_mask, model.to(args.device), neg_sampler, split_mode='val')
    end_val = timeit.default_timer()

    print(f"INFO: val: Evaluation Setting: >>> ONE-VS-MANY <<< ")
    print(f"\tval: {metric}: {perf_metrics_val_mean: .4f} ± {perf_metrics_val_std: .4f}")
    val_time = timeit.default_timer() - start_val
    print(f"\tval: Elapsed Time (s): {val_time: .4f}")


    dataset.load_test_ns()

    # testing ...
    start_test = timeit.default_timer()
    perf_metrics_test_mean, perf_metrics_test_std, perf_list_test = test(data.to(args.device), test_mask, model.to(args.device), neg_sampler, split_mode='test')
    end_test = timeit.default_timer()

    print(f"INFO: Test: Evaluation Setting: >>> ONE-VS-MANY <<< ")
    print(f"\tTest: {metric}: {perf_metrics_test_mean: .4f} ± {perf_metrics_test_std: .4f}")
    test_time = timeit.default_timer() - start_test
    print(f"\tTest: Elapsed Time (s): {test_time: .4f}")

    save_results({'model': MODEL_NAME,
                  'data': DATA,
                  'run': run_idx,
                  'seed': SEED,
                  f'val {metric}': f'{perf_metrics_val_mean: .4f} ± {perf_metrics_val_std: .4f}' ,
                  f'test {metric}': f'{perf_metrics_test_mean: .4f} ± {perf_metrics_test_std: .4f}' ,
                  'test_time': test_time,
                  'tot_train_val_time': val_time
                  }, 
    results_filename)

    print(f"INFO: >>>>> Run: {run_idx}, elapsed time: {timeit.default_timer() - start_run: .4f} <<<<<")
    print('-------------------------------------------------------------------------------')

print(f"Overall Elapsed Time (s): {timeit.default_timer() - start_overall: .4f}")
print("==============================================================")

# save_results({'model': MODEL_NAME,
#             'data': DATA,
#             'run': 1,
#             'seed': SEED,
#             metric: perf_metric_test,
#             'test_time': test_time,
#             'tot_train_val_time': 'NA'
#             }, 
#     results_filename)
"""
Complex Evolutional Pattern Learning for Temporal Knowledge Graph Reasoning
Reference:
- https://github.com/Lee-zix/CEN
Zixuan Li, Saiping Guan, Xiaolong Jin, Weihua Peng, Yajuan Lyu , Yong Zhu, Long Bai, Wei Li, Jiafeng Guo, Xueqi Cheng. 
Complex Evolutional Pattern Learning for Temporal Knowledge Graph Reasoning. ACL 2022.
"""
import timeit
import os
import sys
import os.path as osp
from pathlib import Path
import numpy as np
import torch
import random
from tqdm import tqdm
import json
# internal imports
modules_path = osp.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(modules_path)
from modules.rrgcn import RecurrentRGCNCEN
from tgb.utils.utils import set_random_seed, split_by_time,  save_results
from modules.tkg_utils import get_args_cen, reformat_ts
from modules.tkg_utils_dgl import build_sub_graph
from tgb.linkproppred.evaluate import Evaluator
from tgb.linkproppred.dataset import LinkPropPredDataset 

def test(model, history_len, history_list, test_list, num_rels, num_nodes, use_cuda, model_name, mode, split_mode):
    """
    Test the model
    :param model: model used to test
    :param history_list:    all input history snap shot list, not include output label train list or valid list
    :param test_list:   test triple snap shot list
    :param num_rels:    number of relations
    :param num_nodes:   number of nodes
    :param use_cuda:
    :param model_name:
    :param mode:
    :param split_mode: 'test' or 'val' to state which negative samples to load
    :return mrr
    """
    print("Testing for mode: ", split_mode)
    if split_mode == 'test':
        timesteps_to_eval = test_timestamps_orig
    else:
        timesteps_to_eval = val_timestamps_orig

    idx = 0

    if mode == "test":
        # test mode: load parameter form file 
        if use_cuda:
            checkpoint = torch.load(model_name, map_location=torch.device(args.gpu))
        else:
            checkpoint = torch.load(model_name, map_location=torch.device('cpu'))        
        # use best stat checkpoint:
        print("Load Model name: {}. Using best epoch : {}".format(model_name, checkpoint['epoch']))  
        print("\n"+"-"*10+"start testing"+"-"*10+"\n")
        model.load_state_dict(checkpoint['state_dict'])

    model.eval()

    input_list = [snap for snap in history_list[-history_len:]] 
    perf_list_all = []
    hits_list_all = []
    perf_per_rel = {}
    for rel in range(num_rels):
        perf_per_rel[rel] = []

    for time_idx, test_snap in enumerate(tqdm(test_list)):
        history_glist = [build_sub_graph(num_nodes, num_rels, g, use_cuda, args.gpu) for g in input_list]    
        test_triples_input = torch.LongTensor(test_snap).cuda() if use_cuda else torch.LongTensor(test_snap)
        timesteps_batch =timesteps_to_eval[time_idx]*np.ones(len(test_triples_input[:,0]))

        neg_samples_batch = neg_sampler.query_batch(test_triples_input[:,0], test_triples_input[:,2], 
                                timesteps_batch, edge_type=test_triples_input[:,1], split_mode=split_mode)
        pos_samples_batch = test_triples_input[:,2]

        _, perf_list, hits_list = model.predict(history_glist, test_triples_input, use_cuda, neg_samples_batch, pos_samples_batch, 
                                    evaluator, METRIC) 

        perf_list_all.extend(perf_list)
        hits_list_all.extend(hits_list)
        if split_mode == "test":
            if args.log_per_rel:
                for score, rel in zip(perf_list, test_triples_input[:,1].tolist()):
                    perf_per_rel[rel].append(score)
        # reconstruct history graph list
        input_list.pop(0)
        input_list.append(test_snap)
        idx += 1
    if split_mode == "test":
        if args.log_per_rel:   
            for rel in range(num_rels):
                if len(perf_per_rel[rel]) > 0:
                    perf_per_rel[rel] = float(np.mean(perf_per_rel[rel]))
                else:
                    perf_per_rel.pop(rel)
    mrr = np.mean(perf_list_all) 
    hits10 = np.mean(hits_list_all)
    return mrr, perf_per_rel, hits10



def run_experiment(args, trainvalidtest_id=0, n_hidden=None, n_layers=None, dropout=None, n_bases=None):
    '''
    Run experiment for CEN model
    :param args: arguments for the model
    :param trainvalidtest_id: -1: pretrainig, 0: curriculum training (to find best test history len), 1: test on valid set, 2: test on test set
    :param n_hidden: number of hidden units
    :param n_layers: number of layers
    :param dropout: dropout rate
    :param n_bases: number of bases
    return: mrr, perf_per_rel: mean reciprocal rank and performance per relation
    '''
    # 1) load configuration for grid search the best configuration
    if n_hidden:
        args.n_hidden = n_hidden
    if n_layers:
        args.n_layers = n_layers
    if dropout:
        args.dropout = dropout
    if n_bases:
        args.n_bases = n_bases
    test_history_len = args.test_history_len
    # 2) set save model path
    save_model_dir = f'{osp.dirname(osp.abspath(__file__))}/saved_models/'
    if not osp.exists(save_model_dir):
        os.mkdir(save_model_dir)
        print('INFO: Create directory {}'.format(save_model_dir))
    test_model_name= f'{MODEL_NAME}_{DATA}_{SEED}_{args.run_nr}_{args.test_history_len}'
    test_state_file = save_model_dir+test_model_name
    perf_per_rel ={}
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    # create stat

    model = RecurrentRGCNCEN(args.decoder,
                            args.encoder,
                            num_nodes,
                            num_rels,
                            args.n_hidden,
                            args.opn,
                            sequence_len=args.train_history_len,
                            num_bases=args.n_bases,
                            num_basis=args.n_basis,
                            num_hidden_layers=args.n_layers,
                            dropout=args.dropout,
                            self_loop=args.self_loop,
                            skip_connect=args.skip_connect,
                            layer_norm=args.layer_norm,
                            input_dropout=args.input_dropout,
                            hidden_dropout=args.hidden_dropout,
                            feat_dropout=args.feat_dropout,
                            entity_prediction=args.entity_prediction,
                            relation_prediction=args.relation_prediction,
                            use_cuda=use_cuda,
                            gpu = args.gpu)
    if use_cuda:
        torch.cuda.set_device(args.gpu)
        model.cuda()

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    
    if trainvalidtest_id == 1:  # normal test on validation set  Note that mode=test
        if os.path.exists(test_state_file):
            mrr, _, hits10 = test(model, args.test_history_len, train_list, valid_list, num_rels, num_nodes, use_cuda, 
                        test_state_file, "test", split_mode="val")      
        else:
            print('Cannot do testing because model does not exist: ', test_state_file)
            mrr = 0
            hits10 = 0
    elif trainvalidtest_id == 2: # normal test on test set
        if os.path.exists(test_state_file):
            mrr, perf_per_rel, hits10 = test(model, args.test_history_len, train_list+valid_list, test_list, num_rels, num_nodes, use_cuda, 
                        test_state_file, "test", split_mode="test")
        else:
            print('Cannot do testing because model does not exist: ', test_state_file)
            mrr = 0
            hits10 = 0
    elif trainvalidtest_id == -1:
        print("-------------start pre training model with history length {}----------\n".format(args.start_history_len))
        model_name= f'{MODEL_NAME}_{DATA}_{SEED}_{args.run_nr}_{args.start_history_len}'
        model_state_file = save_model_dir + model_name
        print("Sanity Check: stat name : {}".format(model_state_file))
        print("Sanity Check: Is cuda available ? {}".format(torch.cuda.is_available()))
            
        best_mrr = 0
        best_epoch = 0
        best_hits10= 0

        ## training loop
        for epoch in range(args.n_epochs):
            model.train()
            losses = []

            idx = [_ for _ in range(len(train_list))]
            random.shuffle(idx)
            for train_sample_num in idx:
                if train_sample_num == 0 or train_sample_num == 1: continue
                if train_sample_num - args.start_history_len<0:
                    input_list = train_list[0: train_sample_num]
                    output = train_list[1:train_sample_num+1]
                else:
                    input_list = train_list[train_sample_num-args.start_history_len: train_sample_num]
                    output = train_list[train_sample_num-args.start_history_len+1:train_sample_num+1]

                # generate history graph
                history_glist = [build_sub_graph(num_nodes, num_rels, snap, use_cuda, args.gpu) for snap in input_list]
                output = [torch.from_numpy(_).long().cuda() for _ in output] if use_cuda else [torch.from_numpy(_).long() for _ in output]

                loss= model.get_loss(history_glist, output[-1], None, use_cuda)
                losses.append(loss.item())

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)  # clip gradients
                optimizer.step()
                optimizer.zero_grad()

            print("His {:04d}, Epoch {:04d} | Ave Loss: {:.4f} | Best MRR {:.4f} | Model {} "
                .format(args.start_history_len, epoch, np.mean(losses), best_mrr, model_name))
            
            #! checking GPU usage
            free_mem, total_mem = torch.cuda.mem_get_info()
            print ("--------------GPU memory usage-----------")
            print ("there are ", free_mem, " free memory")
            print ("there are ", total_mem, " total available memory")
            print ("there are ", total_mem - free_mem, " used memory")
            print ("--------------GPU memory usage-----------")

            # validation        
            if epoch % args.evaluate_every == 0:
                mrr, _, hits10 = test(model, args.start_history_len, train_list, valid_list, num_rels, num_nodes, use_cuda, 
                                model_state_file, mode="train", split_mode= "val")

                if mrr< best_mrr:
                    if epoch >= args.n_epochs or epoch - best_epoch > 5:
                        break
                else:
                    best_mrr = mrr
                    best_epoch = epoch
                    best_hits10 = hits10
                    torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, model_state_file)

        
    elif trainvalidtest_id == 0: #curriculum training
        model_name= f'{MODEL_NAME}_{DATA}_{SEED}_{args.run_nr}_{args.start_history_len}'
        init_state_file = save_model_dir + model_name
        init_checkpoint = torch.load(init_state_file, map_location=torch.device(args.gpu))
        # use best stat checkpoint:
        print("Load Previous Model name: {}. Using best epoch : {}".format(init_state_file, init_checkpoint['epoch']))  
        print("\n"+"-"*10+"Load model with history length {}".format(args.start_history_len)+"-"*10+"\n")
        model.load_state_dict(init_checkpoint['state_dict'])
        test_history_len = args.start_history_len

        mrr, _, hits10 = test(model, 
                    args.start_history_len,
                    train_list,
                    valid_list, 
                    num_rels, 
                    num_nodes, 
                    use_cuda, 
                    init_state_file,  
                    mode="test", split_mode= "val") 
        best_mrr_list = [mrr.item()]         
        best_hits_list = [hits10.item()]                                          
        # start knowledge distillation
        ks_idx = 0
        for history_len in range(args.start_history_len+1, args.train_history_len+1, 1):
            # current model
            print("best mrr list :", best_mrr_list)
            # lr = 0.1*args.lr - 0.002*args.lr*ks_idx
            optimizer = torch.optim.Adam(model.parameters(), lr=0.1*args.lr, weight_decay=0.00001)
            model_name= f'{MODEL_NAME}_{DATA}_{SEED}_{args.run_nr}_{history_len}'
            model_state_file = save_model_dir + model_name

            print("Sanity Check: stat name : {}".format(model_state_file))

            # load model with the least history length
            prev_model_name= f'{MODEL_NAME}_{DATA}_{SEED}_{args.run_nr}_{history_len-1}'
            prev_state_file = save_model_dir + prev_model_name
            checkpoint = torch.load(prev_state_file, map_location=torch.device(args.gpu)) 
            model.load_state_dict(checkpoint['state_dict']) 
            print("\n"+"-"*10+"start knowledge distillation for history length at "+ str(history_len)+"-"*10+"\n")
 
            best_mrr = 0
            best_hits10 = 0
            best_epoch = 0
            for epoch in range(args.n_epochs):
                model.train()
                losses = []

                idx = [_ for _ in range(len(train_list))]
                random.shuffle(idx)
                for train_sample_num in idx:
                    if train_sample_num == 0 or train_sample_num == 1: continue
                    if train_sample_num - history_len<0:
                        input_list = train_list[0: train_sample_num]
                        output = train_list[1:train_sample_num+1]
                    else:
                        input_list = train_list[train_sample_num - history_len: train_sample_num]
                        output = train_list[train_sample_num-history_len+1:train_sample_num+1]

                    # generate history graph
                    history_glist = [build_sub_graph(num_nodes, num_rels, snap, use_cuda, args.gpu) for snap in input_list]
                    output = [torch.from_numpy(_).long().cuda() for _ in output] if use_cuda else [torch.from_numpy(_).long() for _ in output]

                    loss= model.get_loss(history_glist, output[-1], None, use_cuda)
                    # print(loss)
                    losses.append(loss.item())

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)  # clip gradients
                    optimizer.step()
                    optimizer.zero_grad()

                print("His {:04d}, Epoch {:04d} | Ave Loss: {:.4f} |Best MRR {:.4f} | Model {} "
                    .format(history_len, epoch, np.mean(losses), best_mrr, model_name))

                #! checking GPU usage
                free_mem, total_mem = torch.cuda.mem_get_info()
                print ("--------------GPU memory usage-----------")
                print ("there are ", free_mem, " free memory")
                print ("there are ", total_mem, " total available memory")
                print ("there are ", total_mem - free_mem, " used memory")
                print ("--------------GPU memory usage-----------")

                # validation
                if epoch % args.evaluate_every == 0:
                    mrr, _, hits10 = test(model, history_len, train_list, valid_list, num_rels, num_nodes, use_cuda, 
                                model_state_file, mode="train", split_mode= "val")
                    
                    if mrr< best_mrr:
                        if epoch >= args.n_epochs or epoch-best_epoch>2:
                            break
                    else:
                        best_mrr = mrr
                        best_epoch = epoch
                        best_hits10 = hits10
                        torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, model_state_file)  
            mrr, _, hits10 = test(model, history_len, train_list, valid_list, num_rels, num_nodes, use_cuda, 
                        model_state_file, mode="test", split_mode= "val")
            ks_idx += 1
            if mrr.item() < max(best_mrr_list):
                test_history_len = history_len-1
                print("early stopping, best history length: ", test_history_len)
                break
            else:
                best_mrr_list.append(mrr.item())
                best_hits_list.append(hits10.item())
        
    return mrr, test_history_len, perf_per_rel, hits10



# ==================
# ==================
# ==================

start_overall = timeit.default_timer()

# set hyperparameters
args, _ = get_args_cen()
args.dataset = 'tkgl-icews'

SEED = args.seed  # set the random seed for consistency
set_random_seed(SEED)

DATA=args.dataset
MODEL_NAME = 'CEN'

print("logging mrrs per relation: ", args.log_per_rel)
print("do test and valid? do only test no validation?: ", args.validtest, args.test_only)

# load data
dataset = LinkPropPredDataset(name=DATA, root="datasets", preprocess=True)

num_rels = dataset.num_rels
num_nodes = dataset.num_nodes 
subjects = dataset.full_data["sources"]
objects= dataset.full_data["destinations"]
relations = dataset.edge_type

timestamps_orig = dataset.full_data["timestamps"]
timestamps = reformat_ts(timestamps_orig, DATA) # stepsize:1
all_quads = np.stack((subjects, relations, objects, timestamps), axis=1)

train_data = all_quads[dataset.train_mask]
val_data = all_quads[dataset.val_mask]
test_data = all_quads[dataset.test_mask]

val_timestamps_orig = list(set(timestamps_orig[dataset.val_mask])) # needed for getting the negative samples
val_timestamps_orig.sort()
test_timestamps_orig = list(set(timestamps_orig[dataset.test_mask])) # needed for getting the negative samples
test_timestamps_orig.sort()

train_list = split_by_time(train_data)
valid_list = split_by_time(val_data)
test_list = split_by_time(test_data)

# evaluation metric
METRIC = dataset.eval_metric
evaluator = Evaluator(name=DATA)
neg_sampler = dataset.negative_sampler
#load the ns samples 
dataset.load_val_ns()
dataset.load_test_ns()

if args.grid_search:
    print("TODO: implement hyperparameter grid search")
# single run
else:
    
    start_train = timeit.default_timer()
    if args.validtest:
        print('directly start testing')
        if args.test_history_len_2 != args.test_history_len:
            args.test_history_len = args.test_history_len_2 # hyperparameter value as given in original paper 
    else:
        print('running pretrain and train')
        # pretrain
        mrr, _, _, hits10 = run_experiment(args, trainvalidtest_id=-1)
        # train
        mrr, args.test_history_len, _, hits10 = run_experiment(args, trainvalidtest_id=0) # overwrite test_history_len with 
        # the best history len (for valid mrr)       
        
    if args.test_only == False:
        print("running test (on val and test dataset) with test_history_len of: ", args.test_history_len)
        # test on val set
        val_mrr, _, _, val_hits10 = run_experiment(args, trainvalidtest_id=1)
    else:
        val_mrr = 0
        val_hits10 = 0

    # test on test set
    start_test = timeit.default_timer()
    test_mrr, _, perf_per_rel, test_hits10 = run_experiment(args, trainvalidtest_id=2)

test_time = timeit.default_timer() - start_test
all_time = timeit.default_timer() - start_train
print(f"\tTest: Elapsed Time (s): {test_time: .4f}")
print(f"\Train and Test: Elapsed Time (s): {all_time: .4f}")

print(f"\tTest: {METRIC}: {test_mrr: .4f}")
print(f"\tValid: {METRIC}: {val_mrr: .4f}")

# saving the results...
results_path = f'{osp.dirname(osp.abspath(__file__))}/saved_results'
if not osp.exists(results_path):
    os.mkdir(results_path)
    print('INFO: Create directory {}'.format(results_path))
Path(results_path).mkdir(parents=True, exist_ok=True)
results_filename = f'{results_path}/{MODEL_NAME}_{DATA}_results.json'

save_results({'model': MODEL_NAME,
              'data': DATA,
              'run': args.run_nr,
              'seed': SEED,
              'test_history_len': args.test_history_len,
              f'val {METRIC}': float(val_mrr),
              f'test {METRIC}': float(test_mrr),
              'test_time': test_time,
              'tot_train_val_time': all_time,
              'test_hits10': float(test_hits10)
              }, 
    results_filename)

if args.log_per_rel:
    results_filename = f'{results_path}/{MODEL_NAME}_{DATA}_results_per_rel.json'
    with open(results_filename, 'w') as json_file:
        json.dump(perf_per_rel, json_file)


sys.exit()
"""
Complex Evolutional Pattern Learning for Temporal Knowledge Graph Reasoning
Reference:
- https://github.com/Lee-zix/CEN
Zixuan Li, Saiping Guan, Xiaolong Jin, Weihua Peng, Yajuan Lyu , Yong Zhu, Long Bai, Wei Li, Jiafeng Guo, Xueqi Cheng. 
Complex Evolutional Pattern Learning for Temporal Knowledge Graph Reasoning. ACL 2022.
"""


import timeit
import argparse
import itertools
import os
import sys
import time
import pickle
import copy
import os.path as osp

import dgl
import numpy as np
import torch
# from tqdm import tqdm
import random
import torch.nn.modules.rnn
from collections import defaultdict

# internal imports
from tgb_modules.rrgcn import RecurrentRGCN
from tgb.utils.utils import set_random_seed
from tgb_modules.recurrencybaseline_utils import reformat_ts
from tgb_modules.cen_utils import split_by_time, build_sub_graph
from tgb.linkproppred.evaluate import Evaluator
from tgb.linkproppred.dataset import LinkPropPredDataset 


# from rgcn.knowledge_graph import _read_triplets_as_list
# from src.hyperparameter_range import hp_range
# from rgcn import utils
# from rgcn.utils import build_sub_graph
# os.environ['KMP_DUPLICATE_LIB_OK']='True'


#TODOs:
# implement data loading and make sure its in the correct format
# integrate test()
# store and load models at correct location
# 

def test(model, history_len, history_list, test_list, num_rels, num_nodes, use_cuda, model_name, mode, neg_samples):
    """
    :param model: model used to test
    :param history_list:    all input history snap shot list, not include output label train list or valid list
    :param test_list:   test triple snap shot list
    :param num_rels:    number of relations
    :param num_nodes:   number of nodes
    :param use_cuda:
    :param model_name:
    :param mode
    : param neg_samples: negative samples
    :return mrr
    """

    idx = 0
    start_time = len(history_list)
    if mode == "test":
        # test mode: load parameter form file
        if use_cuda:
            checkpoint = torch.load(model_name, map_location=torch.device(args.gpu))
        else:
            checkpoint = torch.load(model_name, map_location=torch.device('cpu'))
        print("Load Model name: {}. Using best epoch : {}".format(model_name, checkpoint['epoch']))  # use best stat checkpoint
        print("\n"+"-"*10+"start testing"+"-"*10+"\n")
        model.load_state_dict(checkpoint['state_dict'])

    model.eval()
    # do not have inverse relation in test input
    input_list = [snap for snap in history_list[-history_len:]] #TODO: how do we deal with this! 

    for time_idx, test_snap in enumerate(test_list):
        tc = start_time + time_idx
        history_glist = [build_sub_graph(num_nodes, num_rels, g, use_cuda, args.gpu) for g in input_list]
    
        test_triples_input = torch.LongTensor(test_snap).cuda() if use_cuda else torch.LongTensor(test_snap)
        final_score = model.predict(history_glist, test_triples_input, use_cuda, neg_samples)


        # used to global statistic
        mrr  = 0 # TODO

        # reconstruct history graph list
        input_list.pop(0)
        input_list.append(test_snap)
        idx += 1
    
    mrr = [] # TODO
    return mrr



def run_experiment(args, trainvalidtest_id=0, n_hidden=None, n_layers=None, dropout=None, n_bases=None):
    # load configuration for grid search the best configuration
    if n_hidden:
        args.n_hidden = n_hidden
    if n_layers:
        args.n_layers = n_layers
    if dropout:
        args.dropout = dropout
    if n_bases:
        args.n_bases = n_bases

    neg_samples = 0
    neg_samples_valid =  0

    # 1) TODO: load graph data
    # data = utils.load_data(args.dataset)
    # train_list = utils.split_by_time(data.train)
    # valid_list = utils.split_by_time(data.valid)
    # test_list = utils.split_by_time(data.test)
    # total_data = np.concatenate((data.train, data.valid, data.test), axis=0)
    # print("total data length ", len(total_data))
    # num_nodes = data.num_nodes
    # num_rels = data.num_rels


    # train_list = add_inverse(train_list, num_rels)
    # valid_list = add_inverse(valid_list, num_rels)
    # test_list = add_inverse(test_list, num_rels)

    # 2) set save model path
    save_model_dir = f'{osp.dirname(osp.abspath(__file__))}/saved_models/'
    test_model_name= f'{MODEL_NAME}_{DATA}_{SEED}_{args.run_nr}_{args.test_history_len}'
    test_state_file = save_model_dir+test_model_name

    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    # create stat
    model = RecurrentRGCN(args.decoder,
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
            mrr = test(model, args.test_history_len, 
                        train_list, 
                        valid_list, 
                        num_rels, 
                        num_nodes, 
                        use_cuda, 
                        test_state_file, 
                        "test")                                 
    elif trainvalidtest_id == 2: # normal test on test set
        if os.path.exists(test_state_file):
            mrr = test(model, args.test_history_len, 
                        train_list+valid_list, 
                        test_list, 
                        num_rels, 
                        num_nodes, 
                        use_cuda, 
                        test_state_file,  neg_samples,
                        "test")

    elif trainvalidtest_id == -1:
        print("-------------start pre training model with history length {}----------\n".format(args.start_history_len))
        model_name= f'{MODEL_NAME}_{DATA}_{SEED}_{args.run_nr}_{args.start_history_len}'
        # if not os.path.exists('../models/{}/'.format(args.dataset)):
        #     os.makedirs('../models/{}/'.format(args.dataset))
        # model_state_file = '../models/{}/{}'.format(args.dataset, model_name)  #TODO
        print("Sanity Check: stat name : {}".format(model_state_file))
        print("Sanity Check: Is cuda available ? {}".format(torch.cuda.is_available()))
            
        best_mrr = 0
        best_epoch = 0
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

            # validation        
            if epoch % args.evaluate_every == 0:
                mrr = test(model,
                                args.start_history_len, 
                                train_list, 
                                valid_list, 
                                num_rels, 
                                num_nodes, 
                                use_cuda, 
                                model_state_file, neg_samples_valid,
                                mode="train")
                

                if mrr< best_mrr:
                    if epoch >= args.n_epochs or epoch - best_epoch > 5:
                        break
                else:
                    best_mrr = mrr
                    best_epoch = epoch
                    torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, model_state_file)

        mrr = test(model, args.start_history_len, 
                        train_list+valid_list,
                        test_list, 
                        num_rels, 
                        num_nodes, 
                        use_cuda, 
                        model_state_file, neg_samples,
                        mode="test")
        
    elif trainvalidtest_id == 0: #curriculum training
        # load best model with start history length
        init_state_file = '../models/{}/'.format(args.dataset) + "{}-{}-ly{}-dilate{}-his{}-dp{}-gpu{}".format(args.encoder, args.decoder, args.n_layers, args.dilate_len, args.start_history_len, args.dropout, args.gpu)
        init_checkpoint = torch.load(init_state_file, map_location=torch.device(args.gpu))
        print("Load Previous Model name: {}. Using best epoch : {}".format(init_state_file, init_checkpoint['epoch']))  # use best stat checkpoint
        print("\n"+"-"*10+"Load model with history length {}".format(args.start_history_len)+"-"*10+"\n")
        model.load_state_dict(init_checkpoint['state_dict'])
        test_history_len = args.start_history_len
        mrr_raw, mrr, mrr_raw_r, mrr_r = test(model, 
                                                args.start_history_len,
                                                train_list+valid_list,
                                                test_list, 
                                                num_rels, 
                                                num_nodes, 
                                                use_cuda, 
                                                init_state_file,  neg_samples,
                                                mode="test")
        best_mrr_list = [mrr.item()]                                                   
        # start knowledge distillation
        ks_idx = 0
        for history_len in range(args.start_history_len+1, args.train_history_len+1, 1):
            # current model
            print("best mrr list :", best_mrr_list)
            # lr = 0.1*args.lr - 0.002*args.lr*ks_idx
            optimizer = torch.optim.Adam(model.parameters(), lr=0.1*args.lr, weight_decay=0.00001)
            model_name= f'{MODEL_NAME}_{DATA}_{SEED}_{args.run_nr}_{history_len}'
            # model_name = "{}-{}-ly{}-dilate{}-his{}-dp{}-gpu{}".format(args.encoder, args.decoder, args.n_layers, args.dilate_len, history_len, args.dropout, args.gpu)
            # model_state_file = '../models/{}/'.format(args.dataset) + model_name # TODO
            # print("Sanity Check: stat name : {}".format(model_state_file))# TODO

            # load model with the least history length
            prev_model_name= f'{MODEL_NAME}_{DATA}_{SEED}_{args.run_nr}_{history_len-1}'
            # prev_model_name = "{}-{}-ly{}-dilate{}-his{}-dp{}-gpu{}".format(args.encoder, args.decoder, args.n_layers, args.dilate_len, history_len-1, args.dropout, args.gpu)
            # prev_state_file = '../models/{}/'.format(args.dataset) + prev_model_name # TODO
            # checkpoint = torch.load(prev_state_file, map_location=torch.device(args.gpu)) #TODO
            # print("Load Previous Model name: {}. Using best epoch : {}".format(prev_model_name, checkpoint['epoch']))  # use best stat checkpoint
            model.load_state_dict(checkpoint['state_dict']) # TODO
            # prev_model = copy.deepcopy(model)
            # prev_model.eval()
            print("\n"+"-"*10+"start knowledge distillation for history length at "+ str(history_len)+"-"*10+"\n")
 
            best_mrr = 0
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

                # validation
                if epoch % args.evaluate_every == 0:
                    mrr = test(model,
                                history_len,
                                train_list, 
                                valid_list, 
                                num_rels, 
                                num_nodes, 
                                use_cuda, 
                                model_state_file,  neg_samples_valid,
                                mode="train")
                    
                
                    if mrr< best_mrr:
                        if epoch >= args.n_epochs or epoch-best_epoch>2:
                            break
                    else:
                        best_mrr = mrr
                        best_epoch = epoch
                        torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, model_state_file)  
            mrr = test(model, history_len,
                        train_list,
                        valid_list, 
                        num_rels, 
                        num_nodes, 
                        use_cuda, 
                        model_state_file,  neg_samples,
                        mode="test")
            ks_idx += 1
            if mrr.item() < max(best_mrr_list):
                test_history_len = history_len-1
                break
            else:
                best_mrr_list.append(mrr.item())

        mrr = test(model, test_history_len, 
                   train_list+valid_list,
                    test_list, 
                    num_rels, 
                    num_nodes, 
                    use_cuda, 
                    prev_state_file,  neg_samples,
                    mode="test")
    return mrr



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='REGCN')

    parser.add_argument("--gpu", type=int, default=-1,
                        help="gpu")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="batch-size")
    parser.add_argument("-d", "--dataset", type=str, required=True,
                        help="dataset to use")
  
    parser.add_argument("--run-statistic", action='store_true', default=False,
                        help="statistic the result")

    parser.add_argument("--relation-evaluation", action='store_true', default=False,
                        help="save model accordding to the relation evalution")

    
    # configuration for encoder RGCN stat
    parser.add_argument("--weight", type=float, default=1,
                        help="weight of static constraint")
    parser.add_argument("--task-weight", type=float, default=1,
                        help="weight of entity prediction task")
    parser.add_argument("--kl-weight", type=float, default=0.7,
                        help="weight of entity prediction task")
   
    parser.add_argument("--encoder", type=str, default="uvrgcn",
                        help="method of encoder")

    parser.add_argument("--dropout", type=float, default=0.2,
                        help="dropout probability")
    parser.add_argument("--skip-connect", action='store_true', default=False,
                        help="whether to use skip connect in a RGCN Unit")
    parser.add_argument("--n-hidden", type=int, default=200,
                        help="number of hidden units")
    parser.add_argument("--opn", type=str, default="sub",
                        help="opn of compgcn")

    parser.add_argument("--n-bases", type=int, default=100,
                        help="number of weight blocks for each relation")
    parser.add_argument("--n-basis", type=int, default=100,
                        help="number of basis vector for compgcn")
    parser.add_argument("--n-layers", type=int, default=2,
                        help="number of propagation rounds")
    parser.add_argument("--self-loop", action='store_true', default=True,
                        help="perform layer normalization in every layer of gcn ")
    parser.add_argument("--layer-norm", action='store_true', default=False,
                        help="perform layer normalization in every layer of gcn ")
    parser.add_argument("--relation-prediction", action='store_true', default=False,
                        help="add relation prediction loss")
    parser.add_argument("--entity-prediction", action='store_true', default=False,
                        help="add entity prediction loss")


    # configuration for stat training
    parser.add_argument("--n-epochs", type=int, default=500,
                        help="number of minimum training epochs on each time step")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate")
    parser.add_argument("--ft_epochs", type=int, default=30,
                        help="number of minimum fine-tuning epoch")
    parser.add_argument("--ft_lr", type=float, default=0.001,
                        help="learning rate")
    parser.add_argument("--norm_weight", type=float, default=0.1,
                        help="learning rate")
    parser.add_argument("--grad-norm", type=float, default=1.0,
                        help="norm to clip gradient to")

    # configuration for evaluating
    parser.add_argument("--evaluate-every", type=int, default=20,
                        help="perform evaluation every n epochs")

    # configuration for decoder
    parser.add_argument("--decoder", type=str, default="convtranse",
                        help="method of decoder")
    parser.add_argument("--input-dropout", type=float, default=0.2,
                        help="input dropout for decoder ")
    parser.add_argument("--hidden-dropout", type=float, default=0.2,
                        help="hidden dropout for decoder")
    parser.add_argument("--feat-dropout", type=float, default=0.2,
                        help="feat dropout for decoder")

    # configuration for sequences stat
    parser.add_argument("--train-history-len", type=int, default=10,
                        help="history length")
    parser.add_argument("--test-history-len", type=int, default=20,
                        help="history length for test")
    parser.add_argument("--start-history-len", type=int, default=1,
                    help="start history length")
    parser.add_argument("--dilate-len", type=int, default=1,
                        help="dilate history graph")

    # configuration for optimal parameters
    parser.add_argument("--grid-search", action='store_true', default=False,
                        help="perform grid search for best configuration")
    parser.add_argument("-tune", "--tune", type=str, default="n_hidden,n_layers,dropout,n_bases",
                        help="stat to use")
    parser.add_argument("--num-k", type=int, default=500,
                        help="number of triples generated")


    args = parser.parse_args()
    # TODO: add the code for hyperparameter tuning here
    
    run_experiment(args)
    sys.exit()



















def get_args():
    parser = argparse.ArgumentParser(description='CEN')

    parser.add_argument("--gpu", type=int, default=-1,
                        help="gpu")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="batch-size")
    parser.add_argument("-d", "--dataset", type=str, default='tkgl-yago',
                        help="dataset to use")
    parser.add_argument("--test", type=int, default=0,
                        help="1: formal test 2: continual test")
  
    parser.add_argument("--run-statistic", action='store_true', default=False,
                        help="statistic the result")

    parser.add_argument("--relation-evaluation", action='store_true', default=False,
                        help="save model accordding to the relation evalution")

    
    # configuration for encoder RGCN stat
    parser.add_argument("--weight", type=float, default=1,
                        help="weight of static constraint")
    parser.add_argument("--task-weight", type=float, default=1,
                        help="weight of entity prediction task")
    parser.add_argument("--kl-weight", type=float, default=0.7,
                        help="weight of entity prediction task")
   
    parser.add_argument("--encoder", type=str, default="uvrgcn",
                        help="method of encoder")

    parser.add_argument("--dropout", type=float, default=0.2,
                        help="dropout probability")
    parser.add_argument("--skip-connect", action='store_true', default=False,
                        help="whether to use skip connect in a RGCN Unit")
    parser.add_argument("--n-hidden", type=int, default=200,
                        help="number of hidden units")
    parser.add_argument("--opn", type=str, default="sub",
                        help="opn of compgcn")

    parser.add_argument("--n-bases", type=int, default=100,
                        help="number of weight blocks for each relation")
    parser.add_argument("--n-basis", type=int, default=100,
                        help="number of basis vector for compgcn")
    parser.add_argument("--n-layers", type=int, default=2,
                        help="number of propagation rounds")
    parser.add_argument("--self-loop", action='store_true', default=True,
                        help="perform layer normalization in every layer of gcn ")
    parser.add_argument("--layer-norm", action='store_true', default=False,
                        help="perform layer normalization in every layer of gcn ")
    parser.add_argument("--relation-prediction", action='store_true', default=False,
                        help="add relation prediction loss")
    parser.add_argument("--entity-prediction", action='store_true', default=False,
                        help="add entity prediction loss")


    # configuration for stat training
    parser.add_argument("--n-epochs", type=int, default=500,
                        help="number of minimum training epochs on each time step")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate")
    parser.add_argument("--ft_epochs", type=int, default=30,
                        help="number of minimum fine-tuning epoch")
    parser.add_argument("--ft_lr", type=float, default=0.001,
                        help="learning rate")
    parser.add_argument("--norm_weight", type=float, default=0.1,
                        help="learning rate")
    parser.add_argument("--grad-norm", type=float, default=1.0,
                        help="norm to clip gradient to")

    # configuration for evaluating
    parser.add_argument("--evaluate-every", type=int, default=20,
                        help="perform evaluation every n epochs")

    # configuration for decoder
    parser.add_argument("--decoder", type=str, default="convtranse",
                        help="method of decoder")
    parser.add_argument("--input-dropout", type=float, default=0.2,
                        help="input dropout for decoder ")
    parser.add_argument("--hidden-dropout", type=float, default=0.2,
                        help="hidden dropout for decoder")
    parser.add_argument("--feat-dropout", type=float, default=0.2,
                        help="feat dropout for decoder")

    # configuration for sequences stat
    parser.add_argument("--train-history-len", type=int, default=10,
                        help="history length")
    parser.add_argument("--test-history-len", type=int, default=20,
                        help="history length for test")
    parser.add_argument("--start-history-len", type=int, default=1,
                    help="start history length")
    parser.add_argument("--dilate-len", type=int, default=1,
                        help="dilate history graph")

    # configuration for optimal parameters
    parser.add_argument("--grid-search", action='store_true', default=False,
                        help="perform grid search for best configuration")
    parser.add_argument("-tune", "--tune", type=str, default="n_hidden,n_layers,dropout,n_bases",
                        help="stat to use")
    parser.add_argument("--num-k", type=int, default=500,
                        help="number of triples generated")
    parser.add_argument('--seed', type=int, help='Random seed', default=1)
    parser.add_argument('--run-nr', type=int, help='Run Number', default=1)

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
    return args, sys.argv 

# ==================
# ==================
# ==================

start_overall = timeit.default_timer()

# set hyperparameters
args, _ = get_args()
SEED = args.seed  # set the random seed for consistency
set_random_seed(SEED)

DATA=args.dataset
MODEL_NAME = 'CEN'

# load data
dataset = LinkPropPredDataset(name=DATA, root="datasets", preprocess=True)

num_rels = dataset.num_rels
num_nodes = dataset.num_nodes 
subjects = dataset.full_data["sources"]
objects= dataset.full_data["destinations"]
relations = dataset.edge_type

timestamps_orig = dataset.full_data["timestamps"]
timestamps = reformat_ts(timestamps_orig) # stepsize:1
all_quads = np.stack((subjects, relations, objects, timestamps), axis=1)

train_data = all_quads[dataset.train_mask]
val_data = all_quads[dataset.val_mask]
test_data = all_quads[dataset.test_mask]

train_list = split_by_time(train_data)
valid_list = split_by_time(val_data)
test_list = split_by_time(test_data)

# evaluation metric
metric = dataset.eval_metric
evaluator = Evaluator(name=DATA)
neg_sampler = dataset.negative_sampler

if args.grid_search:
    print("TODO: implement hyperparameter search")
# single run
else:
    # pretrain
    mrr = run_experiment(args, trainvalidtest_id=-1)
    # train
    mrr = run_experiment(args, trainvalidtest_id=0)
    # valid # only needed if hyperparameter tuning
    mrr = run_experiment(args, trainvalidtest_id=1)
    # test
    mrr = run_experiment(args, trainvalidtest_id=2)
sys.exit()
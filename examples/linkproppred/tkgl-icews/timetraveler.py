"""
TimeTraveler: Reinforcement Learning for Temporal Knowledge Graph Forecasting
Reference:
- https://github.com/JHL-HUST/TITer
Haohai Sun, Jialun Zhong, Yunpu Ma, Zhen Han, Kun He.
TimeTraveler: Reinforcement Learning for Temporal Knowledge Graph Forecasting EMNLP 2021
"""
import sys
import timeit

import torch
from torch.utils.data import Dataset,DataLoader
import logging

import numpy as np
import pickle
from tqdm import tqdm
import os.path as osp
from pathlib import Path
import os

modules_path = osp.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(modules_path)
from modules.timetraveler_agent import Agent
from modules.timetraveler_environment import Env
from modules.timetraveler_dirichlet import Dirichlet, MLE_Dirchlet
from modules.timetraveler_episode import Episode
from modules.timetraveler_policygradient import PG
from tgb.linkproppred.dataset import LinkPropPredDataset 
from tgb.linkproppred.evaluate import Evaluator
from modules.timetraveler_trainertester import Trainer, Tester, getRelEntCooccurrence
from tgb.utils.utils import set_random_seed,save_results 
from modules.tkg_utils import  get_args_timetraveler, reformat_ts, get_model_config_timetraveler

class QuadruplesDataset(Dataset):
    """ this is an internal way how Timetraveler represents the data
    """
    def __init__(self, examples):
        """
        examples: a list of quadruples.
        num_r: number of relations
        """
        self.quadruples = examples.copy()


    def __len__(self):
        return len(self.quadruples)

    def __getitem__(self, item):
        return self.quadruples[item][0], \
               self.quadruples[item][1], \
               self.quadruples[item][2], \
               self.quadruples[item][3], \
               self.quadruples[item][4]
    
def set_logger(save_path):
    """Write logs to checkpoint and console"""
    if args.do_train:
        log_file = os.path.join(save_path, 'train.log')
    else:
        log_file = os.path.join(save_path, 'test.log')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def preprocess_data(args, config, timestamps, save_path, all_quads):
    """
    Preprocess the data and save the state-action space (pickle dump)
    """
    # parser = argparse.ArgumentParser(description='Data preprocess', usage='preprocess_data.py [<args>] [-h | --help]')
    # parser.add_argument('--data_dir', default='data/ICEWS14', type=str, help='Path to data.')

    env = Env(all_quads, config)
    state_actions_space = {}

    with tqdm(total=len(all_quads)) as bar:
        for (head, rel, tail, t, _) in all_quads:
            if (head, t, True) not in state_actions_space.keys():
                state_actions_space[(head, t, True)] = env.get_state_actions_space_complete(head, t, True, args.store_actions_num)
                state_actions_space[(head, t, False)] = env.get_state_actions_space_complete(head, t, False, args.store_actions_num)
            if (tail, t, True) not in state_actions_space.keys():
                state_actions_space[(tail, t, True)] = env.get_state_actions_space_complete(tail, t, True, args.store_actions_num)
                state_actions_space[(tail, t, False)] = env.get_state_actions_space_complete(tail, t, False, args.store_actions_num)
            bar.update(1)
    pickle.dump(state_actions_space, open(os.path.join(save_path,  args.state_actions_path), 'wb'))

def log_metrics(mode, step, metrics):
    """Print the evaluation logs"""
    for metric in metrics:
        logging.info('%s %s at epoch %d: %f' % (mode, metric, step, metrics[metric]))

def main(args):
    """
    Main function to train and test the TimeTraveler model"""

    start_overall = timeit.default_timer()
    #######################Set Logger#################################
    
    save_path = f'{os.path.dirname(os.path.abspath(__file__))}/saved_models/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if args.cuda and torch.cuda.is_available():
        args.cuda = True
    else:
        args.cuda = False
    set_logger(save_path)

    #######################Create DataLoader#################################
    # set hyperparameters
    args.dataset = 'tkgl-icews'

    SEED = args.seed  # set the random seed for consistency
    set_random_seed(SEED)

    DATA=args.dataset
    MODEL_NAME = 'TIMETRAVELER'

    # load data
    dataset = LinkPropPredDataset(name=DATA, root="datasets", preprocess=True)

    num_rels = dataset.num_rels
    num_nodes = dataset.num_nodes 
    subjects = dataset.full_data["sources"]
    objects= dataset.full_data["destinations"]
    relations = dataset.edge_type

    timestamps_orig = dataset.full_data["timestamps"]
    timestamps = reformat_ts(timestamps_orig, DATA) # stepsize:1
    all_quads = np.stack((subjects, relations, objects, timestamps,timestamps_orig), axis=1)

    train_data = all_quads[dataset.train_mask]
    train_entities = np.unique(np.concatenate((train_data[:, 0], train_data[:, 2])))
    RelEntCooccurrence = getRelEntCooccurrence(train_data, num_rels)
    train_data =QuadruplesDataset(train_data)
    val_data = QuadruplesDataset(all_quads[dataset.val_mask])
    test_data = QuadruplesDataset(all_quads[dataset.test_mask])

    METRIC = dataset.eval_metric
    evaluator = Evaluator(name=DATA)
    neg_sampler = dataset.negative_sampler
    #load the ns samples 
    dataset.load_val_ns()
    dataset.load_test_ns()

    train_dataloader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    valid_dataloader = DataLoader(
        val_data,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    test_dataloader = DataLoader(
        test_data,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    ######################Creat the agent and the environment###########################
    config = get_model_config_timetraveler(args, num_nodes, num_rels)
    logging.info(config)
    logging.info(args)

    # creat the agent
    agent = Agent(config)


    # creat the environment
    state_actions_path = os.path.join(save_path, args.state_actions_path)


    ######################preprocessing###########################
    if not os.path.exists(state_actions_path):
        if args.preprocess:
            print("preprocessing data...")
            preprocess_data(args, config, timestamps, save_path, list(all_quads))
            state_action_space = pickle.load(open(os.path.join(save_path, args.state_actions_path), 'rb'))
        else:
            state_action_space = None
    else:
        print("load preprocessed data...")
        state_action_space = pickle.load(open(os.path.join(save_path, args.state_actions_path), 'rb'))


    env = Env(list(all_quads), config, state_action_space)
    # Create episode controller
    episode = Episode(env, agent, config)
    if args.cuda:
        episode = episode.cuda()
    pg = PG(config)  # Policy Gradient
    optimizer = torch.optim.Adam(episode.parameters(), lr=args.lr, weight_decay=0.00001)

    ######################Reward Shaping: MLE DIRICHLET alphas###########################
    if args.reward_shaping: 
        try:
            print("load alphas from pickle file")
            alphas = pickle.load(open(os.path.join(save_path, args.alphas_pkl), 'rb'))
        except:
            print('running MLE dirichlet now')
            mle_d = MLE_Dirchlet(all_quads, num_rels, args.k, args.time_span,
                         args.tol, args.method, args.maxiter)
            pickle.dump(mle_d.alphas, open(os.path.join(save_path, args.alphas_pkl), 'wb'))

            print('dumped alphas')
            alphas = mle_d.alphas
        distributions = Dirichlet(alphas, args.k)
    else:
        distributions = None

    ######################Training and Testing###########################

    trainer = Trainer(episode, pg, optimizer, args, distributions)
    tester = Tester(episode, args, train_entities, RelEntCooccurrence, dataset.metric)
    test_metrics ={}
    val_metrics = {}
    test_metrics[METRIC] = None
    val_metrics[METRIC] = None

    if args.do_train:
        start_train =timeit.default_timer()
        logging.info('Start Training......')
        for i in range(args.max_epochs):
            loss, reward = trainer.train_epoch(train_dataloader, len(train_data))
            logging.info('Epoch {}/{} Loss: {}, reward: {}'.format(i, args.max_epochs, loss, reward))

            #! checking GPU usage
            free_mem, total_mem = torch.cuda.mem_get_info()
            print ("--------------GPU memory usage-----------")
            print ("there are ", free_mem, " free memory")
            print ("there are ", total_mem, " total available memory")
            print ("there are ", total_mem - free_mem, " used memory")
            print ("--------------GPU memory usage-----------")
            
            if i % args.save_epoch == 0 and i != 0:
                trainer.save_model(save_path, 'checkpoint_{}.pth'.format(i))
                logging.info('Save Model in {}'.format(save_path))

            if i % args.valid_epoch == 0 and i != 0:
                logging.info('Start Val......')
                val_metrics = tester.test(valid_dataloader,
                                      len(val_data), num_nodes, neg_sampler, evaluator, split_mode='val')
                for mode in val_metrics.keys():
                    logging.info('{} at epoch {}: {}'.format(mode, i, val_metrics[mode]))

        trainer.save_model(save_path)
        logging.info('Save Model in {}'.format(save_path))
    else:
          # # Load the model parameters
        if os.path.isfile(save_path):
            params = torch.load(save_path)
            episode.load_state_dict(params['model_state_dict'])
            optimizer.load_state_dict(params['optimizer_state_dict'])
            logging.info('Load pretrain model: {}'.format(save_path))
    if args.do_test:
        logging.info('Start Testing......')
        start_test = timeit.default_timer()
        test_metrics = tester.test(test_dataloader,
                              len(test_data), num_nodes, neg_sampler, evaluator, split_mode='test')
        for mode in test_metrics.keys():
            logging.info('Test {} : {}'.format(mode, test_metrics[mode]))

        # saving the results...
        results_path = f'{osp.dirname(osp.abspath(__file__))}/saved_results'
        if not osp.exists(results_path):
            os.mkdir(results_path)
            print('INFO: Create directory {}'.format(results_path))
        Path(results_path).mkdir(parents=True, exist_ok=True)
        results_filename = f'{results_path}/{MODEL_NAME}_{DATA}_results.json'
        test_time = timeit.default_timer() - start_test
        all_time = timeit.default_timer() - start_train 
        all_time_preprocess = timeit.default_timer() - start_overall 

        save_results({'model': MODEL_NAME,
                    'data': DATA,
                    'seed': SEED,
                    f'val {METRIC}': float(val_metrics[METRIC]),
                    f'test {METRIC}': float(test_metrics[METRIC]),
                    'test_time': test_time,
                    'tot_train_val_time': all_time,
                    'tot_preprocess_train_val_time': all_time_preprocess
                    }, 
            results_filename)     

if __name__ == '__main__':
    args = get_args_timetraveler()
    main(args)
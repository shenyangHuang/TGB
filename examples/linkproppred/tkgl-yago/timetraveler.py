"""
TimeTraveler: Reinforcement Learning for Temporal Knowledge Graph Forecasting
Reference:
- https://github.com/JHL-HUST/TITer
Haohai Sun, Jialun Zhong, Yunpu Ma, Zhen Han, Kun He.
TimeTraveler: Reinforcement Learning for Temporal Knowledge Graph Forecasting EMNLP 2021
"""
import sys
sys.path.insert(0, '/home/mila/j/julia.gastinger/TGB2')
import timeit
from tgb.utils.utils import set_random_seed, get_args_timetraveler, save_results, reformat_ts, get_model_config_timetraveler
import os
import torch
import logging
from torch.utils.data import DataLoader
import numpy as np
import pickle

from tgb_modules.timetraveler_agent import Agent
from tgb_modules.timetraveler_environment import Env
from tgb_modules.timetraveler_dirichlet import Dirichlet
from tgb_modules.timetraveler_episode import Episode
from tgb_modules.timetraveler_policygradient import PG
from tgb.linkproppred.dataset import LinkPropPredDataset 
from tgb_modules.timetraveler_trainertester import Trainer, Tester, getRelEntCooccurrence
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

def log_metrics(mode, step, metrics):
    """Print the evaluation logs"""
    for metric in metrics:
        logging.info('%s %s at epoch %d: %f' % (mode, metric, step, metrics[metric]))

def main(args):

    # TODO: preprocessing and dirichlet param estimation steps! 
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
    args.dataset = 'tkgl-yago'

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
    timestamps = reformat_ts(timestamps_orig) # stepsize:1
    all_quads = np.stack((subjects, relations, objects, timestamps), axis=1)

    train_data = all_quads[dataset.train_mask]
    train_entities = np.unique(np.concatenate((train_data[:, 0], train_data[:, 2])))
    RelEntCooccurrence = getRelEntCooccurrence(train_data, num_rels)

    val_data = all_quads[dataset.val_mask]
    test_data = all_quads[dataset.test_mask]

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
    if not os.path.exists(state_actions_path):
        state_action_space = None
    else:
        state_action_space = pickle.load(open(os.path.join(save_path, args.state_actions_path), 'rb'))
    env = Env(list(all_quads), config, state_action_space)

    # Create episode controller
    episode = Episode(env, agent, config)
    if args.cuda:
        episode = episode.cuda()
    pg = PG(config)  # Policy Gradient
    optimizer = torch.optim.Adam(episode.parameters(), lr=args.lr, weight_decay=0.00001)

    # Load the model parameters
    if os.path.isfile(save_path):
        params = torch.load(save_path)
        episode.load_state_dict(params['model_state_dict'])
        optimizer.load_state_dict(params['optimizer_state_dict'])
        logging.info('Load pretrain model: {}'.format(save_path))

    ######################Training and Testing###########################
    if args.reward_shaping: #TODO
        alphas = pickle.load(open(os.path.join(save_path, args.alphas_pkl), 'rb'))
        distributions = Dirichlet(alphas, args.k)
    else:
        distributions = None
    trainer = Trainer(episode, pg, optimizer, args, distributions)
    tester = Tester(episode, args, train_entities, RelEntCooccurrence)
    if args.do_train:
        logging.info('Start Training......')
        for i in range(args.max_epochs):
            loss, reward = trainer.train_epoch(train_dataloader, trainDataset.__len__())
            logging.info('Epoch {}/{} Loss: {}, reward: {}'.format(i, args.max_epochs, loss, reward))

            if i % args.save_epoch == 0 and i != 0:
                trainer.save_model('checkpoint_{}.pth'.format(i))
                logging.info('Save Model in {}'.format(save_path))

            if i % args.valid_epoch == 0 and i != 0:
                logging.info('Start Val......')
                metrics = tester.test(valid_dataloader,
                                      validDataset.__len__(),
                                      baseData.skip_dict,
                                      config['num_ent'])
                for mode in metrics.keys():
                    logging.info('{} at epoch {}: {}'.format(mode, i, metrics[mode]))

        trainer.save_model()
        logging.info('Save Model in {}'.format(save_path))

    if args.do_test:
        logging.info('Start Testing......')
        metrics = tester.test(test_dataloader,
                              testDataset.__len__(),
                              baseData.skip_dict,
                              config['num_ent'])
        for mode in metrics.keys():
            logging.info('Test {} : {}'.format(mode, metrics[mode]))

if __name__ == '__main__':
    args = get_args_timetraveler()
    main(args)
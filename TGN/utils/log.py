"""
Set up loggger and file paths
"""

import logging
import os
import sys
from pathlib import Path
import time


def set_up_log_path(args, sys_argv):
    """
    Set up the logger, checkpoing, and model path
    """
    runtime_id = '{}_{}_TR_{}_LP'.format(args.prefix, args.data, args.tr_neg_sample)
    # logger
    logging.basicConfig(level=logging.INFO, filemode='w', force=True)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    Path("log/").mkdir(parents=True, exist_ok=True)
    log_file_path = 'log/{}.log'.format(runtime_id)
    fh = logging.FileHandler(log_file_path)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARN)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info(args)

    # checkpoint & model paths
    best_model_root = "./saved_models/"
    checkpoint_root = "./saved_checkpoints/"
    Path(best_model_root).mkdir(parents=True, exist_ok=True)
    Path(checkpoint_root).mkdir(parents=True, exist_ok=True)     

    checkpoint_dir = checkpoint_root + runtime_id + '/'
    best_model_dir = best_model_root + runtime_id + '/'
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(best_model_dir).mkdir(parents=True, exist_ok=True)

    # logger.info('Create checkpoint directory {}'.format(checkpoint_dir))
    # logger.info('Create best model directory {}'.format(best_model_dir))

    get_checkpoint_path = lambda run_idx, epoch: (checkpoint_dir + 'checkpoint-run-{}-epoch-{}.pth'.format(run_idx, epoch))
    get_best_model_path = lambda run_idx: (best_model_dir + '{}_{}.pth'.format(runtime_id, run_idx))

    return logger, get_checkpoint_path, get_best_model_path
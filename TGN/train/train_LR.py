"""
Train a TGN model for a Link Regression Task 

Date: 
    - Mar. 5, 2023
"""
import torch
import numpy as np
import pandas as pd
import math
import time
import logging
from evaluation.evaluation_LR import *



def train_val_LR(model, train_val_data, train_val_sampler, n_neigbors, partial_ngh_finder, full_ngh_finder, 
              USE_MEMORY, BATCH_SIZE, BACKPROP_EVERY,
              NUM_EPOCH, criterion, optimizer, early_stopper, logger, run_idx, device, get_checkpoint_path):
    """
    Training procedure for a TGN model
    """
    train_data, val_data, new_node_val_data = train_val_data
    train_rand_sampler, val_rand_sampler, nn_val_rand_sampler = train_val_sampler

    num_instance = len(train_data.sources)
    num_batch = math.ceil(num_instance / BATCH_SIZE)

    logger.info('Number of training instances: {}'.format(num_instance))
    logger.info('Number of batches per epoch: {}'.format(num_batch))


    train_losses = []
    # idx_list = np.arange(num_instance)  # NOTE: does not work correctly with TGN because of the memory module

    for epoch in range(NUM_EPOCH):
        start_epoch = time.time()
        logger.info('Start epoch {}'.format(epoch))

        m_loss = []  # store the loss of each epoch
        # np.random.shuffle(idx_list)  # it does not work with TGN that has memory and process edges in time order

        # =============== Training
        # Reinitialize memory of the model at the start of each epoch
        if USE_MEMORY:
            model.memory.__init_memory__()

        model.set_neighbor_finder(partial_ngh_finder)

        for k in range(0, num_batch, BACKPROP_EVERY):
            loss = 0
            optimizer.zero_grad()

            # Custom loop to allow to perform backpropagation only every a certain number of batches
            for j in range(BACKPROP_EVERY):
                batch_idx = k + j
                if batch_idx >= num_batch:
                    continue
                
                # positive edges
                start_idx = batch_idx * BATCH_SIZE
                end_idx = min(num_instance, start_idx + BATCH_SIZE)
                # batch_idx_list = idx_list[start_idx: end_idx]
                sources_batch = train_data.sources[start_idx: end_idx]  #[batch_idx_list]
                destinations_batch = train_data.destinations[start_idx: end_idx]  # [batch_idx_list]
                edge_idxs_batch = train_data.edge_idxs[start_idx: end_idx]  # [batch_idx_list]
                timestamps_batch = train_data.timestamps[start_idx: end_idx]  # [batch_idx_list]
                pos_edge_weights_batch = np.array(train_data.true_y[start_idx: end_idx]).reshape((len(sources_batch), 1))  # the target of predictions
                pos_edge_weights_batch = torch.from_numpy(pos_edge_weights_batch).to(device=device).float()

                # negative edges
                size = len(sources_batch)
                neg_hist_sources, neg_hist_destinations, neg_rnd_sources, neg_rnd_destinations = train_rand_sampler.sample(
                                                                                                                size,
                                                                                                                timestamps_batch[0],
                                                                                                                timestamps_batch[-1])
                negative_samples_sources = np.concatenate([neg_hist_sources, neg_rnd_sources], axis=0)
                negative_samples_destinations = np.concatenate([neg_hist_destinations, neg_rnd_destinations], axis=0)
                if train_rand_sampler.neg_sample == 'haphaz_rnd':
                    negative_samples_sources = sources_batch
                    logger.info("DEBUG: 'haphaz_rnd' is not supposed to be used!!!")
                
                # define true labels
                with torch.no_grad():
                    neg_edge_weights = torch.zeros((size, 1), dtype=torch.float, device=device)
                
                # compute edge probabilities
                model = model.train()
                pos_e = True
                pos_pred_score = model.compute_edge_weights(sources_batch, destinations_batch,
                                                    timestamps_batch, edge_idxs_batch, pos_e, n_neigbors)

                pos_e = False
                neg_pred_score = model.compute_edge_weights(negative_samples_sources,
                                                    negative_samples_destinations,
                                                    timestamps_batch, edge_idxs_batch, pos_e, n_neigbors)
                # compute & backprop the loss
                loss += criterion(pos_pred_score.squeeze(), pos_edge_weights_batch.squeeze()) \
                        + criterion(neg_pred_score.squeeze(), neg_edge_weights.squeeze())
            
            loss /= BACKPROP_EVERY
            loss.backward()
            optimizer.step()
            m_loss.append(loss.item())

            # Detach memory after 'args.backprop_every' number of batches so we don't backpropagate to
            # the start of time
            if USE_MEMORY:
                model.memory.detach_memory()

        # =============== Validation
        # NOTE: TGN: validation uses the full neighbor finder (TGAT uses partial neighbor finder for the validation)
        model.set_neighbor_finder(full_ngh_finder)

        if USE_MEMORY:
            # backup memory at the end of training so that we can restrore the memory later and use if for validation new nodes
            train_memory_backup = model.memory.backup_memory()
        
        # ========== Transductive 
        logger.info("INFO: Validation: Transductive")
        val_measures_dict = eval_link_reg_rnd_neg(model=model, sampler=val_rand_sampler, data=val_data, logger=logger, 
                                            batch_size=BATCH_SIZE, n_neighbors=n_neigbors)
        for metric_name, metric_value in val_measures_dict.items():
            logger.info('INFO: Validation statistics: Old nodes -- {}: {}'.format(metric_name, metric_value))

        if USE_MEMORY:
            # bakcup the memory after validation so that we can use it for evaluationg on test set
            val_memory_backup = model.memory.backup_memory()
            # restore the memory that we had at the end of training for evaluation of validation new nodes
            model.memory.restore_memory(train_memory_backup)

        # # ========== Inductive: validation on new nodes
        # nn_val_measures_dict = eval_link_pred(model=model, sampler=nn_val_rand_sampler, data=new_node_val_data, logger=logger,
        #                                       batch_size=BATCH_SIZE, n_neighbors=n_neigbors)
        # for metric_name, metric_value in nn_val_measures_dict.items():
        #     logger.info('INFO: Validation statistics: New nodes -- {}: {}'.format(metric_name, metric_value))

        if USE_MEMORY:
            model.memory.restore_memory(val_memory_backup)

        train_losses.append(np.mean(m_loss))

        total_epoch_time = time.time() - start_epoch

        # Early stopping
        if early_stopper.early_stop_check(val_measures_dict['MSE']):
            logger.info('No improvement over {} epochs, stop training'.format(early_stopper.max_round))
            logger.info('Loading the best model at epoch {}'.format(early_stopper.best_epoch))
            best_model_path = get_checkpoint_path(run_idx ,early_stopper.best_epoch)
            model.load_state_dict(torch.load(best_model_path))
            logger.info('Loaded the best model at epoch {} for inference.'.format(early_stopper.best_epoch))
            model.eval()
            break
        else:
            torch.save(model.state_dict(), get_checkpoint_path(run_idx, epoch))

        logger.info('Epoch {} mean loss: {}'.format(epoch, np.mean(m_loss)))
        logger.info('Epoch {} took {:.2f} seconds.'.format(epoch, total_epoch_time))
        



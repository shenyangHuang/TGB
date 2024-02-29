import ray
class RecurrencyBaselinePredictor(object):
    def __init__():
        """ init recurrency baseline predictor
        """
        a =0


    @ray.remote
    def apply_baselines_remote(self, i, num_queries, test_data, all_data, window, basis_dict, score_func_psi, num_nodes, 
                    num_rels, baselinexi_flag, baselinepsi_flag, 
                    lmbda_psi, alpha):
        return self.apply_baselines(i, num_queries, test_data, all_data, window, basis_dict, score_func_psi, num_nodes, 
                    num_rels, baselinexi_flag, baselinepsi_flag, 
                    lmbda_psi, alpha)
    


    def apply_baselines(self, i, num_queries, test_data, all_data, window, basis_dict, score_func_psi, num_nodes, 
                    num_rels, baselinexi_flag, baselinepsi_flag, 
                    lmbda_psi, alpha):
        """
        Apply baselines psi and xi (multiprocessing possible).

        Parameters:
            i (int): process number
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
            baselinexi_flag (boolean): True: use baselinexi, False: do not use baselinexi
            baselinepsi_flag (boolean): True: use baselinepsi, False: do not use baselinepsi
            lambda_psi (float): parameter for time decay function for baselinepsi. 0: no decay, >1 very steep decay
            alpha (float): parameter, weight to combine the scores from psi and xi. alpha*scores_psi + (1-alpha)*scores_xi
        Returns:
            logging_dict (dict): dict with one entry per test query (one per direction) key: string that desribes the query, 
            with xxx before the requested node, 
            values: list with two entries: [[tensor with one score per node], [np array with query_quadruple]]
            example: '14_0_xxx1_336': [tensor([1.8019e+01, ...9592e-05]), array([ 14,   0,   1...ype=int32)]
            scores_dict_eval (dict): dict  with one entry per test query (one per direction) key: str(test_qery), value: 
            tensor with scores, one score per node. example: [14, 0, 1, 336]':tensor([1.8019e+01,5.1101e+02,..., 0.0000e+0])
        """
        try:
            # print("Start process", i, "...")
            num_test_queries = len(test_data) - (i + 1) * num_queries
            if num_test_queries >= num_queries:
                test_queries_idx = range(i * num_queries, (i + 1) * num_queries)
            else:
                test_queries_idx = range(i * num_queries, len(test_data))

            cur_ts = test_data[test_queries_idx[0]][3]
            first_test_query_ts = test_data[0][3]
            edges, all_data_ts = utils.get_window_edges(all_data, cur_ts, window, first_test_query_ts) # get for the current 
                                    # timestep all previous quadruples per relation that fullfill time constraints
            
            if baselinexi_flag:
                obj_dist = {}
                rel_obj_dist = {}
                rel_obj_dist_cur_ts, obj_dist_cur_ts = baselinexi.update_distributions(all_data_ts, edges, obj_dist, rel_obj_dist,num_rels, cur_ts)
            if baselinepsi_flag:
                if len(all_data_ts) >0:
                    sum_delta_t = baselinepsi.update_delta_t(np.min(all_data_ts[:,3]), np.max(all_data_ts[:,3]), cur_ts, lmbda_psi)
                    sum_delta_t = np.max([sum_delta_t, 1e-15]) # to avoid division by zero

            it_start = time.time()
            logging_dict = {} # for logging
            scores_dict_eval = {}
            rel_ob_dist_scores = torch.zeros(num_nodes)
            ob_dist_scores =torch.zeros(num_nodes)
            predictions_xi=torch.zeros(num_nodes) 
            predictions_psi=torch.zeros(num_nodes)

            for j in test_queries_idx:       
                test_query = test_data[j]
                cands_dict = dict() 
                cands_dict_psi = dict() 
                # 1) update timestep and known triples
                if test_query[3] != cur_ts: # if we have a new timestep
                    cur_ts = test_query[3]
                    edges, all_data_ts = utils.get_window_edges(all_data, cur_ts, window, first_test_query_ts) # get for the current timestep all previous quadruples per relation that fullfill time constraints
                    if baselinexi_flag: # update the object and rel-object distritbutions to take into account what timesteps to use
                        if window > -1: #otherwise: multistep, we do not need to update
                            rel_obj_dist_cur_ts, obj_dist_cur_ts = baselinexi.update_distributions(all_data_ts, edges, obj_dist_cur_ts, rel_obj_dist_cur_ts, num_rels, cur_ts)
                    if baselinepsi_flag:
                        if len(all_data_ts) >0:
                            if window > -1: #otherwise: multistep, we do not need to update
                                sum_delta_t = baselinepsi.update_delta_t(np.min(all_data_ts[:,3]), np.max(all_data_ts[:,3]), cur_ts, lmbda_psi)
                                
                #### BASELINE  PSI
                # 2) apply rules for relation of interest, if we have any
                if baselinepsi_flag: 
                    if test_query[1] in basis_dict: # do we have rules for the given relation?
                        for rule in basis_dict[test_query[1]]:  # check all the rules that we have                    
                            walk_edges = utils.match_body_relations(rule, edges, test_query[0]) 
                                                # Find quadruples that match the rule (starting from the test query subject)
                                                # Find edges whose subject match the query subject and the relation matches
                                                # the relation in the rule body. np array with [[sub, obj, ts]]
                            if 0 not in [len(x) for x in walk_edges]: # if we found at least one potential rule
                                if baselinepsi_flag:
                                    cands_dict_psi = baselinepsi.get_candidates_psi(rule, walk_edges[0][:,1:3], cur_ts, cands_dict, score_func_psi, lmbda_psi, sum_delta_t)
                                    if len(cands_dict_psi)>0:                
                                        predictions_psi = logging_utils.create_scores_tensor(cands_dict_psi, num_nodes)

                #### BASELINE XI
                if baselinexi_flag:  # obj_dist, rel_obj_dist            
                    rel_ob_dist_scores = logging_utils.create_scores_tensor(rel_obj_dist_cur_ts[test_query[1]], num_nodes)
                    predictions_xi = rel_ob_dist_scores

                # logging the scores in a format that is similar to other methods. needs a lot of memory.
                currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
                sys.path.insert(1, currentdir) 
                sys.path.insert(1, os.path.join(sys.path[0], '../..'))        
                query_name, gt_test_query_ids = logging_utils.query_name_from_quadruple(test_query, num_rels)
                
                predictions_all = 1000*alpha*predictions_psi + 1000*(1-alpha)*predictions_xi 
                logging_dict[query_name] = [predictions_all, gt_test_query_ids]       
                scores_dict_eval[str(list(gt_test_query_ids))] = predictions_all

        except Exception as error:
        # handle the exception
            print("An exception occurred:", error) # An exception occurred: division by zero
            # print progress
            # if not (j - test_queries_idx[0] + 1) % 100:
            #     it_end = time.time()
            #     it_time = round(it_end - it_start, 6)
            #     # print("Process {0}: test samples finished: {1}/{2}, {3} sec".format(
            #             # i, j - test_queries_idx[0] + 1, len(test_queries_idx), it_time
            #         # ))
            #     it_start = time.time()
            logging_dict = {}
            scores_dict_eval = {}

        return logging_dict, scores_dict_eval
    

    def match_body_relations(self, rule, edges, test_query_sub):
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


    def get_window_edges(all_data, test_query_ts, window=-2, first_test_query_ts=0): #modified eval_paper_authors: added first_test_query_ts for validation set usage
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


    
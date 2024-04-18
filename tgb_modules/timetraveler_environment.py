"""
TimeTraveler: Reinforcement Learning for Temporal Knowledge Graph Forecasting
Reference:
- https://github.com/JHL-HUST/TITer/blob/master/model/environment.py
Haohai Sun, Jialun Zhong, Yunpu Ma, Zhen Han, Kun He.
TimeTraveler: Reinforcement Learning for Temporal Knowledge Graph Forecasting EMNLP 2021
"""

import networkx as nx
from collections import defaultdict
import numpy as np
import torch

class Env(object):
    def __init__(self, examples, config, state_action_space=None):
        """Temporal Knowledge Graph Environment.
        examples: quadruples (subject, relation, object, timestamps);
        config: config dict;
        state_action_space: Pre-processed action space;
        """
        self.config = config
        self.num_rel = config['num_rel']
        self.graph, self.label2nodes = self.build_graph(examples)
        # [0, num_rel) -> normal relations; num_rel -> stay in place，(num_rel, num_rel * 2] reversed relations.
        self.NO_OP = self.num_rel  # Stay in place; No Operation
        self.ePAD = config['num_ent']  # Padding entity
        self.rPAD = config['num_rel'] * 2 # + 1  # Padding relation.
        self.tPAD = 0  # Padding time
        self.state_action_space = state_action_space  # Pre-processed action space
        if state_action_space:
            self.state_action_space_key = self.state_action_space.keys()

    def build_graph(self, examples):
        """The graph node is represented as (entity, time), and the edges are directed and labeled relation.
        return:
            graph: nx.MultiDiGraph;
            label2nodes: a dict [keys -> entities, value-> nodes in the graph (entity, time)]
        """
        graph = nx.MultiDiGraph()
        label2nodes = defaultdict(set)
        examples.sort(key=lambda x: x[3], reverse=True)  # Reverse chronological order
        for example in examples:
            src = example[0]
            rel = example[1]
            dst = example[2]
            time = example[3]

            # Add the nodes and edges of the current quadruple
            src_node = (src, time)
            dst_node = (dst, time)
            if src_node not in label2nodes[src]:
                graph.add_node(src_node, label=src)
            if dst_node not in label2nodes[dst]:
                graph.add_node(dst_node, label=dst)

            graph.add_edge(src_node, dst_node, relation=rel)
            # graph.add_edge(dst_node, src_node, relation=rel+self.num_rel+1) #REMOVED by JULIA 

            label2nodes[src].add(src_node)
            label2nodes[dst].add(dst_node)
        return graph, label2nodes

    def get_state_actions_space_complete(self, entity, time, current_=False, max_action_num=None):
        """Get the action space of the current state.
        Args:
            entity: The entity of the current state;
            time: Maximum timestamp for candidate actions;
            current_: Can the current time of the event be used;
            max_action_num: Maximum number of events stored;
        Return:
            numpy array，shape: [number of events，3], (relation, dst, time)
        """
        if self.state_action_space:
            if (entity, time, current_) in self.state_action_space_key:
                return self.state_action_space[(entity, time, current_)]
        nodes = self.label2nodes[entity].copy()
        if current_:
            # Delete future events, you can see current events, before query time
            nodes = list(filter((lambda x: x[1] <= time), nodes))
        else:
            # No future events, no current events
            nodes = list(filter((lambda x: x[1] < time), nodes))
        nodes.sort(key=lambda x: x[1], reverse=True)
        actions_space = []
        i = 0
        for node in nodes:
            for src, dst, rel in self.graph.out_edges(node, data=True):
                actions_space.append((rel['relation'], dst[0], dst[1]))
                i += 1
                if max_action_num and i >= max_action_num:
                    break
            if max_action_num and i >= max_action_num:
                break
        return np.array(list(actions_space), dtype=np.dtype('int32'))

    def next_actions(self, entites, times, query_times, max_action_num=200, first_step=False):
        """Get the current action space. There must be an action that stays at the current position in the action space.
        Args:
            entites: torch.tensor, shape: [batch_size], the entity where the agent is currently located;
            times: torch.tensor, shape: [batch_size], the timestamp of the current entity;
            query_times: torch.tensor, shape: [batch_size], the timestamp of query;
            max_action_num: The size of the action space;
            first_step: Is it the first step for the agent.
        Return: torch.tensor, shape: [batch_size, max_action_num, 3], (relation, entity, time)
        """
        if self.config['cuda']:
            entites = entites.cpu()
            times = times.cpu()
            query_times = times.cpu()

        entites = entites.numpy()
        times = times.numpy()
        query_times = query_times.numpy()

        actions = self.get_padd_actions(entites, times, query_times, max_action_num, first_step)

        if self.config['cuda']:
            actions = torch.tensor(actions, dtype=torch.long, device='cuda')
        else:
            actions = torch.tensor(actions, dtype=torch.long)
        return actions

    def get_padd_actions(self, entites, times, query_times, max_action_num=200, first_step=False):
        """Construct the model input array.
        If the optional actions are greater than the maximum number of actions, then sample,
        otherwise all are selected, and the insufficient part is pad.
        """
        actions = np.ones((entites.shape[0], max_action_num, 3), dtype=np.dtype('int32'))
        actions[:, :, 0] *= self.rPAD
        actions[:, :, 1] *= self.ePAD
        actions[:, :, 2] *= self.tPAD
        for i in range(entites.shape[0]):
            # NO OPERATION
            actions[i, 0, 0] = self.NO_OP
            actions[i, 0, 1] = entites[i]
            actions[i, 0, 2] = times[i]

            if times[i] == query_times[i]:
                action_array = self.get_state_actions_space_complete(entites[i], times[i], False)
            else:
                action_array = self.get_state_actions_space_complete(entites[i], times[i], True)

            if action_array.shape[0] == 0:
                continue

            # Whether to keep the action NO_OPERATION
            start_idx = 1
            if first_step:
                # The first step cannot stay in place
                start_idx = 0

            if action_array.shape[0] > (max_action_num - start_idx):
                # Sample. Take the latest events.
                actions[i, start_idx:, ] = action_array[:max_action_num-start_idx]
            else:
                actions[i, start_idx:action_array.shape[0]+start_idx, ] = action_array
        return actions
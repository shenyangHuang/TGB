"""
TimeTraveler: Reinforcement Learning for Temporal Knowledge Graph Forecasting
Reference:
- https://github.com/JHL-HUST/TITer/blob/master/model/agent.py
Haohai Sun, Jialun Zhong, Yunpu Ma, Zhen Han, Kun He.
TimeTraveler: Reinforcement Learning for Temporal Knowledge Graph Forecasting EMNLP 2021
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class HistoryEncoder(nn.Module):
    def __init__(self, config):
        super(HistoryEncoder, self).__init__()
        self.config = config
        self.lstm_cell = torch.nn.LSTMCell(input_size=config['action_dim'],
                                           hidden_size=config['state_dim'])

    def set_hiddenx(self, batch_size):
        """Set hidden layer parameters. Initialize to 0"""
        if self.config['cuda']:
            self.hx = torch.zeros(batch_size, self.config['state_dim'], device='cuda')
            self.cx = torch.zeros(batch_size, self.config['state_dim'], device='cuda')
        else:
            self.hx = torch.zeros(batch_size, self.config['state_dim'])
            self.cx = torch.zeros(batch_size, self.config['state_dim'])

    def forward(self, prev_action, mask):
        """mask: True if NO_OP. ON_OP does not affect history coding results"""
        self.hx_, self.cx_ = self.lstm_cell(prev_action, (self.hx, self.cx))
        self.hx = torch.where(mask, self.hx, self.hx_)
        self.cx = torch.where(mask, self.cx, self.cx_)
        return self.hx

class PolicyMLP(nn.Module):
    def __init__(self, config):
        super(PolicyMLP, self).__init__()
        self.mlp_l1= nn.Linear(config['mlp_input_dim'], config['mlp_hidden_dim'], bias=True)
        self.mlp_l2 = nn.Linear(config['mlp_hidden_dim'], config['action_dim'], bias=True)

    def forward(self, state_query):
        hidden = torch.relu(self.mlp_l1(state_query))
        output = self.mlp_l2(hidden).unsqueeze(1)
        return output

class DynamicEmbedding(nn.Module):
    def __init__(self, n_ent, dim_ent, dim_t):
        super(DynamicEmbedding, self).__init__()
        self.ent_embs = nn.Embedding(n_ent, dim_ent - dim_t)
        self.w = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, dim_t))).float())
        self.b = torch.nn.Parameter(torch.zeros(dim_t).float())

    def forward(self, entities, dt):
        dt = dt.unsqueeze(-1)
        batch_size = dt.size(0)
        seq_len = dt.size(1)

        dt = dt.view(batch_size, seq_len, 1)
        t = torch.cos(self.w.view(1, 1, -1) * dt + self.b.view(1, 1, -1))
        t = t.squeeze(1)  # [batch_size, time_dim]

        e = self.ent_embs(entities)
        return torch.cat((e, t), -1)

class StaticEmbedding(nn.Module):
    def __init__(self, n_ent, dim_ent):
        super(StaticEmbedding, self).__init__()
        self.ent_embs = nn.Embedding(n_ent, dim_ent)

    def forward(self, entities, timestamps=None):
        return self.ent_embs(entities)

class Agent(nn.Module):
    def __init__(self, config):
        super(Agent, self).__init__()
        self.num_rel = config['num_rel'] * 2 + 2
        self.config = config

        # [0, num_rel) -> normal relations; num_rel -> stay in place，(num_rel, num_rel * 2] reversed relations.
        self.NO_OP = self.num_rel  # Stay in place; No Operation
        self.ePAD = config['num_ent']  # Padding entity
        self.rPAD = config['num_rel'] * 2 + 1  # Padding relation
        self.tPAD = 0  # Padding time

        if self.config['entities_embeds_method'] == 'dynamic':
            self.ent_embs = DynamicEmbedding(config['num_ent']+1, config['ent_dim'], config['time_dim'])
        else:
            self.ent_embs = StaticEmbedding(config['num_ent']+1, config['ent_dim'])

        self.rel_embs = nn.Embedding(config['num_ent'], config['rel_dim'])

        self.policy_step = HistoryEncoder(config)
        self.policy_mlp = PolicyMLP(config)

        self.score_weighted_fc = nn.Linear(
            self.config['ent_dim'] * 2 + self.config['rel_dim'] * 2 + self.config['state_dim'],
            1, bias=True)

    def forward(self, prev_relation, current_entities, current_timestamps,
                query_relation, query_entity, query_timestamps, action_space):
        """
        Args:
            prev_relation: [batch_size]
            current_entities: [batch_size]
            current_timestamps: [batch_size]
            query_relation: embeddings of query relation，[batch_size, rel_dim]
            query_entity: embeddings of query entity, [batch_size, ent_dim]
            query_timestamps: [batch_size]
            action_space: [batch_size, max_actions_num, 3] (relations, entities, timestamps)
        """
        # embeddings
        current_delta_time = query_timestamps - current_timestamps
        current_embds = self.ent_embs(current_entities, current_delta_time)  # [batch_size, ent_dim] #dynamic embedding
        prev_relation_embds = self.rel_embs(prev_relation)  # [batch_size, rel_dim]

        # Pad Mask
        pad_mask = torch.ones_like(action_space[:, :, 0]) * self.rPAD  # [batch_size, action_number]
        pad_mask = torch.eq(action_space[:, :, 0], pad_mask)  # [batch_size, action_number]

        # History Encode
        NO_OP_mask = torch.eq(prev_relation, torch.ones_like(prev_relation) * self.NO_OP)  # [batch_size]
        NO_OP_mask = NO_OP_mask.repeat(self.config['state_dim'], 1).transpose(1, 0)  # [batch_size, state_dim]
        prev_action_embedding = torch.cat([prev_relation_embds, current_embds], dim=-1)  # [batch_size, rel_dim + ent_dim]
        lstm_output = self.policy_step(prev_action_embedding, NO_OP_mask)  # [batch_size, state_dim] (5) Path encoding

        # Neighbor/condidate_actions embeddings
        action_num = action_space.size(1)
        neighbors_delta_time = query_timestamps.unsqueeze(-1).repeat(1, action_num) - action_space[:, :, 2]
        neighbors_entities = self.ent_embs(action_space[:, :, 1], neighbors_delta_time)  # [batch_size, action_num, ent_dim]
        neighbors_relations = self.rel_embs(action_space[:, :, 0])  # [batch_size, action_num, rel_dim]

        # agent state representation
        agent_state = torch.cat([lstm_output, query_entity, query_relation], dim=-1)  # [batch_size, state_dim + ent_dim + rel_dim]
        output = self.policy_mlp(agent_state)  # [batch_size, 1, action_dim] action_dim == rel_dim + ent_dim

        # scoring
        entitis_output = output[:, :, self.config['rel_dim']:]
        relation_ouput = output[:, :, :self.config['rel_dim']]
        relation_score = torch.sum(torch.mul(neighbors_relations, relation_ouput), dim=2)
        entities_score = torch.sum(torch.mul(neighbors_entities, entitis_output), dim=2)  # [batch_size, action_number]

        actions = torch.cat([neighbors_relations, neighbors_entities], dim=-1)  # [batch_size, action_number, action_dim]

        agent_state_repeats = agent_state.unsqueeze(1).repeat(1, actions.shape[1], 1)
        score_attention_input = torch.cat([actions, agent_state_repeats], dim=-1)
        a = self.score_weighted_fc(score_attention_input)                                   # (8)
        a = torch.sigmoid(a).squeeze()                      # [batch_size, action_number]   # (8)

        scores = (1 - a) * relation_score + a * entities_score                              # (6) a= beta

        # Padding mask
        scores = scores.masked_fill(pad_mask, -1e10)  # [batch_size ,action_number]

        action_prob = torch.softmax(scores, dim=1)
        action_id = torch.multinomial(action_prob, 1)  # Randomly select an action. [batch_size, 1] # ACTION SELECTION

        logits = torch.nn.functional.log_softmax(scores, dim=1)  # [batch_size, action_number]
        one_hot = torch.zeros_like(logits).scatter(1, action_id, 1)
        loss = - torch.sum(torch.mul(logits, one_hot), dim=1)
        return loss, logits, action_id

    def get_im_embedding(self, cooccurrence_entities):
        """Get the inductive mean representation of the co-occurrence relation.
        cooccurrence_entities: a list that contains the trained entities with the co-occurrence relation.
        return: torch.tensor, representation of the co-occurrence entities.
        """
        entities = self.ent_embs.ent_embs.weight.data[cooccurrence_entities]
        im = torch.mean(entities, dim=0)
        return im

    def update_entity_embedding(self, entity, ims, mu):
        """Update the entity representation with the co-occurrence relations in the last timestamp.
        entity: int, the entity that needs to be updated.
        ims: torch.tensor, [number of co-occurrence, -1], the IM representations of the co-occurrence relations
        mu: update ratio, the hyperparam.
        """
        self.source_entity = self.ent_embs.ent_embs.weight.data[entity]
        self.ent_embs.ent_embs.weight.data[entity] = mu * self.source_entity + (1 - mu) * torch.mean(ims, dim=0)

    def entities_embedding_shift(self, entity, im, mu):
        """Prediction shift."""
        self.source_entity = self.ent_embs.ent_embs.weight.data[entity]
        self.ent_embs.ent_embs.weight.data[entity] = mu * self.source_entity + (1 - mu) * im

    def back_entities_embedding(self, entity):
        """Go back after shift ends."""
        self.ent_embs.ent_embs.weight.data[entity] = self.source_entity
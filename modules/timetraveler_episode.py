"""
TimeTraveler: Reinforcement Learning for Temporal Knowledge Graph Forecasting
Reference:
- https://github.com/JHL-HUST/TITer/blob/master/model/episode.py
Haohai Sun, Jialun Zhong, Yunpu Ma, Zhen Han, Kun He.
TimeTraveler: Reinforcement Learning for Temporal Knowledge Graph Forecasting EMNLP 2021
"""

import torch
import torch.nn as nn

class Episode(nn.Module):
    def __init__(self, env, agent, config):
        super(Episode, self).__init__()
        self.config = config
        self.env = env
        self.agent = agent
        self.path_length = config['path_length']
        self.num_rel = config['num_rel']
        self.max_action_num = config['max_action_num']

    def forward(self, query_entities, query_timestamps, query_relations):
        """
        Args:
            query_entities: [batch_size]
            query_timestamps: [batch_size]
            query_relations: [batch_size]
        Return:
            all_loss: list
            all_logits: list
            all_actions_idx: list
            current_entities: torch.tensor, [batch_size]
            current_timestamps: torch.tensor, [batch_size]
        """
        query_entities_embeds = self.agent.ent_embs(query_entities, torch.zeros_like(query_timestamps))
        query_relations_embeds = self.agent.rel_embs(query_relations)

        current_entites = query_entities
        current_timestamps = query_timestamps
        prev_relations = torch.ones_like(query_relations) * self.num_rel  # NO_OP

        all_loss = []
        all_logits = []
        all_actions_idx = []

        self.agent.policy_step.set_hiddenx(query_relations.shape[0])
        for t in range(self.path_length):
            if t == 0:
                first_step = True
            else:
                first_step = False

            action_space = self.env.next_actions(
                current_entites,
                current_timestamps,
                query_timestamps,
                self.max_action_num,
                first_step
            )

            loss, logits, action_id = self.agent(
                prev_relations,
                current_entites,
                current_timestamps,
                query_relations_embeds,
                query_entities_embeds,
                query_timestamps,
                action_space,
            )

            chosen_relation = torch.gather(action_space[:, :, 0], dim=1, index=action_id).reshape(action_space.shape[0])
            chosen_entity = torch.gather(action_space[:, :, 1], dim=1, index=action_id).reshape(action_space.shape[0])
            chosen_entity_timestamps = torch.gather(action_space[:, :, 2], dim=1, index=action_id).reshape(action_space.shape[0])

            all_loss.append(loss)
            all_logits.append(logits)
            all_actions_idx.append(action_id)

            current_entites = chosen_entity
            current_timestamps = chosen_entity_timestamps
            prev_relations = chosen_relation

        return all_loss, all_logits, all_actions_idx, current_entites, current_timestamps

    def beam_search(self, query_entities, query_timestamps, query_relations):
        """
        Args:
            query_entities: [batch_size]
            query_timestamps: [batch_size]
            query_relations: [batch_size]
        Return:
            current_entites: [batch_size, test_rollouts_num]
            beam_prob: [batch_size, test_rollouts_num]
        """
        batch_size = query_entities.shape[0]
        query_entities_embeds = self.agent.ent_embs(query_entities, torch.zeros_like(query_timestamps))
        query_relations_embeds = self.agent.rel_embs(query_relations)

        self.agent.policy_step.set_hiddenx(batch_size)

        # In the first step, if rollouts_num is greater than the maximum number of actions, select all actions
        current_entites = query_entities
        current_timestamps = query_timestamps
        prev_relations = torch.ones_like(query_relations) * self.num_rel  # NO_OP
        action_space = self.env.next_actions(current_entites, current_timestamps,
                                             query_timestamps, self.max_action_num, True)
        loss, logits, action_id = self.agent(
            prev_relations,
            current_entites,
            current_timestamps,
            query_relations_embeds,
            query_entities_embeds,
            query_timestamps,
            action_space
        )  # logits.shape: [batch_size, max_action_num]

        action_space_size = action_space.shape[1]
        if self.config['beam_size'] > action_space_size:
            beam_size = action_space_size
        else:
            beam_size = self.config['beam_size']
        beam_log_prob, top_k_action_id = torch.topk(logits, beam_size, dim=1)  # beam_log_prob.shape [batch_size, beam_size]
        beam_log_prob = beam_log_prob.reshape(-1)  # [batch_size * beam_size]

        current_entites = torch.gather(action_space[:, :, 1], dim=1, index=top_k_action_id).reshape(-1)  # [batch_size * beam_size]
        current_timestamps = torch.gather(action_space[:, :, 2], dim=1, index=top_k_action_id).reshape(-1) # [batch_size * beam_size]
        prev_relations = torch.gather(action_space[:, :, 0], dim=1, index=top_k_action_id).reshape(-1)  # [batch_size * beam_size]
        self.agent.policy_step.hx = self.agent.policy_step.hx.repeat(1, 1, beam_size).reshape([batch_size * beam_size, -1])  # [batch_size * beam_size, state_dim]
        self.agent.policy_step.cx = self.agent.policy_step.cx.repeat(1, 1, beam_size).reshape([batch_size * beam_size, -1])  # [batch_size * beam_size, state_dim]

        beam_tmp = beam_log_prob.repeat([action_space_size, 1]).transpose(1, 0)  # [batch_size * beam_size, max_action_num]
        for t in range(1, self.path_length):
            query_timestamps_roll = query_timestamps.repeat(beam_size, 1).permute(1, 0).reshape(-1)
            query_entities_embeds_roll = query_entities_embeds.repeat(1, 1, beam_size)
            query_entities_embeds_roll = query_entities_embeds_roll.reshape([batch_size * beam_size, -1])  # [batch_size * beam_size, ent_dim]
            query_relations_embeds_roll = query_relations_embeds.repeat(1, 1, beam_size)
            query_relations_embeds_roll = query_relations_embeds_roll.reshape([batch_size * beam_size, -1])  # [batch_size * beam_size, rel_dim]

            action_space = self.env.next_actions(current_entites, current_timestamps,
                                                     query_timestamps_roll, self.max_action_num)

            loss, logits, action_id = self.agent(
                prev_relations,
                current_entites,
                current_timestamps,
                query_relations_embeds_roll,
                query_entities_embeds_roll,
                query_timestamps_roll,
                action_space
            ) # logits.shape [bs * rollouts_num, max_action_num]

            hx_tmp = self.agent.policy_step.hx.reshape(batch_size, beam_size, -1)
            cx_tmp = self.agent.policy_step.cx.reshape(batch_size, beam_size, -1)

            beam_tmp = beam_log_prob.repeat([action_space_size, 1]).transpose(1, 0) # [batch_size * beam_size, max_action_num]
            beam_tmp += logits
            beam_tmp = beam_tmp.reshape(batch_size, -1)  # [batch_size, beam_size * max_actions_num]

            if action_space_size * beam_size >= self.config['beam_size']:
                beam_size = self.config['beam_size']
            else:
                beam_size = action_space_size * beam_size

            top_k_log_prob, top_k_action_id = torch.topk(beam_tmp, beam_size, dim=1)  # [batch_size, beam_size]
            offset = top_k_action_id // action_space_size  # [batch_size, beam_size]
            offset = offset.unsqueeze(-1).repeat(1, 1, self.config['state_dim'])  # [batch_size, beam_size]
            self.agent.policy_step.hx = torch.gather(hx_tmp, dim=1, index=offset)
            self.agent.policy_step.hx = self.agent.policy_step.hx.reshape([batch_size * beam_size, -1])
            self.agent.policy_step.cx = torch.gather(cx_tmp, dim=1, index=offset)
            self.agent.policy_step.cx = self.agent.policy_step.cx.reshape([batch_size * beam_size, -1])

            current_entites = torch.gather(action_space[:, :, 1].reshape(batch_size, -1), dim=1, index=top_k_action_id).reshape(-1)
            current_timestamps = torch.gather(action_space[:, :, 2].reshape(batch_size, -1), dim=1, index=top_k_action_id).reshape(-1)
            prev_relations = torch.gather(action_space[:, :, 0].reshape(batch_size, -1), dim=1, index=top_k_action_id).reshape(-1)

            beam_log_prob = top_k_log_prob.reshape(-1)  # [batch_size * beam_size]

        return action_space[:, :, 1].reshape(batch_size, -1), beam_tmp
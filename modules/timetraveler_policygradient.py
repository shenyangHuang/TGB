"""
TimeTraveler: Reinforcement Learning for Temporal Knowledge Graph Forecasting
Reference:
- https://github.com/JHL-HUST/TITer/blob/master/model/policyGradient.py and
https://github.com/JHL-HUST/TITer/blob/master/model/baseline.py
Haohai Sun, Jialun Zhong, Yunpu Ma, Zhen Han, Kun He.
TimeTraveler: Reinforcement Learning for Temporal Knowledge Graph Forecasting EMNLP 2021
"""

import torch
import numpy as np
import math

class ReactiveBaseline(object):
    def __init__(self, config, update_rate):
        self.update_rate = update_rate
        self.value = torch.zeros(1)
        if config['cuda']:
            self.value = self.value.cuda()

    def get_baseline_value(self):
        return self.value

    def update(self, target):
        self.value = torch.add((1 - self.update_rate) * self.value, self.update_rate * target)

class PG(object):
    def __init__(self, config):
        self.config = config
        self.positive_reward = 1.0
        self.negative_reward = 0.0
        self.baseline = ReactiveBaseline(config, config['lambda'])
        self.now_epoch = 0

    def get_reward(self, current_entites, answers):
         positive = torch.ones_like(current_entites, dtype=torch.float32) * self.positive_reward
         negative = torch.ones_like(current_entites, dtype=torch.float32) * self.negative_reward
         reward = torch.where(current_entites == answers, positive, negative)
         return reward

    def calc_cum_discounted_reward(self, rewards):
        running_add = torch.zeros([rewards.shape[0]])
        cum_disc_reward = torch.zeros([rewards.shape[0], self.config['path_length']])
        if self.config['cuda']:
            running_add = running_add.cuda()
            cum_disc_reward = cum_disc_reward.cuda()

        cum_disc_reward[:, self.config['path_length'] - 1] = rewards
        for t in reversed(range(self.config['path_length'])):
            running_add = self.config['gamma'] * running_add + cum_disc_reward[:, t]
            cum_disc_reward[:, t] = running_add
        return cum_disc_reward

    def entropy_reg_loss(self, all_logits):
        all_logits = torch.stack(all_logits, dim=2)
        entropy_loss = - torch.mean(torch.sum(torch.mul(torch.exp(all_logits), all_logits), dim=1))
        return entropy_loss

    def calc_reinforce_loss(self, all_loss, all_logits, cum_discounted_reward):
        loss = torch.stack(all_loss, dim=1)
        base_value = self.baseline.get_baseline_value()
        final_reward = cum_discounted_reward - base_value

        reward_mean = torch.mean(final_reward)
        reward_std = torch.std(final_reward) + 1e-6
        final_reward = torch.div(final_reward - reward_mean, reward_std)

        loss = torch.mul(loss, final_reward)
        entropy_loss = self.config['ita'] * math.pow(self.config['zita'], self.now_epoch) * self.entropy_reg_loss(all_logits)

        total_loss = torch.mean(loss) - entropy_loss
        return total_loss
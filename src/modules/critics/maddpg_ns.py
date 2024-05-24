""" FACtored Multi-Agent Centralised policy gradients (not sharing parameters)

code adapted from https://github.com/oxwhirl/facmac/

Paper at https://proceedings.neurips.cc/paper/2021/hash/65b9eea6e1cc6bb9f0cd2a47751a186f-Abstract.html

Preprint at https://arxiv.org/abs/2003.06709
"""

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from pymarl_application.modules.critics.mlp import MLP


class MADDPGCriticNS(nn.Module):
    def __init__(self, scheme,
                 n_actions,
                 n_agents,
                 hidden_dim,
                 obs_individual_obs: bool = False,
                 obs_last_action: bool = False):
        super(MADDPGCriticNS, self).__init__()

        self.n_actions = n_actions
        self.n_agents = n_agents
        self.obs_individual_obs = obs_individual_obs
        self.obs_last_action = obs_last_action

        self.input_shape = self._get_input_shape(scheme) + self.n_actions * self.n_agents
        if self.obs_last_action:
            self.input_shape += self.n_actions
        self.output_type = "q"
        self.critics = [MLP(self.input_shape, hidden_dim, 1) for _ in range(self.n_agents)]

    def forward(self, inputs, actions):
        inputs = th.cat((inputs, actions), dim=-1)
        qs = []
        for i in range(self.n_agents):
            q = self.critics[i](inputs[:, :, i]).unsqueeze(2)
            qs.append(q)
        return th.cat(qs, dim=2)

    def _get_input_shape(self, scheme):
        # state
        input_shape = scheme["state"]["vshape"]
        # observation
        if self.obs_individual_obs:
            input_shape += scheme["obs"]["vshape"]
        return input_shape

    def parameters(self):
        params = list(self.critics[0].parameters())
        for i in range(1, self.n_agents):
            params += list(self.critics[i].parameters())
        return params

    def state_dict(self):
        return [a.state_dict() for a in self.critics]

    def load_state_dict(self, state_dict):
        for i, c in enumerate(self.critics):
            c.load_state_dict(state_dict[i])

    def cuda(self):
        for c in self.critics:
            c.cuda()

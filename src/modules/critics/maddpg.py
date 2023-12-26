# code adapted from https://github.com/oxwhirl/facmac/
import torch as th
import torch.nn as nn
import torch.nn.functional as F


class MADDPGCritic(nn.Module):
    def __init__(self, scheme,
                 n_actions,
                 n_agents,
                 hidden_dim,
                 obs_individual_obs: bool = False,
                 obs_last_action: bool = False,
                 obs_agent_id: bool = False):
        super(MADDPGCritic, self).__init__()

        self.n_actions = n_actions
        self.n_agents = n_agents
        self.obs_individual_obs = obs_individual_obs
        self.obs_last_action = obs_last_action
        self.obs_agent_id = obs_agent_id

        self.input_shape = self._get_input_shape(scheme) + self.n_actions * self.n_agents
        if self.obs_last_action:
            self.input_shape += self.n_actions
        self.output_type = "q"

        # Set up network layers
        self.fc1 = nn.Linear(self.input_shape, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, inputs, actions):
        inputs = th.cat((inputs, actions), dim=-1)
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q

    def _get_input_shape(self, scheme):
        # state
        input_shape = scheme["state"]["vshape"]
        # print(scheme["state"]["vshape"], scheme["obs"]["vshape"], self.n_agents, scheme["actions_one"])
        # whether to add the individual observation
        if self.obs_individual_obs:
            input_shape += scheme["obs"]["vshape"]
        # agent id
        if self.obs_agent_id:
            input_shape += self.n_agents
        return input_shape

"""RNN Agent

code adapted from https://github.com/wendelinboehmer/dcg
"""

import torch.nn as nn
import torch.nn.functional as F
import torch as th


class RNNAgent(nn.Module):
    """RNN Agent

    The RNN Agent creates a 3 layer neural network. 
    The second layer has :obj:`hidden_dim` input and :obj:`hidden_dim` output.

    The second layer can be a :obj:`GRUCell` if :obj:`use_rnn` is :obj:`True`, 
    or a :obj:`Linear` if :obj:`use_rnn` is `False`.
    
    Parameters:
    ------------
        input_shape : int
            the input shape
        n_actions : int
            number of actions
        hidden_dim : int
            hidden dim of the network
        use_rnn : bool
            whether to use GRUCell
    """

    def __init__(self, input_shape:int,
                 n_actions:int,
                 hidden_dim:int,
                 use_rnn: bool):
        super(RNNAgent, self).__init__()

        self.use_rnn = use_rnn
        self.hidden_dim = hidden_dim

        self.fc1 = nn.Linear(input_shape, hidden_dim)
        if self.use_rnn:
            self.rnn = nn.GRUCell(hidden_dim, hidden_dim)
        else:
            self.rnn = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_actions)

    def init_hidden(self):
        """make hidden states on same device as model"""
        return self.fc1.weight.new(1, self.hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        """Forward pass
        
        linear-rnn-linear if :obj:`use_rnn`

        linear-linear-linear if not :obj:`use_rnn`
        """
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.hidden_dim)
        if self.use_rnn:
            h = self.rnn(x, h_in)
        else:
            h = F.relu(self.rnn(x))
        q = self.fc2(h)
        if th.isnan(q).any():
            print("NaN")
        if th.isnan(h).any():
            print("NaN")
        return q, h


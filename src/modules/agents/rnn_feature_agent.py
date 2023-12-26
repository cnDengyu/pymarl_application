import torch.nn as nn


class RNNFeatureAgent(nn.Module):
    """ Identical to rnn_agent, but does not compute value/probability for each action, only the hidden state. """
    def __init__(self, input_shape,
                 hidden_dim):
        nn.Module.__init__(self)

        self.hidden_dim = hidden_dim

        self.fc1 = nn.Linear(input_shape, hidden_dim)
        self.rnn = nn.GRUCell(hidden_dim, hidden_dim)

    def init_hidden(self):
        return self.fc1.weight.new(1, self.hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        x = nn.functional.relu(self.fc1(inputs))
        h = self.rnn(x, hidden_state.reshape(-1, self.hidden_dim))
        return None, h
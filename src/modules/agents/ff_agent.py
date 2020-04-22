from collections import OrderedDict
import torch as th
import torch.nn as nn
import torch.nn.functional as F


class FFAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(FFAgent, self).__init__()
        self.args = args
        self.input_shape = input_shape
        if isinstance(input_shape, tuple):
            assert len(input_shape) == 1, "Input shape has unsupported dimensionality: {}".format(input_shape)
            input_shape = input_shape[0]
        elif isinstance(input_shape, (dict, OrderedDict)):  # assemble all 1d input regions
            # input_shape = sum([v[0] for v in input_shape.values() if len(v) == 1])
            input_shape = input_shape["1d"][0]

        # Easiest to reuse rnn_hidden_dim variable
        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc3 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        if isinstance(inputs, OrderedDict):
            # inputs = th.cat(list(inputs.values()), -1)
            inputs = inputs["1d"]
        x = F.relu(self.fc1(inputs))
        # h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = F.relu(self.fc2(x))
        q = self.fc3(h)
        return q, h

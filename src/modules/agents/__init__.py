REGISTRY = {}

from .rnn_agent import RNNAgent, RNNConvDDPGAgent
REGISTRY["rnn"] = RNNAgent
REGISTRY["rnn_ddpg"] = RNNConvDDPGAgent

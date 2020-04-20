REGISTRY = {}

from .rnn_agent import RNNAgent, RNNConvDDPGAgent
from .ff_agent import FFAgent

REGISTRY["rnn"] = RNNAgent
REGISTRY["rnn_ddpg"] = RNNConvDDPGAgent
REGISTRY["ff"] = FFAgent
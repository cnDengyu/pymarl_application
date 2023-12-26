REGISTRY = {}

from .rnn_agent import RNNAgent
from .rnn_ns_agent import RNNNSAgent
from .rnn_feature_agent import RNNFeatureAgent
REGISTRY["rnn"] = RNNAgent
REGISTRY["rnn_ns"] = RNNNSAgent
REGISTRY["rnn_feat"] = RNNFeatureAgent

def registered_agent(args):
    if args.agent == "rnn":
        agent_fn = lambda input_shape: RNNAgent(input_shape, args.n_actions, args.hidden_dim, args.use_rnn)
    elif args.agent == "rnn_ns":
        agent_fn = lambda input_shape: RNNNSAgent(input_shape, args.n_agents, args.n_actions, args.hidden_dim, args.use_rnn)
    elif args.agent == "rnn_feat":
        agent_fn = lambda input_shape: RNNFeatureAgent(input_shape, args.hidden_dim)
    return agent_fn

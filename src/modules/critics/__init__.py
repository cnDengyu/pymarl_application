from .coma import COMACritic
from .centralV import CentralVCritic
from .coma_ns import COMACriticNS
from .centralV_ns import CentralVCriticNS
from .maddpg import MADDPGCritic
from .maddpg_ns import MADDPGCriticNS
from .ac import ACCritic
from .ac_ns import ACCriticNS
from .pac_ac_ns import PACCriticNS
from .pac_dcg_ns import DCGCriticNS
REGISTRY = {}

REGISTRY["coma_critic"] = COMACritic
REGISTRY["cv_critic"] = CentralVCritic
REGISTRY["coma_critic_ns"] = COMACriticNS
REGISTRY["cv_critic_ns"] = CentralVCriticNS
REGISTRY["maddpg_critic"] = MADDPGCritic
REGISTRY["maddpg_critic_ns"] = MADDPGCriticNS
REGISTRY["ac_critic"] = ACCritic
REGISTRY["ac_critic_ns"] = ACCriticNS
REGISTRY["pac_critic_ns"] = PACCriticNS
REGISTRY["pac_dcg_critic_ns"] = DCGCriticNS

def registered_critic(name: str, scheme, args):
    if name == "coma_critic":
        critic = COMACritic(scheme, args.n_actions, args.n_agents, args.hidden_dim, args.obs_individual_obs, 
                            args.obs_last_action, args.obs_agent_id)
    elif name == "cv_critic":
        critic = CentralVCritic(scheme, args.n_actions, args.n_agents, args.hidden_dim, args.obs_individual_obs,
                                args.obs_last_action)
    elif name == "coma_critic_ns":
        critic = COMACriticNS(scheme, args.n_actions, args.n_agents, args.hidden_dim, args.obs_individual_obs,
                              args.obs_last_action)
    elif name == "cv_critic_ns":
        critic = CentralVCriticNS(scheme, args.n_actions, args.n_agents, args.hidden_dim, args.obs_individual_obs,
                                  args.obs_last_action)
    elif name == "maddpg_critic":
        critic = MADDPGCritic(scheme, args.n_actions, args.n_agents, args.hidden_dim, args.obs_individual_obs, 
                              args.obs_last_action, args.obs_agent_id)
    elif name == "maddpg_critic_ns":
        critic = MADDPGCriticNS(scheme, args.n_actions, args.n_agents, args.hidden_dim, args.obs_individual_obs,
                                args.obs_last_action)
    elif name == "ac_critic":
        critic = ACCritic(scheme, args.n_actions, args.n_agents, args.hidden_dim)
    elif name == "ac_critic_ns":
        critic = ACCriticNS(scheme, args.n_actions, args.n_agents, args.hidden_dim)
    elif name == "pac_critic_ns":
        critic = PACCriticNS(scheme, args.n_actions, args.n_agents, args.hidden_dim, args.obs_individual_obs,
                             args.obs_last_action, args.use_cuda)
    elif name == "pac_dcg_critic_ns":
        critic = DCGCriticNS(scheme, args.n_agents, args.n_actions, args.hidden_dim, args.agent_output_type,
                             args.cg_payoff_rank, args.msg_iterations, args.msg_normalized, args.msg_anytime,
                             args.cg_utilities_hidden_dim, args.cg_edges, args.cg_payoffs_hidden_dim)
    return critic

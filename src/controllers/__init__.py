REGISTRY = {}

from .basic_controller import BasicMAC
from .non_shared_controller import NonSharedMAC
from .maddpg_controller import MADDPGMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["non_shared_mac"] = NonSharedMAC
REGISTRY["maddpg_mac"] = MADDPGMAC

def registered_controller(scheme, groups, agent_fn, action_selector, args):
    mask_before_softmax = getattr(args, "mask_before_softmax", True)
    if args.mac == "basic_mac":
        mac = BasicMAC(scheme, groups, args.n_agents, agent_fn, args.agent_output_type, action_selector, 
                       args.obs_last_action, args.obs_agent_id, mask_before_softmax)
    elif args.mac == "non_shared_mac":
        mac = NonSharedMAC(scheme, groups, args.n_agents, agent_fn, args.agent_output_type, action_selector, 
                           args.obs_last_action, args.obs_agent_id, mask_before_softmax)
    elif args.mac == "maddpg_mac":
        mac = MADDPGMAC(scheme, groups, args.n_agents, agent_fn, args.agent_output_type, 
                        args.obs_last_action, args.obs_agent_id)
    return mac

import torch as th
from components.episode_buffer import ReplayBuffer

class MAPPO:
    def __init__(
        self,
        num_agents,
        env,
        actor,
        critic,
        multiagent_controller,
        buffer_size
        ):

        self.num_agnets = num_agents
        self.env = env
        self.actor = actor
        self.critic = critic
        self.mac = multiagent_controller

        # Default/Base scheme
        scheme = {
            "state": {"vshape": env_info["state_shape"]},
            "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
            "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
            "avail_actions": {
                "vshape": (env_info["n_actions"],),
                "group": "agents",
                "dtype": th.int,
            },
            "reward": {"vshape": (1,)},
            "terminated": {"vshape": (1,), "dtype": th.uint8},
        }
        groups = {"agents": args.n_agents}
        preprocess = {"actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])}

        buffer = ReplayBuffer(
            scheme,
            groups,
            buffer_size,
            env_info["episode_limit"] + 1,
            preprocess=preprocess,
            device="cpu" if args.buffer_cpu_only else args.device,
        )

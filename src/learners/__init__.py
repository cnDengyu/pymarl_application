from .q_learner import QLearner
from .coma_learner import COMALearner
from .qtran_learner import QLearner as QTranLearner
from .actor_critic_learner import ActorCriticLearner
from .actor_critic_pac_learner import PACActorCriticLearner
from .actor_critic_pac_dcg_learner import PACDCGLearner
from .maddpg_learner import MADDPGLearner
from .ppo_learner import PPOLearner
REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY["qtran_learner"] = QTranLearner
REGISTRY["actor_critic_learner"] = ActorCriticLearner
REGISTRY["maddpg_learner"] = MADDPGLearner
REGISTRY["ppo_learner"] = PPOLearner
REGISTRY["pac_learner"] = PACActorCriticLearner
REGISTRY["pac_dcg_learner"] = PACDCGLearner

def registered_learner(mac, critic, state_value, scheme, logger, args):
    if args.learner == "q_learner":
        learner = QLearner(mac, scheme, logger, args.n_agents, args.mixer, args.lr, args.learner_log_interval, args.use_cuda,
                           args.standardise_returns, args.standardise_rewards, args.double_q, args.gamma, args.grad_norm_clip,
                           args.target_update_interval_or_tau, args.state_shape, args.mixing_embed_dim, args.hypernet_embed, args.hypernet_layers)
    elif args.learner == "coma_learner":
        learner = COMALearner(mac, scheme, logger, args.n_agents, args.n_actions, args.learner_log_interval, args.critic, args.lr,
                              args.use_cuda, args.standardise_returns, args.standardise_rewards, args.entropy_coef, args.grad_norm_clip,
                              args.target_update_interval_or_tau, args.q_nstep, args.gamma, args.add_value_last_step)
    elif args.learner == "qtran_learner":
        learner = QTranLearner(mac, scheme, logger, args.n_agents, args.n_actions, args.mixer, args.lr, args.optim_alpha, args.optim_eps,
                               args.learner_log_interval, args.double_q, args.gamma, args.opt_loss, args.nopt_min_loss,
                               args.grad_norm_clip, args.target_update_interval, args.state_shape, args.qtran_arch,
                               args.mixing_embed_dim, args.rnn_hidden_dim, args.network_size)
    elif args.learner == "actor_critic_learner":
        learner = ActorCriticLearner(mac, scheme, logger, args.n_agents, args.n_actions, args.lr, critic,
                                     args.learner_log_interval, args.use_cuda, args.standardise_returns,
                                     args.standardise_rewards, args.entropy_coef, args.grad_norm_clip,
                                     args.target_update_interval_or_tau, args.q_nstep, args.gamma, args.add_value_last_step)
    elif args.learner == "maddpg_learner":
        learner = MADDPGLearner(mac, scheme, logger, args.n_agents, args.n_actions, args.critic, args.lr,
                                args.learner_log_interval, args.use_cuda, args.standardise_returns,
                                args.standardise_rewards, args.gamma, args.grad_norm_clip, args.reg,
                                args.target_update_inetrval_or_tau, args.obs_individual_obs,
                                args.obs_last_action, args.obs_agent_id)
    elif args.learner == "ppo_learner":
        learner = PPOLearner(mac, scheme, logger, args.n_agents, args.n_actions, args.lr, critic, args.learner_log_interval,
                             args.use_cuda, args.standardise_returns, args.standardise_rewards, args.epochs, args.eps_clip,
                             args.entropy_coef, args.grad_norm_clip, args.target_update_interval_or_tau,
                             args.q_nstep, args.gamma, args.add_value_last_step)
    elif args.learner == "pac_learner":
        learner = PACActorCriticLearner(mac, scheme, logger, args.n_agents, args.n_actions, args.lr, critic, state_value,
                                        args.learner_log_interval, args.use_cuda, args.t_max, args.entropy_end_ratio,
                                        args.final_entropy_coef, args.initial_entropy_coef, args.grad_norm_clip,
                                        args.target_update_interval_or_tau, args.standardise_rewards, args.q_nstep,
                                        args.gamma)
    elif args.learner == "pac_dcg_learner":
        learner = PACDCGLearner(mac, scheme, logger, args.n_agents, args.n_actions, args.lr, critic, state_value,
                                args.learner_log_interval, args.use_cuda, args.t_max, args.entropy_end_ratio,
                                args.final_entropy_coef, args.initial_entropy_coef, args.grad_norm_clip,
                                args.target_update_interval_or_tau, args.standardise_rewards,
                                args.q_nstep, args.gamma)
    return learner

REGISTRY = {}

from .episode_runner import EpisodeRunner
REGISTRY["episode"] = EpisodeRunner

from .parallel_runner import ParallelRunner
REGISTRY["parallel"] = ParallelRunner

def registered_runner(env_fn, device, logger, args):
    if args.runner == "episode":
        runner = EpisodeRunner(logger, args.batch_size_run, env_fn, args.env_args, device,
                               args.render, args.test_nepisode, args.runner_log_interval)
    elif args.runner == "parallel":
        runner = ParallelRunner(logger, args.batch_size_run, env_fn, args.env_args, device, 
                                args.render, args.test_nepisode, args.runner_log_interval)
    return runner

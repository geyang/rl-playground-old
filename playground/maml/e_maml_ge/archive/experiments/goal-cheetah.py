from datetime import datetime

import os

import config
import scripts.run_maml as run_maml


def experiment(args):
    # config.G.run_mode = run_mode
    config.G.env_name = "HalfCheetahGoalVel-v0"
    config.G.activation = "relu"
    config.G.hidden_size = 100
    config.G.first_order = True
    config.G.n_epochs = 800
    # config.G.n_graphs = n_graphs
    config.G.n_parallel_envs = 8
    config.G.batch_timesteps = 5
    # config.G.alpha = alpha
    # config.G.beta = beta
    # config.G.inner_alg = inner_alg
    config.G.inner_optimizer = "SGD"
    config.G.meta_alg = "PPO"
    config.G.meta_optimizer = "Adam"

    # config.G.n_grad_steps = n_grad_steps
    config.G.update(args)
    config.G.eval_grad_steps = list(range(args['n_grad_steps'] + 1))

    config.Reporting.plot_server = "http://127.0.0.1"
    config.Reporting.report_mean = False
    config.DEBUG.no_task_resample = False

    now = datetime.now()

    config.RUN.prefix = "goal_cheetah_sweep"
    config.RUN.log_directory = ROOT_DIR + config.DIR_TEMPLATE.format(now=now, **vars(config.RUN), **vars(config))
    config.RUN.job_name = config.JOB_TEMPLATE.format(now=now, **vars(config.RUN))
    os.makedirs(config.RUN.log_directory, exist_ok=True)

    # config.DEBUG.no_task_resample = 1
    # config.DEBUG.no_weight_reset = 1

    try:
        run_maml.run_e_maml()
    except Exception as e:
        print("this run is terminated with", e)


ROOT_DIR = "/mnt/slab/krypton/e_maml_rebuttal/"
if __name__ == "__main__":
    params = []
    inner_alg = "VPG"
    n_grad_steps = 1
    alpha = 0.01
    beta = 0.01
    n_tasks = 40
    run_mode = 'maml'
    for i in range(10):
        params.append(dict(start_seed=i, inner_alg=inner_alg, beta=beta, alpha=alpha,
                           n_grad_steps=n_grad_steps, n_tasks=n_tasks, run_mode=run_mode))

    for p in params:
        experiment(p)

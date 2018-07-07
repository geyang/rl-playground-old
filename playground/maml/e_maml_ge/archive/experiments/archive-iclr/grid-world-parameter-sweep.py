import config
from scripts.run_maml_gce import run_e_maml_gce


def experiment(*, alpha, beta, n_tasks, n_grad_steps=None, run_mode=None, inner_alg=None, **_):
    def maze_parameter_sweep():
        config.G.run_mode = run_mode
        config.G.env_name = "Maze10-v0"
        config.G.first_order = True
        config.G.n_epochs = 1700
        config.G.n_tasks = n_tasks
        config.G.n_parallel_envs = 8
        config.G.batch_timesteps = 5
        config.G.n_grad_steps = n_grad_steps
        config.G.eval_grad_steps = list(range(n_grad_steps + 1))
        config.G.alpha = alpha
        config.G.beta = beta
        config.G.inner_alg = inner_alg
        config.G.inner_optimizer = "SGD"
        config.G.meta_alg = "PPO"
        config.G.meta_optimizer = "Adam"

        config.Reporting.report_mean = True
        config.DEBUG.no_task_resample = False

    def multiple_gradient_steps():
        config.G.run_mode = run_mode
        config.G.env_name = "MediumWorld-v0"
        config.G.first_order = True
        config.G.n_epochs = 800  # 26 * 10
        config.G.n_tasks = n_tasks
        config.G.n_parallel_envs = 8
        config.G.batch_timesteps = 5
        config.G.n_grad_steps = n_grad_steps
        config.G.eval_grad_steps = list(range(n_grad_steps + 1))
        config.G.alpha = alpha
        config.G.beta = beta
        config.G.inner_alg = inner_alg
        config.G.inner_optimizer = "SGD"
        config.G.meta_alg = "PPO"
        config.G.meta_optimizer = "Adam"

        config.Reporting.report_mean = True
        config.DEBUG.no_task_resample = False

    #   ================= RAN ==================
    def ppo_inner_loop():
        config.G.run_mode = run_mode
        config.G.env_name = "MediumWorld-v0"
        config.G.first_order = True
        config.G.n_epochs = 800  # 26 * 10
        config.G.n_tasks = n_tasks
        config.G.n_parallel_envs = 8
        config.G.batch_timesteps = 5
        config.G.n_grad_steps = 1
        config.G.eval_grad_steps = [0, 1]
        config.G.alpha = alpha
        config.G.beta = beta
        config.G.inner_alg = "PPO"
        config.G.inner_optimizer = "SGD"
        config.G.meta_alg = "PPO"
        config.G.meta_optimizer = "Adam"

        config.Reporting.report_mean = True
        config.DEBUG.no_task_resample = False

    def e_maml_parameter_sweep():
        config.G.run_mode = "e_maml"
        config.G.env_name = "MediumWorld-v0"
        config.G.first_order = True
        config.G.n_epochs = 800  # 26 * 10
        config.G.n_tasks = n_tasks
        config.G.n_parallel_envs = 8
        config.G.batch_timesteps = 5
        config.G.n_grad_steps = 1
        config.G.eval_grad_steps = [0, 1]
        config.G.alpha = alpha
        config.G.beta = beta
        config.G.inner_alg = "VPG"
        config.G.inner_optimizer = "SGD"
        config.G.meta_alg = "PPO"
        config.G.meta_optimizer = "Adam"

        config.Reporting.report_mean = True
        config.DEBUG.no_task_resample = False

    def maml_parameter_sweep():
        config.G.run_mode = "maml"
        config.G.env_name = "MediumWorld-v0"
        config.G.first_order = True
        config.G.n_epochs = 800  # 26 * 10
        config.G.n_tasks = n_tasks
        config.G.n_parallel_envs = 8
        config.G.batch_timesteps = 5
        config.G.n_grad_steps = 1
        config.G.eval_grad_steps = [0, 1]
        config.G.alpha = alpha
        config.G.beta = beta
        config.G.inner_alg = "VPG"
        config.G.inner_optimizer = "SGD"
        config.G.meta_alg = "PPO"
        config.G.meta_optimizer = "Adam"

        config.Reporting.report_mean = True
        config.DEBUG.no_task_resample = False

    return maze_parameter_sweep


SAFTY_ON = True

if __name__ == "__main__":
    import os

    os.environ['action'] = "submit"

    for inner_alg in ['VPG']:
        for beta in [0.001, 0.01, 0.1, 1.0]:
            for alpha in [0.001, 0.01, 0.1, 1.0]:
                for n_grad_steps in [1]:
                    for n_tasks in [32, 64, 128]:
                        for run_mode in ['e_maml']:
                            run_e_maml_gce(config_fn=experiment(**{k: v for k, v in locals().items() if k[0] != "_"}),
                                           dry_run=SAFTY_ON)

    # for inner_alg in ['PPO']:
    #     for n_graphs in [32, 64, 128]:
    #         for alpha in [0.00001, 0.00003, 0.0001, 0.0003]:
    #             for beta in [0.05, 0.1, 0.5]:
    #                     for n_grad_steps in [1, 3, 5]:
    #                         for run_mode in ['maml', 'e_maml']:
    #                             run_e_maml_gce(config_fn=experiment(**locals()), dry_run=SAFTY_ON)

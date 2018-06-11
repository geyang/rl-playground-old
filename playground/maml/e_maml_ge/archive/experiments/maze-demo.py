import config
from scripts.run_maml_gce import run_e_maml_gce


def maze_demo():
    print(locals())

    def maze_test():
        config.G.env_name = "Maze10-v0"
        config.G.first_order = True
        config.G.n_epochs = 800  # 26 * 10
        config.G.n_tasks = 128
        config.G.n_parallel_envs = 10
        config.G.batch_timesteps = 5
        config.G.n_grad_steps = 1
        config.G.eval_grad_steps = [0, 1]
        config.G.alpha = 1.0
        config.G.beta = 0.01
        config.G.inner_alg = "VPG"
        config.G.inner_optimizer = "SGD"
        config.G.meta_alg = "PPO"
        config.G.meta_optimizer = "Adam"

        config.Reporting.report_mean = True
        config.DEBUG.no_task_resample = False

    return maze_test


SAFTY_ON = False

if __name__ == "__main__":
    import os
    os.environ['action'] = "run"
    run_e_maml_gce(config_fn=maze_demo(), dry_run=SAFTY_ON)

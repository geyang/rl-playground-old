import matplotlib

matplotlib.use('TkAgg')

from datetime import datetime
from rl_algs import logger
import config
from scripts.run_maml import run_e_maml


def gce_run(config_fn=None):
    """Runner function wrapper that sets the RUN.log_directory during `gce.call`. """
    prefix = config_fn.__name__.replace('_', "-")
    config.RUN.prefix = prefix
    config.RUN.log_directory = logger.get_dir()

    if callable(config_fn):
        config_fn()

    run_e_maml()


# gca call example function
def run_e_maml_gce(config_fn, dry_run=False):
    """set the action env parameter to run locally."""
    from rcall import gce

    gce.GCE_BUCKET = "openai-ge-bucket"
    gce.CHECKED_BUCKET_EXISTS = True
    gce.GCE_INCLUDE_PATHS = ["./rl-algs-2",
                             "./exploration_in_meta_learning_reproduction_code/krazy_grid_world/maze_data",
                             "./exploration_in_meta_learning_reproduction_code/e_maml_ge",
                             "./exploration_in_meta_learning_reproduction_code/packages"]
    gce.GCE_IGNORE_PATTERNS = ['.idea', '*.pyc', '.git', '*.o', '*.so', '*.dylib', '*.egg-info', '__pycache__',
                               '.DS_Store', '.cache', '.ipynb_checkpoints']
    gce.GCE_IMAGE = 'universe-162007:rl-algs-py36-1'  # same as the rcall default, just to be explicit
    gce.GCE_EXTRA_SETUP = (
        "pip install -e /root/code/rl-algs-2 --ignore-installed\n"
        "pip install ruamel.yaml mock pygame visdom\n"
        "pip install cloudpickle --ignore-installed\n"
        "export PYTHONPATH=/root/code/exploration_in_meta_learning_reproduction_code/e_maml_ge:$PYTHONPATH\n"
        "export PYTHONPATH=/root/code/exploration_in_meta_learning_reproduction_code/packages:$PYTHONPATH")

    # Calling config function to update G so that we could format the log_path note: I don't like this double tap.
    config_fn()
    # now configure the run
    now = datetime.now()
    prefix = config_fn.__name__.replace('_', "-")
    job_name = config.RUN.job_name.format(RUN=config.RUN, now=now, prefix=prefix)
    log_path = config.RUN.log_directory.format(RUN=config.RUN, G=config.G, now=now, prefix=prefix)

    thunk = (lambda: gce_run(config_fn=config_fn))
    if dry_run:
        print('dry run is complete.')
    else:
        gce.call(thunk, job_name=job_name, log_relpath=log_path, machine_type='n1-standard-4', boot_disk_size=20)


if __name__ == '__main__':
    def default():
        config.G.run_mode = "maml"
        config.G.env_name = "MediumWorld-v0"
        config.G.first_order = True
        config.G.n_epochs = 90  # 26 * 10
        config.G.n_tasks = 2
        config.G.n_parallel_envs = 8
        config.G.batch_timesteps = 5
        config.G.n_grad_steps = 1
        config.G.eval_grad_steps = [0, 1]
        # config.G.n_grad_steps = 5
        # config.G.eval_grad_steps = [0, 1, 2, 3, 4, 5]
        config.G.alpha = 0.05
        config.G.beta = 0.008
        config.G.inner_alg = "VPG"
        config.G.inner_optimizer = "SGD"
        config.G.meta_alg = "PPO"
        config.G.meta_optimizer = "Adam"

        config.Reporting.report_mean = True
        config.DEBUG.no_task_resample = False


    run_e_maml_gce(config_fn=default)

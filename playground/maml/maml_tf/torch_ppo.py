from ml_logger import logger
from params_proto import cli_parse

from playground.maml.maml_tf.meta_rl_tasks import MetaRLTasks
# from playground.maml.maml_tf.trainer import comet_logger


@cli_parse
class G:
    log_dir = "http://54.71.92.65:8081"
    log_prefix = "torch_ppo"

    env_name = "HalfCheetahGoalDir-v0"
    n_parallel_envs = 40
    start_seed = 40
    batch_timesteps = 100
    env_max_timesteps = None


def run_maml(_G=None):
    if _G is not None:
        G.update(_G)

    # for k, v in vars(G).items():
    #     comet_logger.log_parameter(k, v)

    # todo: let's take the control of the log director away from the train script. It should all be set from outside.
    logger.configure(log_directory=G.log_dir, prefix=G.log_prefix)
    logger.log_params(G=vars(G), )
    logger.log_file(__file__)

    tasks = MetaRLTasks(env_name=G.env_name, batch_size=G.n_parallel_envs,
                        start_seed=G.start_seed, max_steps=G.env_max_timesteps)

    env = tasks.sample()

    print(env)

    

def launch(**_G):
    import baselines.common.tf_util as U
    import traceback
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = str(1)

    try:
        run_maml(_G=_G)
    except Exception as e:
        tb = traceback.format_exc()
        logger.print(tb)
        logger.print(U.ALREADY_INITIALIZED)
        raise e


if __name__ == '__main__':
    launch()

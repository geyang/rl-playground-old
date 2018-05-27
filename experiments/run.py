from experiments.jaynes_call import rcall
from params_proto import cli_parse


@cli_parse
class Args:
    log_dir = "http://54.71.92.65:8081"
    docker_image = "thanard/matplotlib:latest"
    # log_dir = os.path.realpath("./outputs")
    log_prefix = "test_launch"
    spot_price = 0.273
    mode = "ssh"


def test_launch(**_Args):
    from ml_logger import logger
    Args.update(_Args)

    logger.configure(log_directory=Args.log_dir, prefix=Args.log_prefix)
    logger.print('yo!!! diz is vorking!')


test_launch.Args = Args

if __name__ == "__main__":
    # launch = test_launch
    from playground.maml.maml_torch.maml_torch import launch

    # the location of the log server
    launch.Args.log_dir = "http://54.71.92.65:8081"

    ips = ["52.88.91.243"]
    SSH_IP = ips[0]

    J = rcall(_verbose=True,
              _s3_prefix="s3://ge-bair/",
              _code_name=None,
              _code_root="../",
              _excludes="--exclude='*.png' --exclude='*__pycache__' --exclude='*.git' "
                        "--exclude='*.idea' --exclude='*.egg-info' --exclude='dist' --exclude='build' "
                        "--exclude='.pytest_cache' --exclude='__dataset' --exclude='outputs'"
              )

    # These are the k steps from the dataset
    _mode = launch.Args.mode or "ssh"
    J.run(launch, **vars(launch.Args),
          _log_dir=f"/tmp/jaynes-runs/{launch.Args.log_prefix}",
          _instance_prefix=launch.Args.log_prefix,
          _mode=_mode,
          _instance_type=launch.Args.instance_type or "p2.xlarge",
          _spot_price=launch.Args.spot_price or None,
          _ip=SSH_IP,
          _as_daemon=True,
          # we can probably absorb all of these into just the launch function! Muhaha
          _docker_image=launch.Args.docker_image or "python:3.6",
          _use_gpu=launch.Args.use_gpu if launch.Args.use_gpu is not None else False,
          _startup_script=(
              "echo `which python3`",
              "python3 -V",
              "pip install --upgrade pip jaynes cloudpickle ml-logger moleskin params_proto "
              "torch_helpers dill tqdm networkx astar",
              "export PYTHONIOENCODING=utf-8",),
          )

    print('finished launching!')

    if _mode == "ssh":
        while True:
            from time import sleep

            sleep(100)
            print('waiting for docker logger')

from experiments.jaynes_call import rcall
from params_proto import cli_parse
from waterbear import Bear


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
    # run = test_launch
    from playground.maml.maml_torch.experiments.out_of_distribution import launch_training

    # the location of the log server
    from datetime import datetime

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
    _mode = "spot"
    _price = 0.472
    _image = "ufoym/deepo:cpu"
    _use_gpu = False
    _instance_type = "c4.8xlarge"

    _ = launch.Args
    _ext = f"{_instance_type}-{_price}" if _mode == "spot" else f"ssh"
    now = datetime.now()
    _.log_prefix = f"{now:%Y-%m-%d}/maml_torch/out-of-distribution-beta_{_.beta}-{_ext}"

    J.run(launch, **vars(launch.Args),
          _log_dir=f"/tmp/jaynes-runs/{_.log_prefix}",
          _instance_prefix=launch.Args.log_prefix,
          _mode=_mode,
          _spot_price=_price,
          _instance_type=_instance_type or "p2.xlarge",
          _ip=SSH_IP,
          _as_daemon=True,
          # we can probably absorb all of these into just the run function! Muhaha
          _docker_image=_image,
          _use_gpu=_use_gpu,
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

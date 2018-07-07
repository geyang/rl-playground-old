from experiments.jaynes_call import rcall
from params_proto import cli_parse


@cli_parse
class RunConfig:
    mode = "spot"
    log_dir = "http://54.71.92.65:8081"
    docker_image = "ufoym/deepo:cpu"
    price = 0.472
    instance_type = "c4.4xlarge"


# the location of the log server
ips = ["52.88.91.243"]
SSH_IP = ips[0]

J = rcall(_verbose=True,
          _s3_prefix="s3://ge-bair/",
          _code_name=None,
          _code_root="../../../../",
          _excludes="--exclude='*.png' --exclude='*__pycache__' --exclude='*.git' "
                    "--exclude='*.idea' --exclude='*.egg-info' --exclude='dist' --exclude='build' "
                    "--exclude='.pytest_cache' --exclude='__dataset' --exclude='outputs'"
          )


def run(fn, log_prefix, **p):
    J.run(fn, **p,
          _log_dir=f"/tmp/jaynes-runs/{log_prefix}",
          _instance_prefix=log_prefix + (".ssh" if RunConfig.mode is "ssh" else ""),
          _mode=RunConfig.mode,
          _spot_price=RunConfig.price,
          _instance_type=RunConfig.instance_type,
          _ip=SSH_IP,
          _as_daemon=True,
          # we can probably absorb all of these into just the run function! Muhaha
          _docker_image=RunConfig.docker_image,
          _use_gpu=False,
          _startup_script=(
              "echo `which python3`",
              "python3 -V",
              "pip install --upgrade pip jaynes cloudpickle ml-logger moleskin params_proto "
              "torch_helpers dill tqdm networkx astar",
              "export PYTHONIOENCODING=utf-8",),
          )

from experiments.jaynes_call import rcall
from params_proto import cli_parse


@cli_parse
class RunConfig:
    mode = "spot"
    log_dir = "http://54.71.92.65:8081"
    docker_image = "ufoym/deepo:cpu"
    price = 0.472
    instance_type = "c4.4xlarge"


if __name__ == "__main__":
    from playground.maml.maml_torch.maml_multi_step import launch_maml_mlp, launch_maml_rnn, G, now, launch_reptile_mlp, \
    launch_reptile_rnn, launch_reptile_auto_rnn, launch_maml_auto_rnn, launch_maml_lstm, launch_maml_auto_lstm, \
    launch_reptile_lstm, launch_reptile_auto_lstm

    fns = [
        # launch_maml_rnn,
        # launch_maml_auto_rnn,
        # launch_reptile_rnn,
        # launch_reptile_auto_rnn,
        launch_maml_lstm,
        launch_maml_auto_lstm,
        launch_reptile_lstm,
        launch_reptile_auto_lstm,
    ]
    # the location of the log server
    ips = ["52.88.91.243"]
    SSH_IP = ips[0]


    G.log_dir = "http://54.71.92.65:8081"
    G.task_batch_n = 25
    G.k_shot = 10

    params = []
    # raise NotImplementedError('need to check this out.')
    for fn in fns:
        func_name = fn.__name__[7:].replace('_','-')
        for G.n_gradient_steps in range(1, 6):
            G.test_interval = 20
            G.test_grad_steps = list(range(G.n_gradient_steps + 1))
            G.log_prefix = f'{now:%Y-%m-%d}/debug-maml-baselines/sine-rl2-{func_name}-{G.n_gradient_steps}-step'
            params.append((fn, vars(G)))

    J = rcall(_verbose=True,
              _s3_prefix="s3://ge-bair/",
              _code_name=None,
              _code_root="../../../../",
              _excludes="--exclude='*.png' --exclude='*__pycache__' --exclude='*.git' "
                        "--exclude='*.idea' --exclude='*.egg-info' --exclude='dist' --exclude='build' "
                        "--exclude='.pytest_cache' --exclude='__dataset' --exclude='outputs'"
              )
    
    for fn, p in params:
        J.run(fn, **p,
              _log_dir=f"/tmp/jaynes-runs/{p['log_prefix']}",
              _instance_prefix=p['log_prefix'] + (".ssh" if RunConfig.mode is "ssh" else ""),
              _mode=RunConfig.mode,
              _spot_price=RunConfig.price,
              _instance_type=RunConfig.instance_type,
              _ip=SSH_IP,
              _as_daemon=True,
              # we can probably absorb all of these into just the run function! Muhaha
              _docker_image=RunConfig.docker_image,
              _use_gpu=True,
              _startup_script=(
                  "echo `which python3`",
                  "python3 -V",
                  "pip install --upgrade pip jaynes cloudpickle ml-logger moleskin params_proto "
                  "torch_helpers dill tqdm networkx astar",
                  "export PYTHONIOENCODING=utf-8",),
              )

    print('finished launching!')

    if RunConfig.mode == "ssh":
        while True:
            from time import sleep

            sleep(100)
            print('waiting for docker logger')

import multiprocessing
from params_proto import cli_parse

ALLOWED_ALGS = "rl_algs.PPO", "rl_algs.VPG", "PPO", "VPG"
DIR_TEMPLATE = "{prefix}/{now:%Y-%m-%d}/{now:%H%M}-{prefix}/{now:%H%M%S.%f}" \
               "-{G.run_mode}-{G.env_name}-n_grad({G.n_grad_steps})" \
               "-{G.inner_alg}-{G.inner_optimizer}" \
               "-{G.meta_alg}-{G.meta_optimizer}-alpha({G.alpha})-beta({G.beta})" \
               "-n_tasks({G.n_tasks})-env_norm({G.normalize_env})"  # type: "directory to use for logging"
JOB_TEMPLATE = "{prefix}-{now:%Y-%m-%d}-{now:%H%M%S-%f}"


@cli_parse
class RUN:
    log_dir = "http://54.71.92.65:8081"
    log_prefix = "maml-debug-run"
    prefix = "debug-run"  # type: "will be replaced by the run_time prefix during gce.call"
    job_name = JOB_TEMPLATE  # type: "template for the cloud jobs"


# decorator help generate a as command line parser.
@cli_parse
class G:
    term_reward_threshold = -8000.0
    run_mode = "maml"  # type:  "Choose between maml and e_maml. Switches the loss function used for training"
    # env_name = 'HalfCheetah-v2'  # type:  "Name of the task environment"
    env_name = 'HalfCheetahGoalVel-v0'  # type:  "Name of the task environment"
    task_seed = 69  # type:  "seed to use in grid tasks"
    start_seed = 69  # type:  "seed for initialization of each game"
    test_task_seed = 69  # type:  "seed to use in grid tasks during test"
    test_start_seed = 69  # type:  "seed for initializing each game in test"
    render = False
    n_cpu = 2 * multiprocessing.cpu_count() # type: "number of threads used"
    eval_test_interval = 0  # type:  "The interval to test the agent on a fixed set of tasks"
    # Note: (E_)MAML Training Parameters
    n_tasks = 40  # type:  "40 for locomotion, 20 for 2D navigation ref:cbfinn"
    n_grad_steps = 1  # type:  "number of gradient descent steps for the worker." #TODO change back to 1
    n_epochs = 2000  # type:  "Number of epochs"
    # 40k per task (action, state) tuples, or 20k (per task) if you have 10/20 meta tasks
    n_parallel_envs = 40  # type:  "Number of parallel envs in minibatch. The SubprocVecEnv batch_size."
    batch_timesteps = 40  # type:  "max_steps for each episode, used to set env._max_steps parameter"
    env_max_timesteps = 0  # type:  "max_steps for each episode, used to set env._max_steps parameter. 0 to use gym default."
    single_sampling = 0  # type:  "flag for running a single sampling step. 1 ON, 0 OFF"
    # NOTE: NOT USED in maml or e_maml, only in baseline.
    # n_optimization_epochs = 10  # type:  'the number of epochs in PPO for optimization on the same dataset'
    # n_optimization_batches = 1  # type:  "number of batches in the optimization inner loop"
    # optimizer = "Adam"
    # shuffle_samples = 1
    # n_updates_per_batch = 16  # type:  "Run optimization 16 times for the mini_batch. Used only in baseline runs."
    eval_grad_steps = [0, 1]  # type:  "the gradient steps at which we evaluate the policy. Used to make pretty plots."
    # Note: MAML Options
    first_order = False  # type:  "Whether to stop gradient calculation during meta-gradient calculation"
    alpha = 0.0001  # type:  "worker learning rate. use 0.1 for first step, 0.05 afterward ref:cbfinn"
    beta = 0.005  # type:  "meta learning rate"
    inner_alg = "PPO"  # type:  '"PPO" or "VPG", "rl_algs.VPG" or "rl_algs.PPO" for rl_algs baselines'
    inner_optimizer = "SGD"  # type:  '"Adam" or "SGD"'
    meta_alg = "PPO"  # type:  "PPO or TRPO, TRPO is not yet implemented."
    meta_optimizer = "Adam"  # type:  '"Adam" or "SGD"'
    activation = "tanh"
    hidden_size = 32  # type: "hidden size for the MLP policy"
    # Model options
    normalize_env = False  # type: "normalize the environment"
    vf_coef = 0.5  # type:  "loss weighing coefficient for the value function loss. with the VPG loss being 1.0"
    ent_coef = 0.0  # type:  "PPO entropy coefficient"
    max_grad_norm = 0.5  # type:  "PPO maximum gradient norm"
    clip_range = 0.2  # type:  "PPO clip_range parameter"
    # GAE runner options
    gamma = 0.99  # type:  "GAE gamma"
    lam = 0.95  # type:  "GAE lambda"
    # plotting settings
    dpi = 150  # type:  "dpi for plot output"
    # Grid World config parameters
    change_colors = 0  # type:  "shuffle colors of the board game"
    change_dynamics = 0  # type:  'shuffle control actions (up down, left right) of the game'


@cli_parse
class Reporting:
    report_mean = False  # type:  "plot the mean instead of the total reward per episode"

    plot_server = "http://slab-krypton.uchicago.edu"  # type: "server url, need to include protocol [http(s)://]."
    plot_server_port = 8097  # type: "port for the visdom server"

    plot_interval = 10  # type: "plotting batch size"
    plot_smoothing = 20  # type: "smoothing factor"
    save_interval = 0  # type: "plotting batch size"


@cli_parse
class DEBUG:
    """To debug:
    Set debug_params = 1,
    set debug_apply_gradient = 1.
    Then the gradient ratios between the worker and the meta runner should be print out, and they should be 1.
    Otherwise, the runner model is diverging from the meta network.
    """
    no_weight_reset = 0  # type:  "flag to turn off the caching and resetting the weights"
    no_maml_apply_gradient = 0  # type:  "by-pass maml gradient"
    no_task_resample = 0  # type:  "by-pass task re-sample"
    debug_params = 0  # type:  'get the param values for debug purpose'
    debug_apply_gradient = 0  # type:  'use alternative SGD implementation'

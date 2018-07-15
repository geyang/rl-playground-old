"""stands for 'correction-MAML. Could also argue complete-maml. Whatever."""
import matplotlib
from tqdm import trange
import tensorflow as tf

from .config import G, RUN
from .vpg import Inputs as VPGInputs, VPG, Optimize as VPG_Optimize
from .ppo import Inputs as PPOInputs, PPO, Optimize as PPO_Optimize
from .ge_utils import defaultlist, make_with_custom_variables, GradientSum, Cache

matplotlib.use("Agg")

import baselines.common.tf_util as U
from .ge_policies import MlpPolicy

ALLOWED_ALGS = ('VPG', 'PPO')


class Meta:
    def __init__(self, *, scope_name, act_space, ob_shape, algo, reuse=False, trainables=None, optimizer=None,
                 add_loss=None, loss_only=False):
        assert algo in ALLOWED_ALGS, "model algorithm need to be one of {}".format(ALLOWED_ALGS)
        with tf.variable_scope(scope_name, reuse=False):
            self.obs = obs = tf.placeholder(dtype=tf.float32, shape=ob_shape, name='obs')  # obs
            if algo == "PPO":
                self.inputs = inputs = PPOInputs(action_space=act_space)
                Optimize = PPO_Optimize
            elif algo == "VPG":
                self.inputs = inputs = VPGInputs(action_space=act_space)
                Optimize = VPG_Optimize
            else:
                raise NotImplementedError('Only supports PPO and VPG')
            inputs.X = obs
            self.policy = policy = MlpPolicy(ac_space=act_space, X=obs, reuse=reuse)

            # note that policy.trainables are the original trainable parameters, not the mocked variables.
            if trainables is None:
                trainables = policy.trainables
            self.trainables = trainables

            ext_loss = add_loss(inputs.ADV) if callable(add_loss) else None
            if algo == "PPO":
                self.model = PPO(inputs=inputs, policy=policy, vf_coef=G.vf_coef, ent_coef=G.ent_coef)
            elif algo == "VPG":
                self.model = VPG(inputs=inputs, policy=policy, vf_coef=G.vf_coef)

            self.loss = self.model.loss if ext_loss is None else (self.model.loss + ext_loss)

            if loss_only:
                self.optim = None
            else:
                inputs.LR = tf.placeholder(tf.float32, [], name="LR")
                self.optim = Optimize(lr=inputs.LR, loss=self.loss, trainables=trainables,
                                      max_grad_norm=G.max_grad_norm, optimizer=optimizer)


def cmaml_loss(neglogpacs, advantage):
    #  add in correction term.
    mean_adv = U.mean(advantage)
    exploration_term = U.mean(neglogpacs) * mean_adv
    return exploration_term


class SingleTask:
    def __init__(self, act_space, ob_shape, trainables):
        # no need to go beyond despite of large G.eval_grad_steps, b/c RL samples using runner policy.

        self.workers = defaultlist(None)

        params = defaultlist(None)
        params[0] = trainables
        for k in range(G.n_grad_steps + 1):
            if k < G.n_grad_steps:  # 0 - 9,
                self.workers[k] = worker = make_with_custom_variables(
                    lambda: Meta(scope_name='inner_{grad_step}_grad_network'.format(grad_step=k),
                                 act_space=act_space, ob_shape=ob_shape, algo=G.inner_alg,
                                 optimizer=G.inner_optimizer, reuse=True, trainables=params[-1]
                                 ),  # pass in the trainables for proper gradient
                    params[-1]
                )
                with tf.variable_scope('SGD_grad_{}'.format(k)):
                    if G.first_order:
                        params[k + 1] = [worker.optim.apply_grad(grad=tf.stop_gradient(g), var=v) for g, v in
                                         zip(worker.optim.grads, params[-1])]
                    else:
                        params[k + 1] = [worker.optim.apply_grad(grad=g, var=v) for g, v in
                                         zip(worker.optim.grads, params[-1])]

            if k == G.n_grad_steps:  # 10 or 1.
                loss_fn = lambda ADV: cmaml_loss([w.model.neglogpac for w in self.workers], ADV) \
                    if G.run_mode == "e_maml" else None
                self.meta = make_with_custom_variables(
                    lambda: Meta(scope_name="meta_network", act_space=act_space, ob_shape=ob_shape,
                                 algo=G.meta_alg, reuse=True, add_loss=loss_fn, loss_only=True),
                    params[-1]
                )

        # Expose as non-public API for debugging purposes
        self._params = params


# Algorithm Summary
# 1. [sample] with pi(theta) `run_episode`
# 2. compute policy gradient (vanilla)
# 3. apply gradient to get \theta' using SGD
# 4. [sample] with pi(theta') `run_episode`
# 5. use PPO, compute meta gradient
# 6. sum up the PPO gradient from multiple tasks and average
# 6. apply this gradient
class E_MAML:
    def __init__(self, ob_space, act_space):
        """
        Usage:
            self.env = env
            ob_shape = (None,) + self.env.observation_space.shape
        """
        ob_shape = (None,) + ob_space.shape

        # Meta holds policy, inner optimizer
        self.runner = Meta(scope_name='runner', act_space=act_space, ob_shape=ob_shape, algo=G.inner_alg,
                           optimizer=G.inner_optimizer)
        trainables = self.runner.policy.trainables

        self.beta = tf.placeholder(tf.float32, [], name="beta")

        print(">>>>>>>>>>> Constructing Meta Graph <<<<<<<<<<<")
        # todo: we can do multi-GPU placement of the graph here.
        self.graphs = []
        for t in trange(G.n_graphs):
            with tf.variable_scope(f"graph_{t}"):
                task_graph = SingleTask(act_space=act_space, ob_shape=ob_shape, trainables=trainables)
                self.graphs.append(task_graph)

                with tf.variable_scope('meta_optimizer'):
                    assert G.n_graphs == 1, "hard code the n_graphs to 1 b/c we don't do multi-device placement yet."
                    # call gradient_sum.set_op first, then add_op. Call k times in-total.

                    # do NOT apply gradient norm here.
                    # if G.meta_alg == "PPO":
                    #     Optimize = PPO_Optimize
                    # elif G.meta_alg == "VPG":
                    #     Optimize = VPG_Optimize
                    # meta_optim = Optimize(lr=self.beta, loss=task_graph.meta.loss, trainables=trainables,
                    #                       max_grad_norm=None, optimizer=G.meta_optimizer)
                    # self.meta_grads = meta_optim.grads
                    # self.gradient_sum = GradientSum(trainables, self.meta_grads)
                    # # question: apply grad norm before or after?
                    # grads = [c / G.n_tasks for c in self.gradient_sum.cache]
                    # if G.max_grad_norm:
                    #     self.grads, grad_norm = tf.clip_by_global_norm(grads, G.max_grad_norm)
                    # self.meta_optimize_op = [meta_optim.apply_grad(grad=g, var=v) for v, g in zip(grads, trainables)]

                    # mistake. single task?
                    self.meta_grads = tf.gradients(task_graph.meta.loss, trainables)
                    self.gradient_sum = GradientSum(trainables, self.meta_grads)
                    grads = [c / G.n_tasks for c in self.gradient_sum.cache]

                    if G.max_grad_norm:  # allow 0 to be by-pass
                        grads = [g * tf.stop_gradient(G.max_grad_norm / tf.maximum(G.max_grad_norm, tf.norm(g))) for g
                                 in grads]

                    # do NOT apply gradient norm here.
                    if G.meta_optimizer == "Adam":
                        Optim = tf.train.AdamOptimizer
                    elif G.meta_optimizer == "SGD":
                        Optim = tf.train.GradientDescentOptimizer
                    self.meta_optimizer = Optim(learning_rate=self.beta)
                    self.meta_update_op = self.meta_optimizer.apply_gradients(zip(grads, trainables))

                # Only do this after the meta graph has finished using policy.trainables
                # Note: stateful operators for saving to a cache and loading from it. Only used to reset runner
                # Note: Slots are not supported. Only weights.
                with tf.variable_scope("weight_cache"):
                    self.cache = Cache(trainables)
                    self.save_checkpoint = U.function([], [self.cache.save])
                    self.load_checkpoint = U.function([], [self.cache.load])

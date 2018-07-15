import tensorflow as tf
from gym import spaces

import baselines.common.tf_util as U
from .config import RUN, DEBUG, G

# Here we use a input class to make it easy to define defaults.
from .ge_utils import placeholders_from_variables


class Inputs:
    def __init__(self, *, action_space, A=None, ADV=None, R=None, LR=None):
        # self.X = X or tf.placeholder(tf.float32, [None], name="obs")
        if isinstance(action_space, spaces.Discrete):
            self.A = A or tf.placeholder(tf.int32, [None], name="A")
        else:
            self.A = A or tf.placeholder(tf.float32, [None] + list(action_space.shape), name="A")

        self.ADV = ADV or tf.placeholder(tf.float32, [None], name="ADV")
        self.R = R or tf.placeholder(tf.float32, [None], name="R")


class VPG:
    def __init__(self, *, inputs, policy, vf_coef):
        self.inputs = inputs
        self.policy = policy
        with tf.variable_scope("VPG"):
            self.neglogpac = policy.pd.neglogp(inputs.A)
            self.vf_loss = tf.square(policy.vf - inputs.R)
            self.vpg_loss = tf.reduce_mean(inputs.ADV * self.neglogpac)
            self.loss = self.vpg_loss + self.vf_loss * vf_coef  # <== this is the value function loss ratio.


class Optimize(object):
    def __init__(self, *, lr, loss, trainables, max_grad_norm=None, optimizer="SGD", **_):
        """
        :param trainables: Optional array used for the gradient calculation
        """
        with tf.variable_scope('VPG_Optimize'):
            grad_placeholders = placeholders_from_variables(trainables)
            # optimizer.gradients is just a wrapper around tf.gradients, with extra assertions. This is why it raises
            # errors on non-trainables.
            _grads = tf.gradients(loss, trainables)
            assert _grads[0] is not None, 'Grads are not defined'

            if max_grad_norm:  # allow 0 to be by-pass
                # print('setting max-grad-norm to', max_grad_norm)
                # tf.clip_by_global_norm is just fine. No need to use my own.
                _grads = [g * tf.stop_gradient(max_grad_norm / tf.maximum(max_grad_norm, tf.norm(g))) for g in _grads]
                # _grads, grad_norm = tf.clip_by_global_norm(_grads, max_grad_norm)

            self.grads = _grads

            # if trainables and hasattr(trainables[0], '_variable'):
            #     if optimizer == "Adam":
            #         self.optimizer = tf.train.AdamOptimizer(learning_rate=lr)
            #     elif optimizer == 'SGD':
            #         self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
            #     else:
            #         self.optimizer = optimizer
            #
            #     optimize_op = self.optimizer.apply_gradients(zip(_grads, trainables))
            #     apply_grads_op = self.optimizer.apply_gradients(zip(grad_placeholders, trainables))
            #
            #     def run_optimize(*, feed_dict):
            #         assert lr in feed_dict, 'feed_dict need to contain learning rate.'
            #         return tf.get_default_session().run([_grads, optimize_op], feed_dict)[0]
            #
            #     def run_apply_grads(*, grads, lr):
            #         """function that applies the gradients to the weights"""
            #         feed_dict = {p: g for p, g in zip(grad_placeholders, grads)}
            #         feed_dict[lr] = lr
            #         return tf.get_default_session().run(apply_grads_op, feed_dict=feed_dict)
            #
            #     self.run_optimize = run_optimize
            #     self.run_apply_grads = run_apply_grads

            # beta = tf.get_variable('RMSProp_beta')
            # avg_grad = tf.get_variable('RMSProp_avg_g')
            # avg_grad = beta * avg_grad + (1 - beta) * grad
            # graph operator for updating the parameter. used by maml with the SGD inner step
            self.apply_grad = lambda *, grad, var: var - lr * grad
            self.optimize = [v.assign(self.apply_grad(grad=g, var=v)) for v, g in zip(trainables, self.grads)]
            self.run_optimize = lambda feed_dict: tf.get_default_session().run(self.optimize, feed_dict=feed_dict)

        # Function to compute the PPO gradients
        self.run_grads = lambda *, feed_dict: tf.get_default_session().run([_grads], feed_dict)


def path_to_feed_dict(*, inputs: Inputs, paths, lr=None, **_r):
    # add linear feature baselines
    if G.baseline == 'linear':
        from playground.maml.maml_tf.value_baselines.linear_feature_baseline import LinearFeatureBaseline
        baseline = LinearFeatureBaseline()
        baseline.fit(paths)
        values = baseline.predict(paths)
    elif G.baseline == 'critic':
        values = paths['values']

    advs = paths['returns'] - values
    advs_normalized = (advs - advs.mean()) / (advs.std() + 1e-8)

    n_timesteps, n_envs, *_ = paths['obs'].shape
    n = n_timesteps * n_envs

    feed_dict = {
        inputs.X: paths['obs'].reshape(n, -1),
        inputs.A: paths['actions'].reshape(n, -1),
        inputs.ADV: advs_normalized.reshape(-1),
        inputs.R: paths['returns'].reshape(-1)
    }
    if lr is not None:
        feed_dict[inputs.LR] = lr
    return feed_dict

import tensorflow as tf
from gym import spaces

import baselines.common.tf_util as U
from .config import RUN, DEBUG

# Here we use a input class to make it easy to define defaults.
from .ge_utils import placeholders_from_variables


class Inputs:
    def __init__(self, *, action_space, A=None, ADV=None, v_targs=None, LR=None):
        # self.X = X or tf.placeholder(tf.float32, [None], name="obs")
        self.R = v_targs or tf.placeholder(tf.float32, [None], name="v_targs")
        self.ADV = ADV or tf.placeholder(tf.float32, [None], name="ADV")
        if isinstance(action_space, spaces.Discrete):
            self.A = A or tf.placeholder(tf.int32, [None], name="A")
        else:
            self.A = A or tf.placeholder(tf.float32, [None] + list(action_space.shape), name="A")


class VPG:
    def __init__(self, *, inputs, policy, vf_coef):
        self.inputs = inputs
        self.policy = policy
        with tf.variable_scope("VPG_loss"):
            self.vf_loss = tf.square(inputs.R - policy.vf)
            self.neglogpac = policy.pd.neglogp(inputs.A)
            self.vpg_loss = vpg_loss = tf.reduce_mean(inputs.ADV * self.neglogpac)
            self.loss = vpg_loss + self.vf_loss * vf_coef  # <== this is the value function loss ratio.


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
                # _grads = [g * max_grad_norm / tf.maximum(max_grad_norm, tf.norm(g)) for g in _grads]
                _grads, grad_norm = tf.clip_by_global_norm(_grads, max_grad_norm)

            self.grads = _grads

            if trainables and hasattr(trainables[0], '_variable'):
                if optimizer == "Adam":
                    optimizer = tf.train.AdamOptimizer(learning_rate=lr)
                elif optimizer == 'SGD':
                    optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)

                optimize_op = optimizer.apply_gradients(zip(_grads, trainables))
                apply_grads_op = optimizer.apply_gradients(zip(grad_placeholders, trainables))

                def run_optimize(*, feed_dict):
                    assert lr in feed_dict, 'feed_dict need to contain learning rate.'
                    return tf.get_default_session().run(optimize_op, feed_dict)

                def run_apply_grads(*, grads, lr):
                    feed_dict = {p: g for p, g in zip(grad_placeholders, grads)}
                    feed_dict[lr] = lr
                    return tf.get_default_session().run(apply_grads_op, feed_dict=feed_dict)

                self.run_optimize = run_optimize
                self.run_apply_grads = run_apply_grads

        def apply_grad(*, grad, var):
            # Note: used by MAML
            return var - lr * grad

        self.apply_grad = apply_grad

        # note: debug and verification only.
        if DEBUG.debug_apply_gradient and optimizer == "SGD":
            # the tf.SGD is tested to be identical to this following apply gradient implementation. Kept here for
            # reference.
            with tf.variable_scope('SGD'):
                try:
                    optimize_op = tf.group(
                        *[tf.assign(t, apply_grad(grad=g, var=t)) for g, t in zip(_grads, trainables)])
                    apply_grads_op = tf.group(
                        *[tf.assign(t, apply_grad(grad=p, var=t)) for p, t in zip(grad_placeholders, trainables)])
                except:
                    print('trainables are not trainable variables.')

        def run_grads(*, feed_dict):
            """
            Function to compute the PPO gradients
            :param feed_dict:
            :return: grads, pg_loss, vf_loss, entropy, approxkl, clipfrac
            """
            return tf.get_default_session().run(
                [_grads, vf_loss],
                feed_dict
            )

        self.run_grads = run_grads


def path_to_feed_dict(*, inputs: Inputs, paths, lr=None, **_r):
    advs = paths['returns'] - paths['values']
    advs_normalized = (advs - advs.mean()) / (advs.std() + 1e-8)
    feed_dict = {
        inputs.X: paths['obs'],
        inputs.A: paths['actions'],
        inputs.ADV: advs_normalized,
        inputs.R: paths['returns'],
    }
    if lr is not None:
        feed_dict[inputs.LR] = lr
    return feed_dict

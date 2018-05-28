from pprint import pformat

import tensorflow as tf
from gym import spaces

import baselines.common.tf_util as U
from .config import RUN, DEBUG

# best way to define the input interface is to use a named_tuple and then others could just import the tuple from here:
# https://pymotw.com/2/collections/namedtuple.html
# InputT = namedtuple("Inputs", 'A ADV R OLD_NEG_LOG_P_AC OLD_V_PRED CLIP_RANGE X_act X_train')


# Here we use a input class to make it easy to define defaults.
from .ge_utils import placeholders_from_variables


class Inputs:
    def __init__(self, *, action_space, A=None, ADV=None, R=None, OLD_NEG_LOG_PAC=None, OLD_V_PRED=None,
                 CLIP_RANGE=None, LR=None):
        # self.v_targs = v_targs or tf.placeholder(tf.float32, [None], name="v_targs")
        if isinstance(action_space, spaces.Discrete):
            self.A = A or tf.placeholder(tf.int32, [None], name="A")
        else:
            self.A = A or tf.placeholder(tf.float32, [None] + list(action_space.shape), name="A")
        self.ADV = ADV or tf.placeholder(tf.float32, [None], name="ADV")
        self.R = R or tf.placeholder(tf.float32, [None], name="R")
        self.OLD_NEG_LOG_P_AC = OLD_NEG_LOG_PAC or tf.placeholder(tf.float32, [None], name="OLD_NEG_LOG_P_AC")
        self.OLD_V_PRED = OLD_V_PRED or tf.placeholder(tf.float32, [None], name="OLD_V_PRED")
        self.CLIP_RANGE = CLIP_RANGE or tf.placeholder(tf.float32, [], name="CLIP_RANGE")


class PPO:
    def __init__(self, *, inputs, policy, vf_coef, ent_coef):
        self.inputs = inputs
        self.policy = policy
        with tf.variable_scope('PPO'):
            self.neglogpac = policy.pd.neglogp(inputs.A)
            entropy = tf.reduce_mean(policy.pd.entropy())

            vpred = policy.vf
            vpred_clipped = inputs.OLD_V_PRED + \
                            tf.clip_by_value(policy.vf - inputs.OLD_V_PRED, - inputs.CLIP_RANGE, inputs.CLIP_RANGE)
            vf_losses1 = tf.square(vpred - inputs.R)
            vf_losses2 = tf.square(vpred_clipped - inputs.R)
            vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))
            ratio = tf.exp(inputs.OLD_NEG_LOG_P_AC - self.neglogpac)
            pg_losses = -inputs.ADV * ratio
            pg_losses2 = -inputs.ADV * tf.clip_by_value(ratio, 1.0 - inputs.CLIP_RANGE, 1.0 + inputs.CLIP_RANGE)
            pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
            # note: these are somehow not used.
            self.approxkl = .5 * tf.reduce_mean(tf.square(self.neglogpac - inputs.OLD_NEG_LOG_P_AC))
            self.clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), inputs.CLIP_RANGE)))
            # note: get_cmaml_loss(wp, mp)
            self.loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef


class Optimize(object):
    def __init__(self, *, lr, loss, trainables, max_grad_norm=None, optimizer="SGD", **_):
        """
        :param trainables: Optional array used for the gradient calculation
        """
        with tf.variable_scope('PPO_Optimize'):
            grad_placeholders = placeholders_from_variables(trainables)
            # optimizer.gradients is just a wrapper around tf.gradients, with extra assertions. This is why it raises
            # errors on non-trainables.
            _grads = tf.gradients(loss, trainables)

            for g in _grads:
                if g is None:
                    print(g)
                assert g is not None, "Grad is {}. It shouldn't be this case.\n{}".format(g, pformat(_grads))

            if max_grad_norm:  # allow 0 to be by-pass
                # print('setting max-grad-norm to', max_grad_norm)
                # tf.clip_by_global_norm is just fine. No need to use my own.
                # _grads = [g * max_grad_norm / tf.maximum(max_grad_norm, tf.norm(g)) for g in _grads]
                _grads, grad_norm = tf.clip_by_global_norm(_grads, max_grad_norm)

            self.grads = _grads

            if trainables and hasattr(trainables[0], '_variable'):
                if optimizer == "Adam":
                    print('WARNING: Adam is NOT supported as MAML runner at the moment,', end=" ")
                    print('but FINE as the Meta Learner.')
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


def path_to_feed_dict(*, inputs: Inputs, paths, lr=None, clip_range, **_r):
    advs = paths['returns'] - paths['values']
    advs_normalized = (advs - advs.mean()) / (advs.std() + 1e-8)
    feed_dict = {
        inputs.X: paths['obs'],
        inputs.A: paths['actions'],
        inputs.ADV: advs_normalized,
        inputs.OLD_NEG_LOG_P_AC: paths['neglogpacs'],
        inputs.OLD_V_PRED: paths['values'],
        inputs.R: paths['returns'],
        inputs.CLIP_RANGE: clip_range
    }
    if lr is not None:
        feed_dict[inputs.LR] = lr
    return feed_dict

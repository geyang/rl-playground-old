import numpy as np
import tensorflow as tf
from gym import spaces

from baselines.a2c.utils import ortho_init
from baselines.common.distributions import make_pdtype
from .ge_utils import placeholders_from_variables
from . import config


def fc(x, scope, nh, act=tf.nn.relu, init_scale=1.0):
    with tf.variable_scope(scope):
        nin = x.get_shape()[1].value
        w = tf.get_variable("w", [nin, nh], initializer=ortho_init(init_scale))
        b = tf.get_variable("b", [nh], initializer=tf.constant_initializer(0.0))
        z = tf.matmul(x, w)+b
        h = act(z)
        return h


class MlpPolicy(object):

    def __repr__(self):
        return "{self.__class__} {self.name}".format(self=self)

    def __init__(self, ac_space, X, scope='MlpPolicy', reuse=False):
        if isinstance(scope, tf.VariableScope):
            self.scope_name = scope.name
        else:
            self.scope_name = scope
        self.name = (self.scope_name + "_reuse") if reuse else self.scope_name

        self.X = X

        # done: this only applies to Discrete action space. Need to make more general.
        # now it works for both discrete action and gaussian policies.
        if isinstance(ac_space, spaces.Discrete):
            actdim = ac_space.n
        else:
            actdim = ac_space.shape[0]

        hidden_size = config.G.hidden_size
        if config.G.activation == 'tanh':
            act = tf.tanh
        elif config.G.activation == "relu":
            act = tf.nn.relu
        else:
            raise TypeError(f"{config.G.activation} is not available in this MLP.")
        with tf.variable_scope(scope, reuse=reuse):
            h1 = fc(X, 'pi_fc1', nh=hidden_size, init_scale=np.sqrt(2), act=act)
            h2 = fc(h1, 'pi_fc2', nh=hidden_size, init_scale=np.sqrt(2), act=act)
            pi = fc(h2, 'pi', actdim, act=lambda x: x, init_scale=0.01)
            h1 = fc(X, 'vf_fc1', nh=hidden_size, init_scale=np.sqrt(2), act=act)
            h2 = fc(h1, 'vf_fc2', nh=hidden_size, init_scale=np.sqrt(2), act=act)
            vf = fc(h2, 'vf', 1, act=lambda x: x)[:, 0]

            if isinstance(ac_space, spaces.Box):  # gaussian policy requires logstd
                logstd = tf.get_variable(name="logstd", shape=[1, actdim], initializer=tf.zeros_initializer())
                # GaussianPd takes 2 * [act_length] b/c of the logstd concatenation.
                pi = tf.concat([pi, pi * 0.0 + logstd], axis=1)

            # list of parameters is fixed at graph time.
            self.trainables = tf.trainable_variables()

            placeholders = placeholders_from_variables(self.trainables)
            self._assign_placeholder_dict = {t.name: p for t, p in zip(self.trainables, placeholders)}
            self._assign_op = tf.group(*[v.assign(p) for v, p in zip(self.trainables, placeholders)])

        with tf.variable_scope("Gaussian_Action"):
            self.pdtype = make_pdtype(ac_space)
            self.pd = self.pdtype.pdfromflat(pi)

            self.a = a = self.pd.sample()
            self.neglogpac = self.pd.neglogp(a)

        self.pi = pi
        self.vf = vf

    @property
    def state_dict(self):
        # todo: should make the tensor names scoped locally.
        return {t.name: v for t, v in zip(self.trainables, tf.get_default_session().run(self.trainables))}

    def load_from_state_dict(self, state_dict):
        # todo: this adds new assign ops each time, and causes the graph to grow.
        feed_dict = {self._assign_placeholder_dict[t.name]: state_dict[t.name] for t in self.trainables}
        return tf.get_default_session().run(self._assign_op, feed_dict=feed_dict)

    def step(self, ob, feed_dict=None):
        if feed_dict:
            feed_dict.update({self.X: ob})
        else:
            feed_dict = {self.X: ob}
        sess = tf.get_default_session()
        return sess.run([self.a, self.vf, self.neglogpac], feed_dict=feed_dict)

    def value(self, ob, feed_dict=None):
        if feed_dict:
            feed_dict.update({self.X: ob})
        else:
            feed_dict = {self.X: ob}
        sess = tf.get_default_session()
        return sess.run(self.vf, feed_dict=feed_dict)

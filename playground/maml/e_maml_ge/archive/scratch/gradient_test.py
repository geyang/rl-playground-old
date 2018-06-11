import tensorflow as tf
import baselines.common.tf_util as U


def _eval(p, feed_dict={}):
    return tf.get_default_session().run(p, feed_dict=feed_dict)


with U.single_threaded_session():
    x = tf.get_variable('x', (1, 3), tf.float32)
    y = x * 5
    g = tf.gradients(y, x)
    print(_eval(g))

    z = y * 5
    g_zy = tf.gradients(z, y)
    g_zx = tf.gradients(z, x)
    # shows that tf.gradient correctly gives the intermediate gradient
    print(_eval(g_zy))
    # this is the gradient w.r.t the trainables.
    print(_eval(g_zx))

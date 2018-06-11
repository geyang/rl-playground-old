import tensorflow as tf
from moleskin import moleskin as M

sess = tf.InteractiveSession()


def serial(*statements):
    p = None
    ps = []
    for i, s in enumerate(statements):
        if p is not None:
            with tf.control_dependencies([p]):
                p = s()
        else:
            p = s()
        ps.append(p)
    return tf.group(*ps)


def make_flip_flop():
    a = tf.get_variable('a', shape=(), dtype=tf.float32, initializer=tf.zeros_initializer())
    b = tf.get_variable('b', shape=(), dtype=tf.float32, initializer=tf.ones_initializer())
    t = tf.get_variable('t', shape=(), dtype=tf.float32, initializer=tf.zeros_initializer())
    sess.run(tf.global_variables_initializer())

    a2t = lambda: t.assign(a)
    t2a = lambda: a.assign(t)
    b2t = lambda: t.assign(a)
    t2b = lambda: b.assign(t)
    b2a = lambda: a.assign(b)
    a2b = lambda: b.assign(a)

    unreliable_swap_op = tf.group(a2t(), b2a(), t2b())
    good_swap_op = serial(a2t, b2a, t2b)

    # M.debug(sess.run((a, b, t)))
    # sess.run(a2t)
    # M.debug(sess.run([a, b, t]))
    # sess.run(b2a)
    # M.debug(sess.run([a, b, t]))
    # sess.run(t2b)
    # M.debug(sess.run([a, b, t]))
    #
    # sess.run(unreliable_swap_op)
    # M.debug(sess.run([a, b, t]))

    sess.run(good_swap_op)
    M.debug(sess.run([a, b, t]))

    @M.timeit
    def run(n=10000):
        for i in range(n):
            sess.run(good_swap_op)
            # sess.run(a2t)
            # sess.run(b2a)
            # sess.run(t2b)

    return run


run = make_flip_flop()
run()

# 146.93 ms / run, 48.97 ms / sess.run.

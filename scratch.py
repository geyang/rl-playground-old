if __name__ == "__main__":

    import tensorflow as tf

    x = tf.Variable(1)
    y = tf.Variable(2)

    print(f"x.name: {x.name}, y.name: {y.name}")

    a = tf.Variable(1)
    b = tf.Variable(2)

    # override a, otherwise a content is 1
    as_op = a.assign(b)
    print(f"a.name: {a.name}, b.name: {b.name}")

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        print(sess.run([x, y, as_op, a, b]))

        tf.au

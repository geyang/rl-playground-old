
if __name__=="__main__":
    from subprocess import check_output
    r = check_output(['nvidia-smi'])
    print(r.decode('ascii'))

    from params_proto import cli_parse
    
    @cli_parse
    class G:
        a = 10

    import tensorflow as tf
    print('this is vorking!')

    with tf.device('/device:GPU:0'):
        a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
        b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    c = tf.matmul(a, b)

    sess_config = tf.ConfigProto(log_device_placement=True)
    with tf.Session(config=sess_config) as sess:
        print(sess.run(c))

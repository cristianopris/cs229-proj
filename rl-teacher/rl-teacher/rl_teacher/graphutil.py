import tensorflow as tf
from tensorflow.python.platform import gfile

def main_():
    with tf.Session() as sess:
        model_filename ='/Users/cristian/dev/cs229-proj/bin/old/two_paddle_hum_2017-11-19-222723184491/204791/pposgd_policy_graph.pb'
        with gfile.FastGFile(model_filename, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            g_in = tf.import_graph_def(graph_def)
    LOGDIR='tmp'
    train_writer = tf.summary.FileWriter(LOGDIR)
    train_writer.add_graph(sess.graph)

def main():
    from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
    print_tensors_in_checkpoint_file(file_name='/Users/cristian/dev/cs229-proj/bin/old/two_paddle_rl_2017-11-20-013306620579/153600', tensor_name='', all_tensors=False)


if __name__ == '__main__':
    main()

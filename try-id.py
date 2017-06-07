from __future__ import absolute_import
from __future__ import division

import argparse
import sys
import os

import tensorflow as tf
slim = tf.contrib.slim
import numpy as np
import cv2

from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile

#from timer import Timer
from mk_utils import MkLoader

input_name = 'input:0'
bottleneck_names = ['MobileNet/conv_ds_6/pw_batch_norm/Relu:0',
                    'MobileNet/conv_ds_12/pw_batch_norm/Relu:0',
                    'MobileNet/conv_ds_14/pw_batch_norm/Relu:0',
                    'SSD_1/feat_0/f_dwc/pc/Elu:0',
                    'SSD_1/feat_1/f_dwc/pc/Elu:0',
                    'SSD_1/feat_2/f_dwc/pc/Elu:0']


## Directory Setup
def get_dir(*args):
    d = os.path.join(*args)
    if not os.path.exists(d):
        os.makedirs(d)
    return d

data_root = get_dir('data', 'id')
bottleneck_root = get_dir(data_root, 'btl')
log_root = get_dir('/tmp', 'id_logs')

run_id = 'run_%02d' % len(os.walk(log_root).next()[1])
run_log_root = os.path.join(log_root, run_id)

output_root = get_dir(data_root, run_id)
output_graph_path = os.path.join(output_root, 'output_graph.pb')

## Learning Configs
batch_size = 128
learning_rate = 1e-3
train_steps = int(10e3)
test_steps = 10

parser = argparse.ArgumentParser()

parser.add_argument(
    '--graph',
    required=False,
    type=str,
    default='data/train/4/output_graph.pb',
    help='Absolute path to graph file (.pb)')

def load_image(filename):
    return cv2.imread(filename)[...,::-1]/255.0 
    #return tf.gfile.FastGFile(filename, 'rb').read()

def load_labels(filename):
    """Read in labels, one label per line."""
    return [line.rstrip() for line in tf.gfile.GFile(filename)]

def load_graph(filename):
    """Unpersists graph from file as default graph."""
    with tf.Graph().as_default() as graph:
        with tf.gfile.FastGFile(filename, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            return_elements = [input_name] + bottleneck_names
            tensors = tf.import_graph_def(graph_def, name='', return_elements=return_elements)
    return graph, tensors[0], tensors[1:]

def report_graph(graph):
    for op in graph.get_operations():
       print('===')
       print(op.name)
       print('Input:')
       for i in op.inputs:
           print('\t %s' % i.name, i.get_shape())
       print('Output:')
       for o in op.outputs:
           print('\t %s' % o.name, o.get_shape())
       print('===')

def id_ops(p1_t, p2_t, label_t, is_training, reuse=None):
    with tf.variable_scope('ID', reuse=reuse):
        #with slim.arg_scope([slim.fully_connected],
        #        activation_fn=tf.nn.elu,
        #        weights_regularizer=slim.l2_regularizer(5e-4)):
        #    l_p1 = slim.fully_connected(p1_t, 2, activation_fn=None, scope='fc') # try first with shallow network
        #    print 'ls', l_p1.shape
        #    pred_t = tf.nn.softmax(l_p1)
        #    print 'ps', pred.shape
        #    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels=label_t)

        d = p1_t.get_shape().as_list()[-1] # feature depth

        with tf.name_scope('pred'):
            #alpha = tf.get_variable('alpha', shape = (1,d), dtype=tf.float32, initializer=tf.constant_initializer(1./d)) # feature weights
            #bias = tf.get_variable('bias', shape = (1,d), dtype=tf.float32, initializer=tf.constant_initializer(0.))
            #logits = alpha * tf.abs(p1_t - p2_t) + bias
            with slim.arg_scope([slim.fully_connected],
                    activation_fn=tf.nn.elu,
                    weights_regularizer=slim.l2_regularizer(5e-4),
                    normalizer_fn = slim.batch_norm,
                    normalizer_params={
                    'is_training' : is_training,
                    'decay' : 0.9,
                    'fused' : True,
                    'reuse' : reuse,
                    'scope' : 'BN'
                    } 
                    ):
                logits = p1_t-p2_t # opt2 : diff
                logits = slim.fully_connected(logits, 256, scope='fc_1')
                logits = slim.fully_connected(logits, 64, scope='fc_2')
                logits = slim.dropout(logits, keep_prob=0.5, is_training=is_training, scope='do_1')
                logits = slim.fully_connected(logits, 2, activation_fn=None, scope='fc_3') # final

            #logits = tf.reduce_mean(logits, axis=-1)
            pred = tf.nn.softmax(logits, name='pred')

            tf.summary.histogram('logits', logits)
            #tf.summary.histogram('alpha', alpha)

        with tf.name_scope('loss'):
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_t, logits=logits)
            #loss = tf.log(1 + tf.exp(-(tf.cast(label_t, tf.float32) - 0.5) * pred))
            #loss = tf.losses.mean_squared_error(labels=tf.cast(label_t,tf.float32), predictions=pred)
            #loss = -tf.where(label_t, tf.log(pred),tf.log(1-pred+1e-6))
            loss = tf.reduce_mean(loss, -1)
            tf.summary.scalar('loss', loss)
            tf.losses.add_loss(loss)

        with tf.name_scope('eval'):
            correct = tf.equal(tf.cast(tf.argmax(pred,axis=-1),tf.int32), label_t)
            acc = tf.reduce_mean(tf.cast(correct, tf.float32), name='acc')
            tf.summary.scalar('accuracy', acc)

    return pred, acc

def basename(s):
    return os.path.splitext(os.path.basename(s))[0]

def get_or_create_bottlenecks(sess, batch, feature_tensor, image_tensor):
    b1s = []
    b2s = []
    ls = []

    for (p1,p2,l) in batch:
        b1_f = os.path.join(bottleneck_root, basename(p1) + '_btl.npy')
        b2_f = os.path.join(bottleneck_root, basename(p2) + '_btl.npy')

        if not os.path.exists(b1_f):
            b1 = sess.run(feature_tensor, feed_dict={image_tensor : loader.get_img(p1)})
            np.save(b1_f, b1, allow_pickle=True)
        if not os.path.exists(b2_f):
            b2 = sess.run(feature_tensor, feed_dict={image_tensor : loader.get_img(p2)})
            np.save(b2_f, b2, allow_pickle=True)

        b1 = np.load(b1_f, allow_pickle=True)
        b2 = np.load(b2_f, allow_pickle=True)

        b1s.append(b1)
        b2s.append(b2)
        ls.append(l)
    return np.stack(b1s,0), np.stack(b2s,0), np.array(ls, dtype=np.int32)

def main(argv):
    """Runs inference on an image."""
    if argv[1:]:
        raise ValueError('Unused Command Line Args: %s' % argv[1:])

    if not tf.gfile.Exists(FLAGS.graph):
        tf.logging.fatal('graph file does not exist %s', FLAGS.graph)

    gpu_options = tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=0.3)
    config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
    loader = MkLoader('/home/yoonyoungcho/Downloads/Market-1501-v15.09.15/', split_ratio=1.0)
    graph, image_tensor, feature_tensors = load_graph(FLAGS.graph)

    with graph.as_default():
        for f_t in feature_tensors:
            f_t.set_shape([1] + f_t.get_shape().as_list()[1:])

        feature_tensors = [tf.reshape(tf.nn.avg_pool(t, ksize=[1]+t.get_shape().as_list()[1:3]+[1], strides=[1,1,1,1], padding='VALID'), [-1]) for t in feature_tensors]
        feature_tensor = tf.concat(feature_tensors, axis=0, name='feature_tensor')

        p1_t = tf.placeholder_with_default(tf.expand_dims(feature_tensor,0), shape=[None]+feature_tensor.get_shape().as_list(), name='p1')
        p2_t = tf.placeholder_with_default(tf.expand_dims(feature_tensor,0), shape=[None]+feature_tensor.get_shape().as_list(), name='p2')
        label_t = tf.placeholder(tf.int32, [None], name='label')
        is_training = tf.placeholder(tf.bool, [], name='is_training')

        pred, acc = id_ops(p1_t, p2_t, label_t, is_training, reuse=None,)
        loss = tf.reduce_mean(tf.losses.get_total_loss(),name='net_loss')
        tf.summary.scalar('net_loss', loss)

        with tf.name_scope('train'):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            train_vars = slim.get_trainable_variables(scope='ID')
            print 'train_vars', train_vars
            opt = tf.train.AdamOptimizer(learning_rate = learning_rate)
            with tf.control_dependencies(update_ops):
                opt = opt.minimize(loss, var_list=train_vars)

        merged = tf.summary.merge_all()

    
    with tf.Session(graph=graph, config=config) as sess:
        sess.run(tf.global_variables_initializer())

        train_writer = tf.summary.FileWriter(os.path.join(run_log_root, 'train'), sess.graph)
        valid_writer = tf.summary.FileWriter(os.path.join(run_log_root, 'valid'), sess.graph)

        for i in range(train_steps):
            batch = loader.get_batch(batch_size, transpose=True, as_img=False, type='train')
            b1s,b2s,ls = get_or_create_bottlenecks(sess, batch, feature_tensor, image_tensor)
            _, s = sess.run([opt,merged], feed_dict={p1_t:b1s, p2_t:b2s, label_t:ls, is_training:True})
            train_writer.add_summary(s, i)

            if i % 20 == 0:
                batch = loader.get_batch(batch_size, transpose=True, as_img=False, type='train')
                b1s,b2s,ls = get_or_create_bottlenecks(sess, batch, feature_tensor, image_tensor)
                p,l,a,s = sess.run([pred,loss,acc,merged], feed_dict={p1_t:b1s, p2_t:b2s, label_t:ls, is_training:False})
                valid_writer.add_summary(s, i)
                print('%d ) Loss : %.3f, Accuracy : %.2f' % (i, l, a))

        print ('saving %s' % pred.name)
        output_graph_def = graph_util.convert_variables_to_constants(
                sess, sess.graph.as_graph_def(), [pred.name[:-2]])
        with gfile.FastGFile(output_graph_path, 'wb') as f:
          f.write(output_graph_def.SerializeToString())


        for i in range(test_steps):
            batch = loader.get_batch(1, transpose=True, as_img=False, type='test')
            p1, p2, label = batch[0]
            b1s,b2s,ls = get_or_create_bottlenecks(sess, batch, feature_tensor, image_tensor)
            im1,im2 = loader.get_img(p1), loader.get_img(p2)
            im1,im2 = (im1[...,::-1]*255).astype(np.uint8), (im2[...,::-1]*255).astype(np.uint8)
            p,l,a,s = sess.run([pred,loss,acc,merged], feed_dict={p1_t:b1s, p2_t:b2s, label_t:ls, is_training:False})

            cv2.imshow('im1', im1)
            cv2.imshow('im2', im2)
            print 'true label : ' , label
            print 'predicted label : ' , np.argmax(p), p
            cv2.waitKey(0)

if __name__ == '__main__':
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=sys.argv[:1]+unparsed)

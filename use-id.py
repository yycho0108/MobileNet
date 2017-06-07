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
feature_name = 'feature_tensor:0'
pred_name = 'ID/pred/pred:0'

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

## Learning Configs
batch_size = 128
learning_rate = 1e-3
train_steps = int(5e3)
test_steps = 10

parser = argparse.ArgumentParser()

parser.add_argument(
    '--graph',
    required=False,
    type=str,
    default='data/id/run_08/output_graph.pb',
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
            return_elements = [input_name, feature_name, pred_name]
            tensors = tf.import_graph_def(graph_def, name='', return_elements=return_elements)
    return graph, tensors

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

    graph, (image_t, feature_t, pred_t) = load_graph(FLAGS.graph)
    report_graph(graph)

    with tf.Session(graph=graph, config=config) as sess:

        p1_t = graph.get_tensor_by_name('p1:0')
        p2_t = graph.get_tensor_by_name('p2:0')
        is_training = graph.get_tensor_by_name('is_training:0')

        while True:
            batch = loader.get_batch(1, transpose=True, as_img=False, type='train')
            b1s,b2s,ls = get_or_create_bottlenecks(sess, batch, feature_t, image_t)

            p1, p2, label = batch[0]
            im1,im2 = loader.get_img(p1), loader.get_img(p2)
            im1,im2 = (im1[...,::-1]*255).astype(np.uint8), (im2[...,::-1]*255).astype(np.uint8)
            p = sess.run(pred_t, feed_dict={p1_t:b1s, p2_t:b2s, is_training:False})

            cv2.imshow('im1', im1)
            cv2.imshow('im2', im2)
            print 'true label : ' , label
            print 'predicted label : %d (%.2f)' % (p[0,1]>0.5, p[0,1])
            if cv2.waitKey(0) == 27:
                break

if __name__ == '__main__':
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=sys.argv[:1]+unparsed)

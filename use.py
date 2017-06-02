from __future__ import absolute_import
from __future__ import division

import argparse
import sys
import os

import tensorflow as tf
import numpy as np
import cv2

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument(
    '--image', required=True, type=str, help='Absolute path to image file.')
parser.add_argument(
    '--num_top_predictions',
    type=int,
    default=10,
    help='Display this many predictions.')
parser.add_argument(
    '--graph',
    required=True,
    type=str,
    help='Absolute path to graph file (.pb)')
parser.add_argument(
    '--labels',
    required=True,
    type=str,
    help='Absolute path to labels file (.txt)')
parser.add_argument(
    '--output_layer',
    type=str,
    default='predictions:0',
    help='Name of the result operation')
parser.add_argument(
    '--input_layer',
    type=str,
    default='input:0',
    help='Name of the input operation')
parser.add_argument(
    '--loop',
    type=str2bool,
    default='',
    help='Loop through directory, instead of single file.')


def load_image(filename):
    return cv2.imread(filename)[...,::-1]/255.0 
    #return tf.gfile.FastGFile(filename, 'rb').read()


def load_labels(filename):
  """Read in labels, one label per line."""
  return [line.rstrip() for line in tf.gfile.GFile(filename)]


def load_graph(filename):
  """Unpersists graph from file as default graph."""
  with tf.gfile.FastGFile(filename, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')


def run_graph(sess, image_data, labels, input_layer_name, output_layer_name,
              num_top_predictions):
    #for op in sess.graph.get_operations():
    #    print('op', op.name)
    # Feed the image_data as input to the graph.
    #   predictions  will contain a two-dimensional array, where one
    #   dimension represents the input image count, and the other has
    #   predictions per class
    softmax_tensor = sess.graph.get_tensor_by_name(output_layer_name)
    is_training = sess.graph.get_tensor_by_name('is_training:0')
    predictions = sess.run(softmax_tensor, {input_layer_name: image_data, is_training : False})
    predictions = np.squeeze(predictions, [2]) # 7x7xnum_classes

    pred_lab = np.argmax(predictions, 2)
    pred_val = np.max(predictions, 2)

    # Sort to show labels in order of confidence
    k = np.bincount(pred_lab.flatten(),minlength=21)

    top_k = np.argsort(k)[-num_top_predictions:]

    print('Top %d : ' % num_top_predictions)
    for node_id in reversed(top_k):
        human_string = labels[node_id]
        score = np.sum(pred_val[pred_lab == node_id]) / np.sum(pred_val)
        print('\t %s (score = %.5f)' % (human_string, score))
    ## visualization

    return pred_lab, pred_val, top_k


def main(argv):
  """Runs inference on an image."""
  if argv[1:]:
    raise ValueError('Unused Command Line Args: %s' % argv[1:])

  if not tf.gfile.Exists(FLAGS.image):
    tf.logging.fatal('image file does not exist %s', FLAGS.image)

  if not tf.gfile.Exists(FLAGS.labels):
    tf.logging.fatal('labels file does not exist %s', FLAGS.labels)

  if not tf.gfile.Exists(FLAGS.graph):
    tf.logging.fatal('graph file does not exist %s', FLAGS.graph)

  WHITE = np.asarray([255,255,255], dtype=np.uint8)
  colors = [WHITE]
  for i in range(20):
      color = np.squeeze(cv2.cvtColor(np.asarray([[[i * 8, 255, 255]]],dtype=np.uint8), cv2.COLOR_HSV2BGR))
      colors.append([int(c) for c in color])

  with tf.Session() as sess:

      # load labels
      labels = load_labels(FLAGS.labels)
          # load graph, which is stored in the default session
      load_graph(FLAGS.graph)

      def run(image_path):

          image_data = load_image(image_path)
          lab,val,top_k = run_graph(sess,image_data, labels, FLAGS.input_layer, FLAGS.output_layer,
                    FLAGS.num_top_predictions)

          frame = cv2.imread(image_path)
          h,w,_ = frame.shape
          lab_frame = np.zeros((7,7,3))

          for k in top_k:
              i,j = np.where(lab==k)
              lab_frame[i,j] = np.outer(val[i,j], colors[k]) # set color

          #lab_frame = cv2.resize(lab_frame, (255,255), 0, 0, cv2.INTER_NEAREST)
          cv2.imshow('frame', frame)
          cv2.imshow('labels', lab_frame)
          if cv2.waitKey(0) == 27:
              return False
          return True

      if FLAGS.loop:
          i = 0
          for sub in os.listdir(FLAGS.image):
              i += 1
              if i < 100:
                  continue;
              image_path = os.path.join(FLAGS.image, sub)
              if not run(image_path):
                  break

      else:
          # load image
          run(FLAGS.image)
          


if __name__ == '__main__':
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=sys.argv[:1]+unparsed)

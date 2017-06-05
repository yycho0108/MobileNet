from __future__ import absolute_import
from __future__ import division

import argparse
import sys
import os

import tensorflow as tf
import numpy as np
import cv2

from timer import Timer

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
    default='data/output_graph.pb',
    help='Absolute path to graph file (.pb)')
parser.add_argument(
    '--labels',
    required=True,
    default='data/labels.txt',
    type=str,
    help='Absolute path to labels file (.txt)')
parser.add_argument(
    '--input_layer',
    type=str,
    default='input:0',
    help='Name of the input operation')
parser.add_argument(
    '--loop',
    type=str2bool,
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


def run_graph(sess, image_data, labels, input_layer_name, output_names,
              num_top_predictions):
    #for op in sess.graph.get_operations():
    #    print('op', op.name)
    # Feed the image_data as input to the graph.
    #   predictions  will contain a two-dimensional array, where one
    #   dimension represents the input image count, and the other has
    #   predictions per class
    box_t, cls_t, val_t = [sess.graph.get_tensor_by_name(o) for o in output_names]
    #print cls_t
    #filter_t = tf.cast(tf.equal(cls_t, 14), tf.float32) # --> person
    idx_t = tf.image.non_max_suppression(box_t, val_t, max_output_size=10, iou_threshold=0.10)
    s_box, s_cls, s_val = tf.gather(box_t, idx_t), tf.gather(cls_t, idx_t), tf.gather(val_t, idx_t)

    is_training = sess.graph.get_tensor_by_name('is_training:0')
    box, cls, val = sess.run([s_box, s_cls, s_val], {input_layer_name: image_data, is_training : False})
    #box, cls, val = sess.run([box_t, cls_t, val_t], {input_layer_name: image_data, is_training : False})
    print box[0], cls[0], val

    return box, cls, val

    #pred_lab = np.argmax(predictions, 2)
    #pred_val = np.max(predictions, 2)
    #
    ## Sort to show labels in order of confidence
    #k = np.bincount(pred_lab.flatten(),minlength=21)

    #top_k = np.argsort(k)[-num_top_predictions:]

    #pred_lab[pred_val < 0.10] = 20 # == background

    #print('Top %d : ' % num_top_predictions)
    #for node_id in reversed(top_k):
    #    human_string = labels[node_id]
    #    vals = pred_val[pred_lab == node_id]
    #    if len(vals) <= 0:
    #        break
    #    score = np.max(pred_val[pred_lab == node_id]) #/ np.sum(pred_val)
    #    print('\t %s (score = %.5f)' % (human_string, score))
    ### visualization
    #return pred_lab, pred_val, top_k

def resize(in_frame, h, w):

    s = in_frame.shape

    if len(s) >= 3:
        d = s[2]
    else:
        d = 1

    out_frame = np.zeros((h,w,d), dtype=in_frame.dtype)
    
    ih, iw = in_frame.shape[:2]
    for i in range(ih):
        for j in range(iw):
            i_s, i_e, j_s, j_e = map( lambda (a,b,c) : int(np.round(float(a)*b/c)),
                    [(i,h,ih),(i+1,h,ih),(j,w,iw),(j+1,w,iw)])
            out_frame[i_s:i_e,j_s:j_e] = in_frame[i,j]
    return out_frame

def putText(frame, loc, txt):
    font = cv2.FONT_HERSHEY_SIMPLEX
    ts = cv2.getTextSize(txt, font, 0.5, 0)[0]
    pt = (int(loc[0] - ts[0]/2.0), int(loc[1] - ts[1]/2.0))
    cv2.putText(frame, txt, pt, font, 0.5, (255,0,0))

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
  colors = []
  for i in range(20):
      color = np.squeeze(cv2.cvtColor(np.asarray([[[i * 8, 255, 255]]],dtype=np.uint8), cv2.COLOR_HSV2BGR))
      colors.append([int(c) for c in color])
  #for i in range(10):
  #    color = np.squeeze(cv2.cvtColor(np.asarray([[[i * 18, 255, 255]]],dtype=np.uint8), cv2.COLOR_HSV2BGR))
  #    colors.append([int(c) for c in color])
  colors.append(WHITE)

  with tf.Session() as sess:

      # load labels
      labels = load_labels(FLAGS.labels) + ['background']
          # load graph, which is stored in the default session
      load_graph(FLAGS.graph)

      def run(image_path):
          image_data = load_image(image_path)
          with Timer('Detection'):
              box, cls, val = run_graph(sess,image_data, labels, FLAGS.input_layer, ['pred_box:0', 'pred_cls:0', 'pred_val:0'],
                        FLAGS.num_top_predictions)

          good_idx = (val > 0.9)
          num = max(1, min(10, np.count_nonzero(good_idx)))
          #best_idx = np.argsort(val)[-num:]
          #print num
          #print len(best_idx)
          #print best_idx

          frame = cv2.imread(image_path)
          H,W,_ = frame.shape
          lab_frame = np.zeros((7,7,3), dtype=np.uint8)

          print [labels[c] for c in cls[:num]]

          drawn = sess.run(tf.image.draw_bounding_boxes(np.expand_dims(frame,0), np.expand_dims(box[:num],0)))[0]
          print 'ds', drawn.shape
          
          for c,b in zip(cls[:num], box[:num]):
              x = int((b[1]+b[3])*W/2)
              y = int((b[0]+b[2])*H/2)
              w = int((b[3]-b[1])*W)
              h = int((b[2]-b[0])*H)
              print 'x,y', x,y
              putText(drawn, (x,y), labels[c])
              putText(frame, (x,y), labels[c])
              cv2.rectangle(frame, (x-w//2,y-h//2), (x+w//2,y+h//2), (255,0,0), 2)


          cv2.imshow('frame', frame)
          cv2.imshow('drawn', drawn)

          if cv2.waitKey(0) == 27:
              return False

          return True

      if FLAGS.loop:
          i = 0
          for sub in os.listdir(FLAGS.image):
              i += 1
              #if i < 300:
              #    continue;
              image_path = os.path.join(FLAGS.image, sub)
              if not run(image_path):
                  break

      else:
          # load image
          run(FLAGS.image)
          


if __name__ == '__main__':
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=sys.argv[:1]+unparsed)

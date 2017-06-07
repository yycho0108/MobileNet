from __future__ import absolute_import
from __future__ import division

import argparse
import sys
import os

import tensorflow as tf
import numpy as np
import cv2

from timer import Timer

from coco_utils import COCOLoader
from voc_utils import VOCLoader

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
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

from tensorflow.python.client import timeline

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
    cv2.putText(frame, txt, pt, font, 0.5, (0,0,255))

def main(argv):
  """Runs inference on an image."""
  if argv[1:]:
    raise ValueError('Unused Command Line Args: %s' % argv[1:])

  if not tf.gfile.Exists(FLAGS.labels):
    tf.logging.fatal('labels file does not exist %s', FLAGS.labels)

  if not tf.gfile.Exists(FLAGS.graph):
    tf.logging.fatal('graph file does not exist %s', FLAGS.graph)

  WHITE = np.asarray([255,255,255], dtype=np.uint8)
  colors = []
  for i in range(20):
      color = np.squeeze(cv2.cvtColor(np.asarray([[[i * 8, 255, 255]]],dtype=np.uint8), cv2.COLOR_HSV2BGR))
      colors.append([int(c) for c in color])
  colors.append(WHITE)
  
  gpu_options = tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=0.3)
  config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)

  with tf.Session(config=config) as sess:
      

      # load labels
      labels = load_labels(FLAGS.labels) + ['background']
      # load graph, which is stored in the default session
      load_graph(FLAGS.graph)
      output_names = ['pred_box_1:0', 'pred_cls_1:0', 'pred_val_1:0']

      # grab tensors
      box_t, cls_t, val_t = [sess.graph.get_tensor_by_name(o) for o in output_names]

      idx_t = tf.reshape(tf.where(tf.equal(cls_t, 14)), [-1]) # filter - people only
      box_t, cls_t, val_t = [tf.gather(t, idx_t) for t in [box_t, cls_t, val_t]]

      idx_t = tf.reshape(tf.where(val_t > 0.01), [-1]) # filter - decent boxes only
      box_t, cls_t, val_t = [tf.gather(t, idx_t) for t in [box_t, cls_t, val_t]]


      input_tensor = sess.graph.get_tensor_by_name('input:0')

      # processing tensors
      idx_t = tf.image.non_max_suppression(box_t, val_t, max_output_size=10, iou_threshold=0.25) # collect best boxes
      box_t, cls_t, val_t = [tf.gather(t, idx_t) for t in [box_t, cls_t, val_t]]

      cv2.namedWindow('frame')
      cv2.namedWindow('truth')
      cv2.moveWindow('frame', 50, 50)
      #cap = cv2.VideoCapture(0)

      #for _ in range(10):
      #    _, image_data = cap.read()

      def run(frame):
          image_data = frame[...,::-1]/255.0

          with Timer('Detection'):
              #run_metadata = tf.RunMetadata()
    
              box, cls, val = sess.run([box_t, cls_t, val_t], {input_tensor: image_data})
              #        options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
              #        run_metadata = run_metadata)

              #trace = timeline.Timeline(step_stats = run_metadata.step_stats)
              #ctf = trace.generate_chrome_trace_format()
              #with open('timeline.json', 'w') as f:
              #    f.write(ctf)

          good_idx = (val > 0.75)
          num = min(10, np.count_nonzero(good_idx))
          print val[:num]

          #best_idx = np.argsort(val)[-num:]
          #print num
          #print len(best_idx)
          #print best_idx

          H,W,_ = frame.shape
          lab_frame = np.zeros((7,7,3), dtype=np.uint8)

          print [labels[c] for c in cls[:num]]

          #drawn = sess.run(tf.image.draw_bounding_boxes(np.expand_dims(frame,0), np.expand_dims(box[:num],0)))[0]
          for c,b in zip(cls[:num], box[:num]):
              x = int((b[1]+b[3])*W/2)
              y = int((b[0]+b[2])*H/2)
              w = int((b[3]-b[1])*W)
              h = int((b[2]-b[0])*H)
              putText(frame, (x,y), labels[c])
              cv2.rectangle(frame, (x-w//2,y-h//2), (x+w//2,y+h//2), (255,0,0), 2)
          cv2.imshow('frame', frame)

      coco_root = os.getenv('COCO_ROOT')
      coco_type = 'val2014'
      loader = COCOLoader(coco_root, coco_type)

      #voc_root = os.getenv('VOC_ROOT')
      #loader = VOCLoader(voc_root)

      #print loader.list_image_sets()
      l = list(loader.list_all(target='person'))
      np.random.shuffle(l)

      for img_id in l:
          print img_id, type(img_id)
          img_path, boxs, lbls = loader.grab(img_id)
          frame = cv2.imread(img_path)
          print lbls
          h,w = frame.shape[:2]
          run(frame.copy())
          for box in boxs:
              y1,x1,y2,x2 = [int(b*s) for (b,s) in zip(box, [h,w,h,w])]
              cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 1)
          cv2.imshow('truth', frame)
          if (cv2.waitKey(0) == 27):
              break


if __name__ == '__main__':
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=sys.argv[:1]+unparsed)

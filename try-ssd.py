from __future__ import absolute_import
from __future__ import division

import math
import tensorflow as tf

from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile
#from tensorflow.python import debug as tf_debug

import numpy as np
import cv2

from datasets import dataset_factory
from nets import nets_factory
from preprocessing import preprocessing_factory

import os
import signal

from voc_utils import VOCLoader
from coco_utils import COCOLoader


import ssd

from utilities import *

import threading

slim = tf.contrib.slim

#################
#   PARAMETERS  #
#################
input_ckpt_path = './data/model.ckpt-906808'

output_graph_path = 'data/train/output_graph.pb'
output_labels_path = 'data/train/labels.txt'
output_ckpt_path = 'data/train/model.ckpt'

voc_loader = VOCLoader(os.getenv('VOC_ROOT')) #17125
train_loader = COCOLoader(os.getenv('COCO_ROOT'),'train2014') #66843
valid_loader = COCOLoader(os.getenv('COCO_ROOT'),'val2014')

log_root = '/tmp/mobilenet_logs/'
if not os.path.exists(log_root):
    os.makedirs(log_root)

bottleneck_root = 'data/bottlenecks'
if not os.path.exists(bottleneck_root):
    os.makedirs(bottleneck_root)

categories = train_loader.list_image_sets() # same
num_classes = len(categories)
train_batch_size = 128
valid_batch_size = 1
MODEL_INPUT_WIDTH = 224
MODEL_INPUT_HEIGHT = 224
MODEL_INPUT_DEPTH = 3

train_iters = int(4e3)
#split_ratio = 0.85

# Learning Rate Params
init_learning_rate = 1e-3
min_learning_rate = 1e-4
num_samples = len(voc_loader.list_all()) + len(train_loader.list_all()) # = 83968
steps_per_epoch = num_samples / train_batch_size # or thereabout.
epochs_per_decay = 0.5
net_decay_steps = train_iters / (epochs_per_decay * steps_per_epoch) # of decay steps in training run
decay_factor = (min_learning_rate / init_learning_rate) ** (1./net_decay_steps)
steps_per_decay = steps_per_epoch * epochs_per_decay

tf.logging.set_verbosity(tf.logging.INFO)

num_threads = 4

##############
# SSD PARAMS #
##############

box_ratios = [1.0, 1.0, 2.0, 3.0, 1.0/2, 1.0/3]
num_boxes = len(box_ratios)

num_outputs = num_boxes * (num_classes + 4)

##################

def dwc(inputs, num_out, scope, stride=1, output_activation_fn=tf.nn.elu):
    dc = slim.separable_conv2d(inputs,
            num_outputs=None,
            stride=stride,
            depth_multiplier=1,
            kernel_size=[3, 3],
            scope=scope+'/dc')
    pc = slim.conv2d(dc,
            num_out,
            kernel_size=[1, 1],
            activation_fn=output_activation_fn,
            scope=scope+'/pc')
    return pc

def ssd_ops(feature_tensors, gt_box_tensor, gt_split_tensor, gt_label_tensor, num_classes, is_training=True, reuse=None):
    with tf.variable_scope('SSD', reuse=reuse) as sc:
        with slim.arg_scope([slim.conv2d, slim.separable_convolution2d],
                activation_fn=tf.nn.elu,
                weights_regularizer=slim.l2_regularizer(4e-5),
                normalizer_fn = slim.batch_norm,
                normalizer_params={
                    'is_training' : is_training,
                    'decay' : 0.99,
                    'fused' : True,
                    'reuse' : reuse,
                    'scope' : 'BN'
                    }
                ):

            feature_tensors = list(feature_tensors) # copy list in case used outside

            # extra features
            depths = [512, 384, 256]
            for i in range(3):
                with tf.variable_scope('feat_%d' % i):
                    logits = dwc(feature_tensors[-1], depths[i], scope='f_dwc', stride=2)
                    feature_tensors.append(logits)

            # bbox predictions
            output_tensors = []
            for i, t in enumerate(feature_tensors):
                h,w = t.get_shape().as_list()[1:3]
                with tf.variable_scope('box_%d' % i):
                    #logits = slim.conv2d(t, num_outputs=num_outputs, kernel_size=[3,3], activation_fn=None, scope='b_conv')
                    logits = dwc(t, num_outputs, scope='b_dwc', output_activation_fn=None)
                    logits = tf.reshape(logits, (-1, h, w, num_boxes, 4+num_classes)) 
                    output_tensors.append(logits)

            d_boxes = []
            net_acc = []

            n = len(output_tensors)

            s_min = 0.1
            s_max = 0.9
            scales = []

            def s(i):
                return s_min + (s_max - s_min) / (n-1) * (i)

            with tf.name_scope('train'):
                for i, logits in enumerate(output_tensors):
                    s_k = s(i)
                    s_kn = s(i+1) 
                    w = np.sqrt(s_kn/s_k)
                    d_box = np.reshape(ssd.default_box(logits, box_ratios, scale=s_k, wildcard=w), (-1,4))
                    d_box = tf.constant(d_box, tf.float32)
                    iou, sel, cls, delta = ssd.create_label_tf(gt_box_tensor, gt_split_tensor, gt_label_tensor, d_box)
                    acc = ssd.ops(logits, iou, sel, cls, delta, num_classes = num_classes, is_training = is_training)
                    d_boxes.append(d_box)
                    net_acc.append(acc)

                acc = tf.reduce_mean(net_acc)
            with tf.name_scope('pred'):
                pred_box, pred_cls, pred_val = ssd.pred(output_tensors, d_boxes, num_classes=num_classes)

    return {
            'box' : tf.identity(pred_box, name='pred_box'),
            'cls' : tf.identity(pred_cls, name='pred_cls'),
            'val' : tf.identity(pred_val, name='pred_val'),
            'acc' : acc,
            }

def get_or_create_bottlenecks(sess, bottleneck_tensors, image, loader, anns, batch_size, distorter=None):

    all = (batch_size <= 0)

    if not all:
        anns = np.random.choice(anns, batch_size, replace=False)

    n = len(bottleneck_tensors)
    btls = [[] for _ in range(n)]
    boxs = []
    lbls = []
    spls = []

    for i, ann in enumerate(anns):

        if all and i%100==0:
            print '%d ) %s' % (i, ann)

        ### GRAB DATA ###
        btl_file = os.path.join(bottleneck_root, str(ann) + '_btl.npz')

        img_file, box, lbl = loader.grab(ann)

        if all or not os.path.exists(btl_file):
            # TODO : currently disabled btl
            image_in = cv2.imread(img_file)[...,::-1]/255.0
            btl = sess.run(bottleneck_tensors, feed_dict={image : image_in})
            d = {str(i) : btl[i] for i in range(n)}
            np.savez(btl_file, **d)

        if not all:
            if distorter is not None and np.random.random() > 0.5:
                # apply distortion
                image_in = cv2.imread(img_file)[...,::-1]/255.0
                image_in, box, lbl = distorter.apply(sess, image_in, box, lbl)
                btl = sess.run(bottleneck_tensors, feed_dict={image : image_in})
                for i in range(n):
                    btls[i].append(btl[i])# for i in range(n))
            else:
                btl = np.load(btl_file, allow_pickle=True)
                for i in range(n):
                    btls[i].append(btl[str(i)])# for i in range(n))

            #btls[i].append(btl[str(i)] for i in range(n))

        boxs.append(box)
        lbls.append(lbl)
        spls.append(len(lbl))
        #################

        ### RUN DISTORTION ###
        #d_im, d_bb, d_lbl = sess.run([d_image, d_bbox, d_label], feed_dict={image : image_in, bbox : bbox_in, label : label_in})
        #btl = sess.run(bottleneck_tensor, feed_dict={image : d_im})
        #lbl = get_label(d_bb, d_lbl, w,h)
        ######################

    if not all:
        btls = [np.concatenate(b, axis=0) for b in btls]
        boxs = np.concatenate(boxs, axis=0)
        lbls = np.concatenate(lbls, axis=0)
        res = btls + [boxs, spls, lbls]
        # no need to concatenate spls
        return res
    else:
        return [], []

stop_request = False
def sigint_handler(signal, frame):
    global stop_request
    stop_request = True

def main(_):
    global stop_request
    signal.signal(signal.SIGINT, sigint_handler)

    with tf.Graph().as_default():

        ####################
        # Select the model #
        ####################
        network_fn = nets_factory.get_network_fn(
                'mobilenet',
                num_classes=1001,
                is_training=False,
                width_multiplier=1.0
                )

        #####################################
        # Select the preprocessing function #
        #####################################
        preprocessing_name = 'mobilenet'
        image_preprocessing_fn = preprocessing_factory.get_preprocessing(
            preprocessing_name,
            is_training=False)

        eval_image_size = network_fn.default_image_size

        image = tf.placeholder(tf.float32, [None,None,3], name='input')
        images = tf.expand_dims(image_preprocessing_fn(image, eval_image_size, eval_image_size), 0)

        ####################
        # Define the model #
        ####################
        #distorter = Distorter(image)
        logits, _ = network_fn(images)

        ###############
        # Restoration #
        ###############

        variables_to_restore = slim.get_variables_to_restore()
        
        #gpu_options = tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=1.0)
        #config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)

        with tf.Session() as sess:
            #sess = tf_debug.LocalCLIDebugWrapperSession(sess)

            ### DATA PROVIDERS ###
            anns_voc = voc_loader.list_all() # use for training
            anns_train = train_loader.list_all()
            anns_valid = valid_loader.list_all()
            data_provider = [(train_loader,anns_train),(voc_loader,anns_voc),(valid_loader,anns_valid)]

            ### DEFINE TENSORS ###
            bottleneck_names = [ # source bottleneck
                    'MobileNet/conv_ds_6/pw_batch_norm/Relu:0',
                    'MobileNet/conv_ds_12/pw_batch_norm/Relu:0',
                    'MobileNet/conv_ds_14/pw_batch_norm/Relu:0'
                    ]
            bottleneck_tensors = [sess.graph.get_tensor_by_name(b) for b in bottleneck_names]

            def create_input_tensors(is_training=True):
                if is_training:
                    input_size = train_batch_size
                    feature_tensors = [tf.placeholder(tf.float32, shape=([train_batch_size]+b.get_shape().as_list()[1:])) for b in bottleneck_tensors]
                else:
                    input_size = valid_batch_size
                    feature_tensors = [tf.placeholder_with_default(b, shape=([valid_batch_size]+b.get_shape().as_list()[1:])) for b in bottleneck_tensors]
                gt_boxes = tf.placeholder(tf.float32, [None, 4]) # ground truth boxes -- aggregated
                gt_splits = tf.placeholder(tf.int32, [input_size]) # # ground truth boxes per sample
                gt_labels = tf.placeholder(tf.int32, [None]) # ground truth labels -- aggregated
                return feature_tensors + [gt_boxes, gt_splits, gt_labels]

            ### CREATE QUEUE ###

            # Train Inputs

            t_input_tensors = create_input_tensors(is_training=True)
            t_input_dtypes = [t.dtype for t in t_input_tensors]
            t_input_shapes = [t.shape for t in t_input_tensors] # remove batch dim.

            t_select_ratio = np.array([float(len(anns_voc)), float(len(anns_train)), 0.0])
            t_select_ratio /= sum(t_select_ratio)
            t_q = tf.PaddingFIFOQueue(capacity=8, dtypes=t_input_dtypes, shapes=t_input_shapes)
            t_enqueue_op = t_q.enqueue(t_input_tensors)
            t_ft_1, t_ft_2, t_ft_3, t_gb, t_gs, t_gl = t_q.dequeue()

            # Validation Inputs

            v_input_tensors = create_input_tensors(is_training=False)
            v_input_dtypes = [t.dtype for t in v_input_tensors]
            v_input_shapes = [t.shape for t in v_input_tensors]

            v_select_ratio = [0.0,0.0,1.0] # only select validation
            v_q = tf.PaddingFIFOQueue(capacity=8, dtypes=v_input_dtypes, shapes=v_input_shapes)
            v_enqueue_op = v_q.enqueue(v_input_tensors)
            v_ft_1, v_ft_2, v_ft_3, v_gb, v_gs, v_gl = v_q.dequeue()

            def enqueue(coord, is_training=True):
                select_ratio = t_select_ratio if is_training else v_select_ratio
                batch_size = train_batch_size if is_training else valid_batch_size
                input_tensors = t_input_tensors if is_training else v_input_tensors
                enqueue_op = t_enqueue_op if is_training else v_enqueue_op
                while not coord.should_stop():
                    loader,anns = data_provider[np.random.choice(3, p=select_ratio)]
                    input_values  = get_or_create_bottlenecks(sess, bottleneck_tensors, image, loader, anns, batch_size)
                    feed_dict = {t:v for (t,v) in zip(input_tensors, input_values)}
                    sess.run(enqueue_op, feed_dict = feed_dict)

            coord = tf.train.Coordinator()

            ### DEFINE MODEL ###
            t_ops = ssd_ops([t_ft_1, t_ft_2, t_ft_3], t_gb, t_gs, t_gl, num_classes, reuse=None, is_training=True)
            v_ops = ssd_ops([v_ft_1, v_ft_2, v_ft_3], v_gb, v_gs, v_gl, num_classes, reuse=True, is_training=False)
            
            loss = tf.losses.get_total_loss()
            v_loss = tf.reduce_sum(tf.losses.get_losses(loss_collection='valid_loss'))

            with tf.name_scope('evaluation'):
                tf.summary.scalar('loss', loss)
                tf.summary.scalar('train_accuracy', t_ops['acc'])
                tf.summary.scalar('v_loss', v_loss, collections=['valid_summary'])
                tf.summary.scalar('valid_accuracy', v_ops['acc'], collections=['valid_summary'])
            
            global_step = slim.get_or_create_global_step()

            learning_rate = tf.train.exponential_decay(init_learning_rate, global_step, steps_per_decay, decay_factor)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

            opt = tf.train.AdamOptimizer(learning_rate = learning_rate)#.minimize(train['loss'])
            train_vars = slim.get_trainable_variables(scope='SSD')

            with tf.control_dependencies(update_ops):
                train_op = opt.minimize(loss, global_step=global_step, var_list=train_vars)

            sess.run(tf.global_variables_initializer())

            saver = tf.train.Saver(variables_to_restore)
            saver.restore(sess, input_ckpt_path) # -- only restores mobilenet weights

            total_saver = tf.train.Saver() # -- save all
            #total_saver.restore(sess, output_ckpt_path) # restore all

            run_id = 'run_%02d' % len(os.walk(log_root).next()[1])
            run_log_root = os.path.join(log_root, run_id)

            train_writer = tf.summary.FileWriter(os.path.join(run_log_root, 'train'), sess.graph)
            valid_writer = tf.summary.FileWriter(os.path.join(run_log_root, 'valid'), sess.graph)

            train_summary = tf.summary.merge_all()
            valid_summary = tf.summary.merge_all('valid_summary')

            ### START TRAINING ###
            t_feed_threads = [threading.Thread(target = enqueue, args=(coord, True)) for _ in range(1)]
            v_feed_threads = [threading.Thread(target = enqueue, args=(coord, False)) for _ in range(1)]

            for t in (v_feed_threads + t_feed_threads):
                t.daemon = True
                t.start()

            for i in range(train_iters):
                if stop_request:
                    break

                s,_ = sess.run([train_summary,train_op])
                train_writer.add_summary(s, i)

                if i % 50 == 0: # -- evaluate
                    l, a, s = sess.run([v_loss, v_ops['acc'], valid_summary])
                    valid_writer.add_summary(s, i)
                    print('%d ) Loss : %.3f, Accuracy : %.2f' % (i, l, a))

                if i>0 and i % 100 == 0: # -- save checkpoint
                    total_saver.save(sess, output_ckpt_path, global_step=global_step)

            coord.request_stop()
            coord.join(t_feed_threads)
            coord.join(v_feed_threads)

            if i > 100: # didn't terminate prematurely
                output_graph_def = graph_util.convert_variables_to_constants(
                        sess, sess.graph.as_graph_def(), [v_ops[s].name[:-2] for s in ['box', 'cls', 'val']]) # strip :0
                with gfile.FastGFile(output_graph_path, 'wb') as f:
                  f.write(output_graph_def.SerializeToString())
                with gfile.FastGFile(output_labels_path, 'w') as f:
                  f.write('\n'.join(categories) + '\n')

if __name__ == '__main__':
  tf.app.run()

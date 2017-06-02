from __future__ import absolute_import
from __future__ import division

import math
import tensorflow as tf

from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile

import numpy as np
import cv2

from datasets import dataset_factory
from nets import nets_factory
from preprocessing import preprocessing_factory

import os
from voc_utils import VOCLoader

import ssd

from utilities import *

slim = tf.contrib.slim

#################
#   PARAMETERS  #
#################
input_ckpt_path = './data/model.ckpt-906808'
bottleneck_name = 'MobileNet/conv_ds_14/pw_batch_norm/Relu:0'

output_graph_path = 'data/train/output_graph.pb'
output_labels_path = 'data/train/labels.txt'
output_ckpt_path = 'data/train/model.ckpt'

loader = VOCLoader(os.getenv('VOC_ROOT'))

log_root = '/tmp/mobilenet_logs/'
if not os.path.exists(log_root):
    os.makedirs(log_root)

bottleneck_root = 'data/bottlenecks'
if not os.path.exists(bottleneck_root):
    os.makedirs(bottleneck_root)

categories = loader.list_image_sets()
num_classes = len(categories)
batch_size = 64
valid_batch_size = 4
MODEL_INPUT_WIDTH = 224
MODEL_INPUT_HEIGHT = 224
MODEL_INPUT_DEPTH = 3

train_iters = 4000
split_ratio = 0.85
learning_rate = 1e-4

tf.logging.set_verbosity(tf.logging.INFO)


##############
# SSD PARAMS #
##############

box_ratios = [1.0, 1.0, 2.0, 3.0, 1.0/2, 1.0/3]
num_boxes = len(box_ratios)

##################

def add_input_distortions(image, bbox, label, flip_left_right=True, random_crop=0.1, random_scale=0.3, random_brightness=0.5):
    # image = (1, 224, 224, 3)
    # l_bbox= (num_bbox, 4), in normalized coordinates (0.0 ~ 1.0)
    # bbox format = (y1,x1,y2,x2)

    ### SCALE ###
    margin_scale = 1.0 + random_crop
    resize_scale = 1.0 + random_scale
    margin_scale_value = tf.constant(margin_scale)
    resize_scale_value = tf.random_uniform((1,),
                                           minval=1.0,
                                           maxval=resize_scale)
    scale_value = tf.multiply(margin_scale_value, resize_scale_value)
    scale_width = tf.multiply(scale_value, MODEL_INPUT_WIDTH)
    scale_height = tf.multiply(scale_value, MODEL_INPUT_HEIGHT)
    scale_shape = tf.cast(tf.concat([scale_height, scale_width],0), dtype=tf.int32)
    #===========
    scaled_image = tf.image.resize_bilinear(image, scale_shape)
    # bbox is invariant to scale
    #############
    scaled_image_3d = tf.squeeze(scaled_image, [0])

    ### CROP ###
    w_offset_max = scale_width - MODEL_INPUT_WIDTH
    h_offset_max = scale_height - MODEL_INPUT_HEIGHT
    w_offset = tf.random_uniform((), minval=0, maxval = w_offset_max) # TODO : int32?
    h_offset = tf.random_uniform((), minval=0, maxval = h_offset_max) 

    w_offset_norm = w_offset / scale_width
    h_offset_norm = h_offset / scale_height
    w_norm = MODEL_INPUT_WIDTH / scale_width
    h_norm = MODEL_INPUT_HEIGHT / scale_height

    split_bbox = tf.unstack(bbox, axis=1)
    split_bbox[0] = tf.clip_by_value((split_bbox[0] - h_offset_norm)/h_norm, 0, 1)# y1
    split_bbox[1] = tf.clip_by_value((split_bbox[1] - w_offset_norm)/w_norm, 0, 1)# x1
    split_bbox[2] = tf.clip_by_value((split_bbox[2] - h_offset_norm)/h_norm, 0, 1)# y2
    split_bbox[3] = tf.clip_by_value((split_bbox[3] - w_offset_norm)/w_norm, 0, 1)# x2

    h_mask = tf.greater(split_bbox[2] - split_bbox[0], 1/16.) 
    w_mask = tf.greater(split_bbox[3] - split_bbox[1], 1/16.)
    mask = tf.logical_and(w_mask, h_mask)
    sel_idx = tf.where(mask)

    split_bbox[0] = tf.gather(split_bbox[0], sel_idx)
    split_bbox[1] = tf.gather(split_bbox[1], sel_idx)
    split_bbox[2] = tf.gather(split_bbox[2], sel_idx)
    split_bbox[3] = tf.gather(split_bbox[3], sel_idx)
    label = tf.gather(label, sel_idx)

    offset = tf.cast(tf.concat([h_offset, w_offset, tf.constant([0.0])],0), tf.int32)
    size = [MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH, MODEL_INPUT_DEPTH]
    #===========
    cropped_image = tf.slice(scaled_image_3d, offset, size)
    cropped_bbox  = tf.concat(split_bbox, axis=1)
    ############

    ### FLIP ###
    if flip_left_right:

        def flip():
            pre_flip = tf.unstack(tf.add(tf.multiply(cropped_bbox, [1,-1,1,-1]), [0,1,0,1]),axis=1)
            return tf.stack([pre_flip[0],pre_flip[3],pre_flip[2],pre_flip[1]], axis=1)
            
        p = (tf.random_uniform(())>0.5)
        flipped_image = tf.cond(p, lambda : tf.image.flip_left_right(cropped_image), lambda : cropped_image)
        flipped_bbox  = tf.cond(p, flip, lambda : cropped_bbox)
    else:
        flipped_image = cropped_image
        flipped_bbox  = cropped_bbox
    ############

    ### BRIGHTEN ###
    brightness_min = 1.0 - random_brightness
    brightness_max = 1.0 + random_brightness
    brightness_value = tf.random_uniform((),
                                         minval=brightness_min,
                                         maxval=brightness_max)
    #==============
    brightened_image = tf.clip_by_value(tf.multiply(flipped_image, brightness_value), 0.0, 1.0)
    ################

    return brightened_image, flipped_bbox, label

def ann2bbox(ann, categories):
    width = int(ann.findChild('width').contents[0])
    height = int(ann.findChild('height').contents[0])
    objs = ann.findAll('object')

    bbox = []
    labels = []

    for obj in objs:
        label = categories.index(obj.findChild('name').contents[0])
        labels.append(label)
        box = obj.findChild('bndbox')
        y_min = float(box.findChild('ymin').contents[0]) / height
        x_min = float(box.findChild('xmin').contents[0]) / width
        y_max = float(box.findChild('ymax').contents[0]) / height
        x_max = float(box.findChild('xmax').contents[0]) / width
        bbox.append([y_min,x_min,y_max,x_max])

    return np.asarray(bbox, dtype=np.float32), np.asarray(labels, dtype=np.int32)

def dwc(inputs, num_out, scope):
    dc = slim.separable_conv2d(inputs,
            num_outputs=None,
            stride=1,
            depth_multiplier=1,
            kernel_size=[3, 3],
            scope=scope+'/dc')
    pc = slim.conv2d(dc,
            num_out,
            kernel_size=[1, 1],
            scope=scope+'/pc')
    return pc

def extended_ops(input_tensor, label_tensor, num_classes, is_training=True, reuse=None):
    with tf.name_scope('extended_ops') as sc:
        with slim.arg_scope([slim.conv2d, slim.separable_convolution2d],
                activation_fn=tf.nn.relu,
                weights_regularizer=slim.l2_regularizer(0.005),
                normalizer_fn = slim.batch_norm,
                normalizer_params={
                    'is_training' : is_training,
                    'decay' : 0.9,
                    'fused' : True,
                    'reuse' : reuse,
                    }
                ):

            _, h, w, d = input_tensor.get_shape().as_list()
            logits = dwc(input_tensor, 256, scope='dwc_1')
            logits = dwc(logits, num_boxes * (num_classes+4), scope='dwc_2')
            logits = tf.reshape(logits, (-1, h, w, num_boxes, 4+num_classes)) 

            df_box = ssd.default_box(input_tensor, box_ratios)

            pred_box, pred_cls, pred_val = ssd.pred([logits], [df_box], num_classes=num_classes, num_boxes=num_boxes)
            loss, acc = ssd.eval(logits, label_tensor, num_classes = num_classes)

        with tf.name_scope('evaluation'):
            tf.summary.scalar('loss', loss)
            tf.summary.scalar('accuracy', acc)

    return {
            'box' : tf.identity(pred_box, name='pred_box'),
            'cls' : tf.identity(pred_cls, name='pred_cls'),
            'score' : tf.identity(pred_val, name='pred_vals'),
            'loss' : loss,
            'acc' : acc,
            }

def get_or_create_bottlenecks(sess, bottleneck_tensor, image, loader, anns, df_boxes, batch_size):

    _, h, w, d = bottleneck_tensor.get_shape().as_list()

    all = (batch_size <= 0)

    if not all:
        anns = np.random.choice(anns, batch_size, replace=False)

    btls = []
    lbls = []

    for i, ann in enumerate(anns):

        if all and i%100==0:
            print '%d ) %s' % (i, ann)

        ### GRAB DATA ###
        btl_file = os.path.join(bottleneck_root, ann + '_btl.npy')
        lbl_file = os.path.join(bottleneck_root, ann + '_lbl.npy')

        if all or not os.path.exists(btl_file):
            img_file, ann = loader.grab_pair(ann)
            image_in = cv2.imread(img_file)[...,::-1]/255.0
            bbox_in, label_in = ann2bbox(ann, categories)
            # TODO : currently disabled btl
            #btl = sess.run(bottleneck_tensor, feed_dict={image : image_in})
            #np.save(btl_file, btl, allow_pickle=True)
            lbl = ssd.create_label(bbox_in, label_in, df_boxes, num_classes)
            np.save(lbl_file, lbl, allow_pickle=True)

        if not all:
            btl = np.load(btl_file, allow_pickle=True)
            lbl = np.load(lbl_file, allow_pickle=True)
            btls.append(btl)
            lbls.append(lbl)
        #################

        ### RUN DISTORTION ###
        #d_im, d_bb, d_lbl = sess.run([d_image, d_bbox, d_label], feed_dict={image : image_in, bbox : bbox_in, label : label_in})
        #btl = sess.run(bottleneck_tensor, feed_dict={image : d_im})
        #lbl = get_label(d_bb, d_lbl, w,h)
        ######################

    if not all:
        btls = np.concatenate(btls, axis=0)
        lbls = np.concatenate(lbls, axis=0)
        return btls, lbls
    else:
        return [], []

def main(_):

    with tf.Graph().as_default():
        slim.get_or_create_global_step()

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

        bbox = tf.placeholder(tf.float32, [None, 4])
        label = tf.placeholder(tf.float32,  [None])
        d_image, d_bbox, d_label = add_input_distortions(tf.expand_dims(image,0), bbox, label)

        # --> original predictions ...
        logits, _ = network_fn(images)
        #predictions = tf.argmax(logits, 1)

        ###############
        # Restoration #
        ###############

        variables_to_restore = slim.get_variables_to_restore()

        with tf.Session() as sess:

            ### DEFINE MODEL ###
            bottleneck_tensor = tf.stop_gradient(sess.graph.get_tensor_by_name(bottleneck_name))
            _, h, w, d = bottleneck_tensor.get_shape().as_list()
            label_tensor = tf.placeholder(tf.float32, (None, h, w, num_boxes, 4+num_classes))
            input_tensor = tf.placeholder_with_default(bottleneck_tensor, shape=(None, h, w, d))
            is_training = tf.placeholder(tf.bool, [], name='is_training')

            train = extended_ops(input_tensor, label_tensor, num_classes, is_training=is_training, reuse=None)
            
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                opt = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(train['loss'])
            ####################

            sess.run(tf.global_variables_initializer())

            saver = tf.train.Saver(variables_to_restore)
            saver.restore(sess, input_ckpt_path)

            run_id = 'run_%02d' % len(os.walk(log_root).next()[1])
            run_log_root = os.path.join(log_root, run_id)

            train_writer = tf.summary.FileWriter(os.path.join(run_log_root, 'train'), sess.graph)
            valid_writer = tf.summary.FileWriter(os.path.join(run_log_root, 'valid'), sess.graph)

            merged = tf.summary.merge_all()

            anns = loader.list_all()
            np.random.shuffle(anns)
            n = len(anns)
            sp = int(n * split_ratio)
            anns_train = anns[:sp]
            anns_valid = anns[sp:]
            
            df_boxes = [ssd.default_box(bottleneck_tensor, box_ratios)]

            # cache call
            #get_or_create_bottlenecks(sess, bottleneck_tensor, image, loader, anns, df_boxes, batch_size=-1)

            for i in range(train_iters):

                btls, lbls = get_or_create_bottlenecks(sess, bottleneck_tensor, image, loader, anns_train, df_boxes, batch_size)

                s,_ = sess.run([merged, opt], feed_dict={input_tensor : btls, label_tensor: lbls, is_training : True})
                train_writer.add_summary(s, i)

                if i % 20 == 0: # -- evaluate
                    btls, lbls = get_or_create_bottlenecks(sess, bottleneck_tensor, image, loader, anns_valid, df_boxes, valid_batch_size)

                    l, a, s = sess.run([train['loss'], train['acc'], merged], feed_dict={input_tensor : btls, label_tensor: lbls, is_training : False})
                    #b, c, sc = sess.run([train['box'], train['cls'], train['score']], feed_dict={input_tensor : btls, label_tensor: lbls, is_training : False})

                    valid_writer.add_summary(s, i)
                    print('%d ) Loss : %.3f, Accuracy : %.2f' % (i, l, a))
                    saver.save(sess, output_ckpt_path)

                ### VISUALIZE DISTORTIONS ###
                # d_im_bbox = tf.image.draw_bounding_boxes(tf.expand_dims(d_image,0), tf.expand_dims(d_bbox, 0))
                # im_bbox = tf.image.draw_bounding_boxes(tf.expand_dims(image,0), tf.expand_dims(bbox, 0))

                # res, res_d = sess.run([im_bbox, d_im_bbox], feed_dict={image : image_in, bbox : bbox_in})

                # cv2.imshow('orig', (res[0,...,::-1] * 255).astype(np.uint8))
                # cv2.imshow('dist', (res_d[0,...,::-1] * 255).astype(np.uint8))
                # if cv2.waitKey(0) == 27:
                #     break
                #############################

                ### RUN PREDICTION ###
                #c_pred = sess.run(predictions, feed_dict={image : c_image})
                ######################
            output_graph_def = graph_util.convert_variables_to_constants(
                    sess, sess.graph.as_graph_def(), [train[s].name[:-2] for s in ['box', 'cls', 'score']]) # strip :0
            with gfile.FastGFile(output_graph_path, 'wb') as f:
              f.write(output_graph_def.SerializeToString())
            with gfile.FastGFile(output_labels_path, 'w') as f:
              f.write('\n'.join(categories) + '\n')

if __name__ == '__main__':
  tf.app.run()

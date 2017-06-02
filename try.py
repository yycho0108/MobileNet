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

slim = tf.contrib.slim

#################
#   PARAMETERS  #
#################
input_ckpt_path = './data/model.ckpt-906808'
bottleneck_name = 'MobileNet/conv_ds_14/pw_batch_norm/Relu:0'

output_graph_path = '/tmp/output_graph.pb'
output_labels_path = '/tmp/labels.txt'
output_ckpt_path = '/tmp/model.ckpt'

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

MODEL_INPUT_WIDTH = 224
MODEL_INPUT_HEIGHT = 224
MODEL_INPUT_DEPTH = 3

train_iters = 1000
split_ratio = 0.85
learning_rate = 1e-3

tf.logging.set_verbosity(tf.logging.INFO)
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

def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

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

        logits = dwc(input_tensor, 256, scope='dwc_4')
        logits = dwc(logits, num_classes, scope='dwc_5')

        #logits = slim.conv2d(
        #        input_tensor,
        #        num_classes,
        #        [3,3],
        #        padding='SAME',
        #        scope='conv_2',
        #        reuse=reuse,
        #        )

    #if reuse is None:
    #    ws = slim.get_variables_by_name('weights')
    #    for i, w in enumerate(ws):
    #        with tf.name_scope('sw_%d' % i):
    #            variable_summaries(w)

    #with tf.variable_scope('dist_scope', reuse=reuse) as scope:
    #    dist = tf.get_variable('dist', initializer=np.ones(num_classes, dtype=np.float32), dtype=np.float32,trainable=False)
    #    dist_update = dist.assign_add(tf.reduce_sum(labels, axis=-1).reshape([-1]))
    #    #dist_update = dist.assign_add(tf.bincount(label_tensor, minlength=num_classes))
    #    class_weights = (1.0/dist_update)
    #    class_weights = num_classes * class_weights / tf.reduce_sum(class_weights)
    #    tf.summary.histogram('dist', dist)

    #labels = tf.one_hot(tf.cast(label_tensor, tf.uint8), depth=num_classes, axis=-1)
    labels = label_tensor #(b, n, h, c)
    pred = tf.nn.softmax(logits, -1)

    label_pr = tf.cast(tf.argmax(pred, axis=-1), tf.int32) # --> it's okay to apply this before softmax
    label_gt = tf.cast(tf.argmax(label_tensor, axis=-1), tf.int32)

    #acc = tf.reduce_mean(tf.cast(tf.equal(pred_label, label_tensor), tf.float32))

    obj_mask_gt = tf.greater(tf.reduce_max(label_tensor, axis=-1), 0.25) # high confidence for object presence
    obj_mask_pr = tf.greater(tf.reduce_max(pred, axis=-1),         0.25)

    obj_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=tf.cast(obj_mask_pr,tf.float32), labels=tf.cast(obj_mask_gt,tf.float32)))

    clf_idx = tf.where(obj_mask_gt)
    clf_loss = tf.reduce_mean(tf.gather_nd(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels), clf_idx))

    acc = tf.reduce_mean(tf.cast(tf.where(obj_mask_gt, tf.equal(label_pr, label_gt), tf.equal(obj_mask_gt, obj_mask_pr)), tf.float32))

    loss = obj_loss + clf_loss # alpha factor??

    #loss = tf.nn.softmax_cross_entropy_with_logits(labels = labels, logits = logits * class_weights)
    #loss = tf.reduce_mean(loss)

    with tf.name_scope('evaluation'):
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('accuracy', acc)

    return {
            'pred' : tf.identity(pred, name='predictions'),
            'acc' : acc,
            'loss' : loss,
            }

def overlap(r_a, r_b):
    ya1,xa1,ya2,xa2 = r_a
    yb1,xb1,yb2,xb2 = r_b
    return max(0, min(xa2,xb2) - max(xa1,xb1)) * max(0, min(ya2,yb2) - max(ya1,yb1))

def get_label(bbox, label, h, w, d):
    res = np.zeros((h,w,d), dtype=np.float32)
    w_f = 1.0/w
    h_f = 1.0/h
    for b,l in zip(bbox, label):
        for i in range(h):
            for j in range(w):
                r = (i*h_f, j*w_f, (i+1)*h_f, (j+1)*w_f)
                o = overlap(b,r)/(w_f*h_f)
                res[i, j, l] += o
    return res


    # res = np.zeros((h,w), dtype=np.float32)
    # ars = map(lambda b : (b[2]-b[0])*(b[3]-b[1]), bbox)
    # idx = np.argsort(ars)
    # for i in reversed(idx): # bigger one first
    #     i0,j0,i1,j1 = map(int, bbox[i] * [h,w,h,w])
    #     res[i0:i1, j0:j1, label[i]] += 1.0
    #     res[i0:i1, j0:j1, 0] = 0.0

    return np.expand_dims(res, 0)

def basename(s):
    return os.path.splitext(os.path.basename(s))[0]

def get_or_create_bottlenecks(sess, bottleneck_tensor, image, loader, anns, batch_size=1):
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
            #btl = sess.run(bottleneck_tensor, feed_dict={image : image_in})
            lbl = get_label(bbox_in, label_in, h, w, num_classes)
            #np.save(btl_file, btl, allow_pickle=True)
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
        lbls = np.stack(lbls, axis=0)
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
            label_tensor = tf.placeholder(tf.float32, (None, h, w, num_classes))
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

            # cache call
            #get_or_create_bottlenecks(sess, bottleneck_tensor, image, loader, anns, batch_size=-1)

            for i in range(train_iters):
                btls, lbls = get_or_create_bottlenecks(sess, bottleneck_tensor, image, loader, anns_train, batch_size)

                s,_ = sess.run([merged, opt], feed_dict={input_tensor : btls, label_tensor: lbls, is_training : True})
                train_writer.add_summary(s, i)

                if i % 20 == 0: # -- evaluate
                    btls, lbls = get_or_create_bottlenecks(sess, bottleneck_tensor, image, loader, anns_valid, 1)
                    a,l,s = sess.run([train['acc'], train['loss'], merged], feed_dict={input_tensor : btls, label_tensor: lbls, is_training : False})
                    valid_writer.add_summary(s, i)
                    print('%d ) Loss : %.3f, Acc : %.2f' % (i, l, a))
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
                    sess, sess.graph.as_graph_def(), [train['pred'].name[:-2]] ) # strip :0
            with gfile.FastGFile(output_graph_path, 'wb') as f:
              f.write(output_graph_def.SerializeToString())
            with gfile.FastGFile(output_labels_path, 'w') as f:
              f.write('\n'.join(categories) + '\n')

if __name__ == '__main__':
  tf.app.run()

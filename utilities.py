import tensorflow as tf

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

class Distorter(object):
    def __init__(self, image):
        self.image = image
        self.bbox = tf.placeholder(tf.float32, [None, 4])
        self.label = tf.placeholder(tf.float32,  [None])
        self.d_image, self.d_bbox, self.d_label = add_input_distortions(tf.expand_dims(self.image,0), self.bbox, self.label)
        self.d_label = tf.squeeze(self.d_label, [-1])
    def apply(self, sess, im_in, bbox_in, label_in):
        while True:
            d_im, d_bb, d_lbl = sess.run([self.d_image, self.d_bbox, self.d_label], feed_dict={self.image : im_in, self.bbox : bbox_in, self.label : label_in})
            if len(d_lbl) > 0:
                break
        return d_im, d_bb, d_lbl


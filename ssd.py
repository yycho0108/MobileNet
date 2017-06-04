import tensorflow as tf
slim = tf.contrib.slim
import numpy as np
from utilities import variable_summaries

def default_box(output_tensor, box_ratios, scale=1.0, wildcard=1.0):
    # output_tensor = [b,h,w,c]
    # box_ratios = [n]
    # y1,x1,y2,x2

    box_ratios = map(lambda b : np.sqrt(b), box_ratios)
    box_ratios = map(lambda b : [1.0/b, b, 1.0/b, b], box_ratios)
    box_ratios = np.array(box_ratios)
    box_ratios[0] *= wildcard

    s = output_tensor.get_shape().as_list()
    h,w = s[1], s[2]

    #ch,cw = 1./h, 1./w # cell width
    ch,cw = scale, scale

    one_cell = np.atleast_2d([-ch/2,-cw/2,ch/2,cw/2]) # dims of one cell

    cell_box_dims = box_ratios * one_cell #(nx4)
    grid_box_dims = np.tile(cell_box_dims, (h,w,1,1)) # should be (h,w,n,4)

    grid_box_locs = np.asarray(
            np.meshgrid(
                np.multiply(1./h, range(h)),
                np.multiply(1./w, range(w)),
                indexing='xy'
                )
            ).T
    grid_box_locs += [ch/2,cw/2]
    grid_box_locs = np.tile(grid_box_locs, (1,1,2))

    grid_box = np.expand_dims(grid_box_locs, 2) + grid_box_dims

    # clip
    grid_box[grid_box<0.0] = 0.0
    grid_box[grid_box>1.0] = 1.0
    return grid_box

def overlap(a, b):
    a_y1,a_x1,a_y2,a_x2 = a
    b_y1,b_x1,b_y2,b_x2 = b
    return max(0, min(a_x2,b_x2) - max(a_x1,b_x1)) * max(0, min(a_y2,b_y2) - max(a_y1,b_y1))

def rect_area(r):
    y1,x1,y2,x2 = r
    return (x2-x1)*(y2-y1)

def jaccard(a,b):
    i = overlap(a,b) # intersection
    u = rect_area(a) + rect_area(b) - i
    eps = 1e-9
    return i/(u+eps)

def calc_offsets(src, dst):
    s_y1,s_x1,s_y2,s_x2 = src
    d_y1,d_x1,d_y2,d_x2 = dst
    eps = 1e-9

    dy = 0.5 * ((d_y1+d_y2)-(s_y1+s_y2))
    dx = 0.5 * ((d_x1+d_x2)-(s_x1+s_x2))
    dh = np.sqrt((d_y2 - d_y1) / (eps + s_y2 - s_y1)) # adding small value to prevent instability, just in case
    dw = np.sqrt((d_x2 - d_x1) / (eps + s_x2 - s_x1))
    return dy,dx,dh,dw

def create_label(gt_boxes, gt_labels, l_d_boxes, n_classes):
    ## --> static label, deprecated
    # gt_boxes = [n, 5] (bbox+class)
    # d_boxes = list([h,w,n,4])
    # result = list([h,w,n,4+n_classes])# --> per bottleneck tensor; each bottleneck tensor always gets constant dims
    # for d_box in d_boxes:
    # ... for gt_box in gt_boxes:
    #         if(iou(d_box, gt_box[:4]) > result[i,j,5]){

    #         }
    # ...     result[i,j,k,:4] = calc_offsets(d_box, gt_box)
    #         result[i,j,k,4] = updated_iou
    #         result[i,j,k,5] = gt_box[4]

    l_label = []
    for d_boxes in l_d_boxes:
        h,w,n = d_boxes.shape[:3]
        label = np.zeros((h,w,n,4+n_classes))

        for gt_box, gt_label in zip(gt_boxes, gt_labels): # loop over ground truths
            top = 0.0
            top_idx = (0,0,0)
            top_offsets = (0,0,0,0)

            for i in range(h):
                for j in range(w):
                    for k in range(n): # k = default box idx, loop over default boxes
                        d_box = d_boxes[i,j,k]
                        iou = jaccard(d_box, gt_box)
                        offsets = calc_offsets(d_box, gt_box)

                        if iou > top: # best match
                            top = iou
                            top_idx = (i,j,k)
                            top_offsets = offsets

                        if iou > 0.5: # good match
                            cls_idx = 4 + gt_label 
                            label[i,j,k,cls_idx] = 1.0
                            label[i,j,k,:4] = offsets # localization offsets

            if top < 0.5: # no good match, match at least one
                i,j,k = top_idx
                label[i,j,k] = 1.0
                label[i,j,k,:4] = top_offsets

        l_label.append(label)

    return l_label

#def overlap(a, b):
#    a_y1,a_x1,a_y2,a_x2 = a
#    b_y1,b_x1,b_y2,b_x2 = b
#    return max(0, min(a_x2,b_x2) - max(a_x1,b_x1)) * max(0, min(a_y2,b_y2) - max(a_y1,b_y1))

def gather_axis(params, indices, axis=0):
    return tf.stack(tf.unstack(tf.gather(tf.unstack(params, axis=axis), indices)), axis=axis)

def create_label_tf(gt_boxes, gt_split_tensor, gt_label_tensor, d_box):
    # b = batch_size
    # n = # gt boxes
    # m = # default boxes
    # h = height of output tensor
    # w = width of output tensor

    #gt_boxes : tensor of [b*n, 4]

    #output_tensor : tensor of [b, h, w, m, 4 + n_classes], m = # boxes
    # pr_boxes, pr_labels = tf.split(output_tensors, [4, n_classes], axis=-1)
    # pr_boxes :  tensor of [b, h, w, m, 4]
    # pr_labels : tensor of [b, h, w, m, n_classes]

    #print 'c0', np.sum((d_box[:,2] - d_box[:,0]) == 0)
    #print 'c1', np.sum((d_box[:,3] - d_box[:,1]) == 0)

    d_box = tf.constant(d_box, tf.float32)#np.reshape(default_box(output_tensor, box_ratios), (-1,4)), tf.float32)

    lr = tf.maximum(
            tf.minimum(gt_boxes[:,3], d_box[:,3,None]) -
            tf.maximum(gt_boxes[:,1], d_box[:,1,None]),
            0
            )
            
    tb = tf.maximum(
            tf.minimum(gt_boxes[:,2], d_box[:,2,None]) -
            tf.maximum(gt_boxes[:,0], d_box[:,0,None]),
            0
            )

    ixn = tf.multiply(tb,lr) # intersection

    unn = tf.subtract( # union
            tf.multiply(d_box[:,3,None] - d_box[:,1,None], d_box[:,2,None] - d_box[:,0,None]) + # d_box areas
            tf.multiply(gt_boxes[:,3] - gt_boxes[:,1], gt_boxes[:,2] - gt_boxes[:,0]), # gt_box areas
            ixn
            )

    iou = tf.div(ixn,unn) # (m, b*n)

    # (b*n, m). iou score for each default box AGAINST GT BOXES ... which is undesirable
    # TODO maybe assert sum(gt_split_tensor) == iou.shape[0]

    iou = tf.split(iou, gt_split_tensor, axis=1) # split by gt boxes, -->  b[(m,n)]
    gt_label_tensor = tf.split(gt_label_tensor, gt_split_tensor, axis=0) # --> b[(n)]

    ## --> each gt_label_tensor would be (n,)
    #print 'gs', gt_label_tensor.shape
    #print 'is', tf.argmax(iou, axis=-1).shape

    #cls = tf.map_fn(lambda (g,i) : tf.gather(g,i), [gt_label_tensor, tf.argmax(iou,axis=-1)], dtype=tf.int32)
    #print 'cs', cls.shape

    gt_sel_idx = [tf.argmax(i, axis=-1) for i in iou] # == which gt box to match, per batch

    cls = [tf.gather(g,s) for (g,s) in zip(gt_label_tensor, gt_sel_idx)]
    cls = tf.stack(cls, axis=0)

    # --> (batch_size, num_dbox)
    ## cls labels should look like [b, m] where each default box is matched to one category

    ## select all default boxes that is in good standing with ground truth box
    # iou = b[(m,n)] ... 
    #print 'gs', gt_label_tensor.shape
    #print 'is', tf.argmax(iou, axis=-1).shape

    #cls2 = tf.gather_nd(gt_label_tensor, tf.argmax(iou, axis=-1))
    #print 'cs', cls2.shape

    iou = tf.stack([tf.reduce_max(i, axis=-1) for i in iou],axis=0)
    #iou = tf.reduce_max(iou, axis=-1) # b[(m)] best iou among gt boxes

    best = tf.reduce_max(iou, axis=-1) # match best iou
    best = tf.equal(iou, best[:,None]) #b[(m,n)] TODO: warning : maybe fix with almost_equal?
    good = tf.greater(iou, 0.5)
    sel = tf.logical_or(best, good) # selected ones among default boxes!

    ## offsets
    gt_boxes = tf.split(gt_boxes, gt_split_tensor, axis=0)
    gt_boxes = [tf.gather(g,s) for (g,s) in zip(gt_boxes, gt_sel_idx)]
    gt_boxes = tf.stack(gt_boxes, axis=0) # should be [64,?]

    # gt_boxes == [64,294,4]
    # d_box == [294,4]

    delta = tf.subtract(gt_boxes, d_box[None,:,:])

    dy1, dx1, dy2, dx2 = tf.unstack(delta, axis=-1)

    dy = (dy1+dy2)/2
    dx = (dx1+dx2)/2
    #dw = (dx2-dx1) # TODO _ FIX?
    #dh = (dy2-dy1)
    a0 = tf.assert_greater(d_box[:,3] - d_box[:,1], 0.0)
    a1 = tf.assert_greater(d_box[:,2] - d_box[:,0], 0.0)
    with tf.control_dependencies([a0,a1]):
        dw = tf.log(tf.div(gt_boxes[:,:,3] - gt_boxes[:,:,1], d_box[None,:,3] - d_box[None,:,1]))
        dh = tf.log(tf.div(gt_boxes[:,:,2] - gt_boxes[:,:,0], d_box[None,:,2] - d_box[None,:,0]))
    
    loc = tf.stack([dy,dx,dw,dh], axis=-1)
    return tf.stack(iou, axis=0),  sel, cls, loc 

def smooth_l1(x):
    l2 = 0.5 * (x**2.0)
    l1 = tf.abs(x) - 0.5

    condition = tf.less(tf.abs(x), 1.0)
    re = tf.where(condition, l2, l1)

    return tf.reduce_mean(re, axis=-1)

def pred(output_tensors, df_boxes, num_classes, num_boxes, max_output_size=200, iou_threshold=0.5): # --> NOT per tensor

    s_boxes = []
    s_cls = []
    s_val = [] 

    for box, output in zip(df_boxes, output_tensors):

        out_box,cls = tf.split(output, [4, -1], axis=4)

        cls = tf.nn.softmax(cls)

        b = tf.shape(out_box)[0]

        pred_cls = tf.reshape(tf.argmax(cls, axis=-1), [-1]) # [b,h,w,n]
        pred_val = tf.reshape(tf.reduce_max(cls, axis=-1), [-1])

        box = np.reshape(box, (-1, 4))
        out_box = tf.reshape(out_box, (b, -1,4))

        y1,x1,y2,x2 = np.transpose(box)

        dy,dx,dh,dw = tf.unstack(out_box, axis=-1)

        dh,dw = tf.exp(dh), tf.exp(dw) # undo the sqrt

        cx,cy = (x1+x2)/2, (y1+y2)/2
        w,h = (x2-x1), (y2-y1)

        x,y = cx+dx, cy+dy
        w,h = w * dw, h * dw
        y1,x1,y2,x2 = (y-h/2),(x-w/2),(y+h/2),(x+w/2)

        out_box = tf.reshape(tf.stack([y1,x1,y2,x2], axis=-1), (-1,4))

        s_boxes.append(out_box) # center-based, to simplify calculations
        s_cls.append(pred_cls)
        s_val.append(pred_val)

    s_boxes = tf.concat(s_boxes, axis=0)
    s_cls = tf.concat(s_cls, 0)
    s_val = tf.concat(s_val, 0)
    return s_boxes, s_cls, s_val
    #with tf.control_dependencies([tf.assert_equal(tf.shape(s_boxes)[0], tf.shape(s_cls)[0])]):
    #idx = tf.image.non_max_suppression(s_boxes, s_val, max_output_size = max_output_size, iou_threshold=iou_threshold)
    #return tf.gather(s_boxes, idx), tf.gather(s_cls, idx), tf.gather(s_val, idx)

def train(output, iou, sel, cls, loc, num_classes, pos_neg_ratio=3.0, alpha = 1.0, conf_thresh = 0.5): # --> per tensor
    # output should be something like
    # [b, m, 4 + num_classes]
    # each default box gets matched to best gt box?
    cls_o = tf.one_hot(cls, depth=num_classes) # (64,294,20)
    # sel = [b, m], m = (h*w*num_bbox) -- encodes which default box was selected
    # loc= [b, m, 4]

    #print 'is', iou.shape   # (b, m])
    #print 'ss', sel.shape   # (b, m])
    #print 'cs', cls.shape   # (b, m, 20])
    #print 'ds', loc.shape # (b, m, 4 ])

    batch_size = iou.shape.as_list()[0] # TODO: currently not dynamic

    output = tf.reshape(output, [batch_size, -1, 4+num_classes]) #
    #print 'os', output.shape # [b, m, 24]

    y_loc, y_cls = tf.split(output, [4, num_classes], axis=2) # TODO : -1 may not work?
    # y_loc= [bs, m, 4]
    # y_cls = [bs, m, 20]

    y_pred = tf.argmax(y_cls, -1)# class prediction per default box, [bs, m]
    y_conf = tf.reduce_max(tf.nn.softmax(y_cls), -1)# prediction confidence per default box, [bs, m]

    ### POSITIVE (Object Exists)
    #p_mask = (iou > conf_thresh) # == i.e. object found at default box, [bs, m]
    p_mask = sel # matches, best + good
    p_mask_f = tf.cast(p_mask, tf.float32)
    n_pos = tf.reduce_sum(p_mask_f)

    ### NEGATIVE (No Object)
    n_mask = tf.logical_not(p_mask)#, t_cls > -0.5)
    n_mask_f = tf.cast(n_mask, tf.float32)
    n_neg = tf.reduce_sum(n_mask_f)

    disparity = tf.abs(tf.subtract(iou, y_conf)) * n_mask_f # maximum disagreement!

    k = tf.cast(tf.minimum(n_pos * pos_neg_ratio + tf.cast(batch_size, tf.float32), n_neg), tf.int32)

    neg_val, neg_idx = tf.nn.top_k(tf.reshape(disparity, [-1]), k = k)
    sub_n_mask = tf.logical_and(n_mask, disparity >= neg_val[-1]) # final negative mask
    sub_n_mask_f = tf.cast(sub_n_mask, tf.float32)

    ### COLLECT LOSSES ###
    pos_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_cls, labels=cls) * p_mask_f

    neg_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_cls, labels=cls) * sub_n_mask_f
    #neg_logits = tf.one_hot(tf.cast(y_conf < conf_thresh,tf.int32), 2) # 1=neg, 0=pos
    #neg_labels = tf.logical_and(tf.cast(n_mask, tf.int32) #
    #neg_loss = 2 * tf.nn.sparse_softmax_cross_entropy_with_logits(logits=neg_logits, labels=neg_labels) * sub_n_mask_f

    loc_loss = 50 * smooth_l1(y_loc - loc) * p_mask_f
    #loc_loss = tf.nn.l2_loss(y_loc - loc) * p_mask_f
    #loc_loss = alpha * tf.nn.l2_loss(y_loc - t_loc) * p_mask_f

    with tf.name_scope('losses'):
        pos_loss = tf.where(n_pos>0, tf.reduce_sum(pos_loss)/(n_pos + batch_size), 0)
        tf.summary.scalar('pos', pos_loss)
        tf.losses.add_loss(pos_loss)
        neg_loss = tf.where(k>0, tf.reduce_sum(neg_loss)/tf.cast(k + batch_size,tf.float32), 0)
        tf.summary.scalar('neg', neg_loss)
        tf.losses.add_loss(neg_loss)
        loc_loss = tf.where(n_pos>0, tf.reduce_sum(loc_loss)/(n_pos + batch_size), 0)
        tf.summary.scalar('loc', loc_loss)
        tf.losses.add_loss(loc_loss)

    acc_clf = tf.cast(tf.logical_and(tf.equal(tf.cast(y_pred, tf.int32), cls),p_mask), tf.float32)
    acc_pos = tf.cast(tf.logical_and(p_mask, y_conf > conf_thresh), tf.float32)
    acc_neg = tf.cast(tf.logical_and(n_mask, y_conf < conf_thresh), tf.float32)
    acc_obj = tf.cast(tf.equal(n_mask, y_conf < conf_thresh),  tf.float32)

    with tf.name_scope('counts'):
        tf.summary.scalar('n_pos', n_pos)
        tf.summary.scalar('n_neg', n_neg)
        tf.summary.scalar('k', k)

    with tf.name_scope('debug_acc'):
        tf.summary.scalar('acc_clf', tf.reduce_sum(acc_clf) / n_pos)
        tf.summary.scalar('acc_pos', tf.reduce_sum(acc_pos) / n_pos)
        tf.summary.scalar('acc_neg', tf.reduce_sum(acc_neg) / n_neg)
        tf.summary.scalar('acc_obj', tf.reduce_sum(acc_obj) / n_neg)

    acc_mask = tf.where(p_mask, acc_clf, acc_obj)
    acc = tf.reduce_mean(tf.cast(acc_mask, tf.float32))

    return acc

if __name__ == "__main__":

    box_ratios = [1.0, 2.0, 3.0, 0.5, 1.0/3.0]
    num_classes = 20

    output_tensors = [
            tf.placeholder(tf.float32, [None, 3, 3, 21]),
            tf.placeholder(tf.float32, [None, 5, 5, 21]),
            tf.placeholder(tf.float32, [None, 7, 7, 21])
            ]
    label_tensors = [ 
            tf.placeholder(tf.float32, [None, 3, 3, len(box_ratios), 4+num_classes])# coord(dx,dy,dw,dh), iou, class

            ]
    boxes = [default_box(o, box_ratios) for o in output_tensors]
    gt_boxes = [ [0.1, 0.2, 0.3, 0.5] for _ in range(2)]
    gt_labels = [2 for _ in range(2)]
    l_label = create_label(gt_boxes, gt_labels, boxes, num_classes)

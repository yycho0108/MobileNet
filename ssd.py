import tensorflow as tf
import numpy as np
from utilities import variable_summaries

def default_box(output_tensor, box_ratios):
    # output_tensor = [b,h,w,c]
    # box_ratios = [n]
    # y1,x1,y2,x2

    box_ratios = map(lambda b : np.sqrt(b), box_ratios)
    box_ratios = map(lambda b : [1./b, b, 1./b, b], box_ratios)
    box_ratios = np.array(box_ratios)

    s = output_tensor.get_shape().as_list()
    h,w = s[1], s[2]
    ch,cw = 1./h, 1./w # cell width
    one_cell = np.atleast_2d([-ch/2,-cw/2,ch/2,cw/2]) # dims of one cell

    cell_box_dims = box_ratios * one_cell #(nx4)
    grid_box_dims = np.tile(cell_box_dims, (h,w,1,1)) # should be (h,w,n,4)

    grid_box_locs = np.asarray(
            np.meshgrid(
                np.multiply(ch, range(h)),
                np.multiply(cw, range(w)),
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


def create_label_tf(gt_boxes, output_tensor, n_classes, box_ratios):
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

    d_box = tf.constant(np.reshape(default_box(output_tensor, box_ratios), (-1,4)), tf.float32)
    gt_box = tf.reshape(gt_boxes, (-1,4))

    lr = tf.maximum(
            tf.minimum(gt_box[:,3,None], d_box[:,3]) -
            tf.maximum(gt_box[:,1,None], d_box[:,1]),
            0
            )
    tb = tf.maximum(
            tf.minimum(gt_box[:,2,None], d_box[:,2]) -
            tf.maximum(gt_box[:,0,None], d_box[:,0]),
            0
            )

    ixn = tf.multiply(tb,lr) # intersection

    unn = tf.subtract( # union
            tf.multiply(d_box[:,3] - d_box[:,1], d_box[:,2] - d_box[:,0]) + # d_box areas
            tf.multiply(gt_box[:,3,None] - gt_box[:,1,None], gt_box[:,2,None] - gt_box[:,0,None]), # gt_box areas
            ixn
            )

    iou = tf.div(ixn,unn)
    # (m,b*n). iou score for each default box.
    best = tf.reduce_max(iou, axis=-1) # match best
    best = tf.equal(iou, best)
    good = tf.greater(iou, 0.5) # match good
    sel = tf.logical_or(best, good)

    ## offsets
    dy1,dx1,dy2,dx2 = [tf.subtract(gt_box[:,i,None], d_box[:,i]) for i in range(4)]
    dy = (dy1+dy2)/2
    dx = (dx1+dx2)/2
    dw = tf.div(gt_box[:,3,None] - gt_box[:,1,None], d_box[:,3] - d_box[:,1])
    dh = tf.div(gt_box[:,2,None] - gt_box[:,0,None], d_box[:,2] - d_box[:,0])
    
    delta = tf.stack([dy,dx,dw,dh], axis=-1)
    
    return iou, sel, delta

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

        dh,dw = dh*dh, dw*dw # undo the sqrt

        cx,cy = (x1+x2)/2, (y1+y2)/2
        w,h = (x2-x1), (y2-y1)

        x,y = cx+dx, cy+dy
        w,h = w*dw, h*dw
        y1,x1,y2,x2 = (y-h/2),(x-w/2),(y+h/2),(x+w/2)

        out_box = tf.reshape(tf.stack([y1,x1,y2,x2], axis=-1), (-1,4))

        s_boxes.append(out_box) # center-based, to simplify calculations
        s_cls.append(pred_cls)
        s_val.append(pred_val)

    s_boxes = tf.concat(s_boxes, -1)
    s_cls = tf.concat(s_cls, -1)
    s_val = tf.concat(s_val, -1)

    #with tf.control_dependencies([tf.assert_equal(tf.shape(s_boxes)[0], tf.shape(s_cls)[0])]):
    idx = tf.image.non_max_suppression(s_boxes, s_val, max_output_size = max_output_size, iou_threshold=iou_threshold)
    return tf.gather(s_boxes, idx), tf.gather(s_cls, idx), tf.gather(s_val, idx)

def eval(output, target, num_classes, pos_neg_ratio=3.0, alpha = 0.03, conf_thresh = 0.5): # --> per tensor
    # output = [b*h*w*n, 4 + num_classes] ???
    # target = [b*h*w*n, 4 + num_classes] ???

    batch_size = tf.shape(output)[0]

    y_loc, y_cls = tf.split(output, [4, num_classes], axis=4) # TODO : -1 may not work?
    t_loc, t_cls = tf.split(target, [4, num_classes], axis=4)

    y_pred = tf.argmax(y_cls, -1)
    t_pred = tf.argmax(t_cls, -1)

    y_conf = tf.reduce_max(y_cls, -1)
    t_conf = tf.reduce_max(t_cls, -1) # still per-box

    ### POSITIVE (Object Exists)
    p_mask = (t_conf > conf_thresh) # == i.e. object found at location
    p_mask_f = tf.cast(p_mask, tf.float32)
    n_pos = tf.reduce_sum(p_mask_f)

    ### NEGATIVE (No Object)
    n_mask = tf.logical_not(p_mask)#, t_cls > -0.5)
    n_mask_f = tf.cast(n_mask, tf.float32)
    n_neg = tf.reduce_sum(n_mask_f)
    disparity = tf.abs(tf.reduce_max(tf.subtract(t_cls, y_cls),-1) * n_mask_f) # maximum disagreement

    k = tf.cast(tf.minimum(n_pos * pos_neg_ratio + tf.cast(batch_size, tf.float32), n_neg), tf.int32)

    #with tf.name_scope('counts'):
    #    tf.summary.scalar('n_pos', n_pos)
    #    tf.summary.scalar('n_neg', n_neg)
    #    tf.summary.scalar('k', k)

    neg_val, neg_idx = tf.nn.top_k(tf.reshape(disparity, [-1]), k = k)
    n_mask = tf.logical_and(n_mask, disparity > neg_val[-1]) # final negative mask?
    n_mask_f = tf.cast(n_mask, tf.float32)

    ### COLLECT LOSSES ###
    pos_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_cls, labels=t_pred) * p_mask_f
    neg_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_cls, labels=tf.cast(p_mask, tf.int32)) * n_mask_f

    loc_loss = smooth_l1(y_loc - t_loc) * p_mask_f
    #loc_loss = alpha * tf.nn.l2_loss(y_loc - t_loc) * p_mask_f

    with tf.name_scope('losses'):
        pos_loss = tf.reduce_mean(pos_loss)
        tf.summary.scalar('pos', pos_loss)

        neg_loss = tf.reduce_mean(neg_loss)
        tf.summary.scalar('neg', neg_loss)

        loc_loss = tf.reduce_mean(loc_loss)
        tf.summary.scalar('loc', loc_loss)

    #acc_mask = tf.where(p_mask, tf.equal(y_pred, t_pred), tf.equal(n_mask, (y_conf < conf_thresh)))
    acc_mask = tf.where(p_mask, tf.equal(y_pred, t_pred), y_conf < conf_thresh)
    #acc_mask = tf.logical_or(
    #        tf.logical_and(p_mask, tf.equal(y_pred,t_pred) ), # when positive, look at class prediction
    #        tf.logical_and(n_mask, y_conf < conf_thresh    )  # when negative, look at presence prediction
    #        )

    acc = tf.reduce_mean(tf.cast(acc_mask, tf.float32))

    return (pos_loss + neg_loss + loc_loss), acc

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

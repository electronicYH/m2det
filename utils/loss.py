import tensorflow as tf

def calc_focal_loss(cls_outputs, cls_targets, alpha=0.25, gamma=2.0):
    """
    Args:
        cls_outputs: [batch_size, num_anchors, num_classes]
        cls_targets: [batch_size, num_anchors, num_classes]
    Returns:
        cls_loss: [batch_size]

    Compute focal loss:
        FL = -(1 - pt)^gamma * log(pt), where pt = p if y == 1 else 1 - p
        cf. https://arxiv.org/pdf/1708.02002.pdf
    """
    positive_mask = tf.equal(cls_targets, 1.0)
    pos = tf.where(positive_mask, 1.0 - cls_outputs, tf.zeros_like(cls_outputs))
    neg = tf.where(positive_mask, tf.zeros_like(cls_outputs), cls_outputs)
    pos_loss = - alpha * tf.pow(pos, gamma) * tf.log(tf.clip_by_value(cls_outputs, 1e-15, 1.0))
    neg_loss = - (1 - alpha) * tf.pow(neg, gamma) * tf.log(tf.clip_by_value(1.0 - cls_outputs, 1e-15, 1.0))
    loss = tf.reduce_sum(pos_loss + neg_loss, axis=[1, 2])
    return loss
    
def calc_box_loss(box_outputs, box_targets, obj_mask, delta=0.1):
    error = box_targets - box_outputs
    sq_loss = 0.5 * tf.pow(error, 2)
    abs_loss = delta * (tf.abs(error) - 0.5 * delta)
    l1_loss = tf.where(tf.less(tf.abs(error), delta), sq_loss, abs_loss)
    box_loss = tf.reduce_sum(l1_loss, axis=-1)
    box_loss = tf.reduce_sum(box_loss * obj_mask, axis=-1)

    return box_loss

def calc_loss(y_true, y_pred, box_loss_weight=50.0):
    """
    Args:
        y_true: [batch_size, num_anchors, 4 + num_classes + 1]
        y_pred: [batch_size, num_anchors, 4 + num_classes]
            num_classes is including the back-ground class
            last element of y_true denotes if the box is positive or negative:
    Returns:
        total_loss:

    cf. https://github.com/tensorflow/tpu/blob/master/models/official/retinanet/retinanet_model.py
    """
    
    box_outputs = y_pred[:, :, :4]
    box_targets = y_true[:, :, :4]
    cls_outputs = y_pred[:, :, 4:]
    cls_targets = y_true[:, :, 4:-1]
    obj_mask = y_true[:, :, -1]

    box_loss = calc_box_loss(box_outputs, box_targets, obj_mask)
    cls_loss = calc_focal_loss(cls_outputs, cls_targets)

    total_loss = cls_loss + box_loss_weight * box_loss

    return tf.reduce_mean(total_loss)

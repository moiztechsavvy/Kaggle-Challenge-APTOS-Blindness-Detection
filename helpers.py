import tensorflow as tf
#from slack import slacktracker

def kappa_loss(y_pred, y_true, y_pow=2, eps=1e-10, N=5, bsize=128, name='kappa'):
 """A continuous differentiable approximation of discrete kappa loss.
     Args:
         y_pred: 2D tensor or array, [batch_size, num_classes]
         y_true: 2D tensor or array,[batch_size, num_classes]
         y_pow: int,  e.g. y_pow=2
         N: typically num_classes of the model
         bsize: batch_size of the training or validation ops
         eps: a float, prevents divide by zero
         name: Optional scope/name for op_scope.
     Returns:
         A tensor with the kappa loss."""
 with tf.name_scope(name):
     y_true = tf.cast(y_true,dtype='float')
     repeat_op = tf.cast(tf.tile(tf.reshape(tf.range(0, N), [N, 1]), [1, N]), dtype='float')
     repeat_op_sq = tf.square((repeat_op - tf.transpose(repeat_op)))
     weights = repeat_op_sq / tf.cast((N - 1) ** 2, dtype='float')

     pred_ = y_pred ** y_pow
     try:
         pred_norm = pred_ / (eps + tf.reshape(tf.reduce_sum(pred_, 1), [-1, 1]))
     except Exception:
         pred_norm = pred_ / (eps + tf.reshape(tf.reduce_sum(pred_, 1), [bsize, 1]))

     hist_rater_a = tf.reduce_sum(pred_norm, 0)
     hist_rater_b = tf.reduce_sum(y_true, 0)

     conf_mat = tf.matmul(tf.transpose(pred_norm), y_true)

     nom = tf.reduce_sum(weights * conf_mat)
     denom = tf.reduce_sum(weights * tf.matmul(
         tf.reshape(hist_rater_a, [N, 1]), tf.reshape(hist_rater_b, [1, N])) /
                           tf.cast(bsize, dtype='float'))
     return nom / (denom + eps)

#callback for sending values to slack
# class Slack(tf.keras.callbacks.Callback):
#         def on_epoch_end(self,epoch, logs={}):
#             print(logs.get('acc'))
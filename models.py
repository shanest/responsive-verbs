import tensorflow as tf


def basic_ffnn(features, labels, mode, params):

    num_verbs = len(params['verbs'])
    training = mode == tf.estimator.ModeKeys.TRAIN

    # -- inputs: [batch_size, item_size]
    inputs = features[params['input_feature']]
    # -- labels: [batch_size]

    net = inputs
    for layer in params['layers']:
        # TODO: dropout?
        net = tf.layers.dense(net,
                              units=layer['units'],
                              activation=layer['activation'])
        if layer['dropout']:
            net = tf.layers.dropout(net,
                                    rate=layer['dropout'],
                                    training=training)
        # -- net: [batch_size, params['layers'][-1]['units']]

    # -- logits: [batch_size, num_classes]
    logits = tf.layers.dense(net, units=params['num_classes'], activation=None)

    # prediction
    # -- predicted_classes: [batch_size]
    predicted_classes = tf.argmax(logits, axis=1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes[:, tf.newaxis],
            'probabilities': tf.nn.softmax(logits),
            'logits': logits,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # loss and training
    loss = tf.losses.sparse_softmax_cross_entropy(labels, logits)
    # TODO: parameterize optimizer?
    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    # -- true_labels: [batch_size]
    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=predicted_classes)
    metrics = {'total_accuracy': accuracy}

    # per-verb metrics
    # -- verb_by_input: [batch_size, num_verbs]
    verb_by_input = inputs[:, -num_verbs:]
    # -- verb_indices: [batch_size]
    verb_indices = tf.to_int32(tf.argmax(verb_by_input, axis=1))
    # -- prediction_by_verb: a list num_verbs long
    # -- prediction_by_verb[i]: Tensor of predictions for verb i
    prediction_by_verb = tf.dynamic_partition(
        predicted_classes, verb_indices, num_verbs)
    # -- label_by_verb: a list num_verbs long
    # -- label_by_verb[i]: Tensor containing true for verb i
    label_by_verb = tf.dynamic_partition(
        labels, verb_indices, num_verbs)
    for idx in xrange(num_verbs):
        # TODO: loss by verb as well?
        key = '{}_accuracy'.format(params['verbs'][idx].__name__)
        metrics[key] = tf.metrics.accuracy(labels=label_by_verb[idx],
                                           predictions=prediction_by_verb[idx])

    return tf.estimator.EstimatorSpec(mode,
                                      loss=loss,
                                      train_op=train_op,
                                      eval_metric_ops=metrics)

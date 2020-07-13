"""
Copyright (C) 2018 Shane Steinert-Threlkeld

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>
"""
from __future__ import division

import tensorflow as tf


def F1_metric(labels, predictions):

    precision_scalar, precision_update = tf.metrics.precision(
        labels=labels, predictions=predictions
    )
    recall_scalar, recall_update = tf.metrics.recall(
        labels=labels, predictions=predictions
    )
    F1_scalar = (
        2 * (precision_scalar * recall_scalar) / (precision_scalar + recall_scalar)
    )
    return F1_scalar, tf.group(precision_update, recall_update)


def basic_ffnn(features, labels, mode, params):

    num_verbs = len(params["verbs"])
    training = mode == tf.estimator.ModeKeys.TRAIN

    # -- inputs: [batch_size, item_size]
    inputs = features[params["input_feature"]]
    batch_size = tf.shape(inputs)[0]
    # -- labels: [batch_size]
    # -- verb_by_input: [batch_size, num_verbs]
    verb_by_input = inputs[:, -num_verbs:]
    # -- verb_indices: [batch_size]
    verb_indices = tf.to_int32(tf.argmax(verb_by_input, axis=1))

    net = inputs
    for layer in params["layers"]:
        # TODO: dropout?
        net = tf.layers.dense(net, units=layer["units"], activation=layer["activation"])
        if layer["dropout"]:
            net = tf.layers.dropout(net, rate=layer["dropout"], training=training)
        # -- net: [batch_size, params['layers'][-1]['units']]

    # -- logits: [batch_size, num_classes]
    logits = tf.layers.dense(net, units=params["num_classes"], activation=None)

    # prediction
    # -- predicted_classes: [batch_size]
    predicted_classes = tf.argmax(logits, axis=1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        # TODO: read dox_w in P, w in dox_w, decl vs int, etc, off the inputs
        verb_names = [v.__name__ for v in params["verbs"]]
        num_worlds = params["num_worlds"]
        # -- dox_w: [batch_size, num_worlds]
        dox_w = inputs[:, -(num_verbs + num_worlds) : -num_verbs]
        # -- world: [batch_size, num_worlds]
        world = inputs[:, -(num_verbs + 2 * num_worlds) : -(num_verbs + num_worlds)]
        # -- embedding: [batch_size, num_worlds*num_worlds]
        embedding = inputs[:, : num_worlds ** 2]
        # -- complements: [batch_size, num_worlds, num_worlds]
        # complements[i][w] is embedding(w) for each w
        complements = tf.stack(
            tf.split(embedding, num_or_size_splits=num_worlds, axis=1), axis=1
        )
        # -- interrogative: [batch_size]
        # whether or not the complement was interrogative
        interrogative = tf.reduce_min(tf.reduce_max(complements, axis=2), axis=1)
        # -- dox_w_expand: [batch_size, num_world, num_worlds]
        # TODO: document this dox_in_p logic!
        dox_w_expand = tf.reshape(
            tf.tile(dox_w, [1, num_worlds]), [-1, num_worlds, num_worlds]
        )
        dox_w_mult = complements * dox_w_expand
        dox_w_equal = tf.equal(dox_w_mult, dox_w_expand)
        dox_in_p = tf.reduce_any(tf.reduce_all(dox_w_equal, axis=2), axis=1)

        predictions = {
            "class_ids": predicted_classes,
            "probabilities": tf.nn.softmax(logits),
            "logits": logits,
            "verb": tf.gather(verb_names, verb_indices),
            # dox_w * w: all 0s unless w in dox_w, in which case there is a
            # single 1; so summing along the row gives the right verdict
            "w_in_dox": tf.reduce_max(dox_w * world, axis=1),
            "interrogative": interrogative,
            "dox_in_p": dox_in_p,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # loss and training
    loss = tf.losses.sparse_softmax_cross_entropy(labels, logits)
    # TODO: parameterize optimizer?
    optimizer = tf.train.RMSPropOptimizer(0.001)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    # -- true_labels: [batch_size]
    accuracy = tf.metrics.accuracy(labels=labels, predictions=predicted_classes)
    metrics = {"total_accuracy": accuracy}

    # per-verb metrics
    # -- prediction_by_verb: a list num_verbs long
    # -- prediction_by_verb[i]: Tensor of predictions for verb i
    prediction_by_verb = tf.dynamic_partition(
        predicted_classes, verb_indices, num_verbs
    )
    # -- label_by_verb: a list num_verbs long
    # -- label_by_verb[i]: Tensor containing true for verb i
    label_by_verb = tf.dynamic_partition(labels, verb_indices, num_verbs)
    for idx in range(num_verbs):
        # TODO: loss by verb as well?
        verb_name = params["verbs"][idx].__name__
        acc_key = "{}_accuracy".format(verb_name)
        metrics[acc_key] = tf.metrics.accuracy(
            labels=label_by_verb[idx], predictions=prediction_by_verb[idx]
        )
        F1_key = "{}_F1".format(verb_name)
        metrics[F1_key] = F1_metric(
            labels=label_by_verb[idx], predictions=prediction_by_verb[idx]
        )
        metrics["{}_tp".format(verb_name)] = tf.metrics.true_positives(
            label_by_verb[idx], prediction_by_verb[idx]
        )
        metrics["{}_tn".format(verb_name)] = tf.metrics.true_negatives(
            label_by_verb[idx], prediction_by_verb[idx]
        )
        metrics["{}_fp".format(verb_name)] = tf.metrics.false_positives(
            label_by_verb[idx], prediction_by_verb[idx]
        )
        metrics["{}_fn".format(verb_name)] = tf.metrics.false_negatives(
            label_by_verb[idx], prediction_by_verb[idx]
        )

    return tf.estimator.EstimatorSpec(
        mode, loss=loss, train_op=train_op, eval_metric_ops=metrics
    )

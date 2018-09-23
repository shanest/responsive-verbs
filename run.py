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
from __future__ import print_function
from collections import defaultdict
import argparse

import verbs
import util
import data
from models import basic_ffnn

import tensorflow as tf
import pandas as pd

tf.logging.set_verbosity(tf.logging.INFO)


class EvalEarlyStopHook(tf.train.SessionRunHook):
    """Evaluates estimator during training and implements early stopping.
    Writes output of a trial as CSV file.
    See https://stackoverflow.com/questions/47137061/. """

    def __init__(self, estimator, eval_input, filename,
                 num_steps=50, stop_loss=0.02):

        self._estimator = estimator
        self._input_fn = eval_input
        self._num_steps = num_steps
        self._stop_loss = stop_loss
        # store results of evaluations
        self._results = defaultdict(list)
        self._filename = filename

    def begin(self):

        self._global_step_tensor = tf.train.get_or_create_global_step()
        if self._global_step_tensor is None:
            raise ValueError("global_step needed for EvalEarlyStop")

    def before_run(self, run_context):

        requests = {'global_step': self._global_step_tensor}
        return tf.train.SessionRunArgs(requests)

    def after_run(self, run_context, run_values):

        global_step = run_values.results['global_step']
        if (global_step-1) % self._num_steps == 0:
            ev_results = self._estimator.evaluate(input_fn=self._input_fn)

            print('')
            for key, value in ev_results.items():
                self._results[key].append(value)
                print('{}: {}'.format(key, value))

            # TODO: add running total accuracy or other complex stop condition?
            if ev_results['loss'] < self._stop_loss:
                run_context.request_stop()

    def end(self, session):
        # write results to csv
        util.dict_to_csv(self._results, self._filename)


def run_trial(eparams, hparams, trial_num,
              write_path='/tmp/tf/verbs'):

    tf.reset_default_graph()

    write_dir = '{}/trial_{}'.format(write_path, trial_num)
    csv_file = '{}/trial_{}.csv'.format(write_path, trial_num)

    # BUILD MODEL
    run_config = tf.estimator.RunConfig(
        save_checkpoints_steps=eparams['eval_steps'],
        save_checkpoints_secs=None,
        save_summary_steps=eparams['eval_steps'])

    # TODO: moar models?
    model = tf.estimator.Estimator(
        model_fn=basic_ffnn,
        params=hparams,
        model_dir=write_dir,
        config=run_config)

    # GENERATE DATA
    generator = data.DataGenerator(
        hparams['verbs'], eparams['num_worlds'], eparams['max_cells'],
        eparams['items_per_bin'], eparams['tries_per_bin'],
        eparams['test_bin_size'])

    train_x, train_y = generator.get_training_data()
    test_x, test_y = generator.get_test_data()

    # input fn for training
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={hparams['input_feature']: train_x},
        y=train_y,
        batch_size=eparams['batch_size'],
        num_epochs=eparams['num_epochs'],
        shuffle=True)

    # input fn for evaluation
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={hparams['input_feature']: test_x},
        y=test_y,
        batch_size=len(test_x),
        shuffle=False)

    print('\n------ TRIAL {} -----'.format(trial_num))

    if eparams['train']:
        print('\n-- TRAINING --')
        # train and evaluate model together, using the Hook
        model.train(input_fn=train_input_fn,
                    hooks=[EvalEarlyStopHook(model, eval_input_fn, csv_file,
                                             eparams['eval_steps'],
                                             eparams['stop_loss'])])

    if eparams['predict']:
        print('\n-- PREDICTING --')
        predictions = pd.DataFrame(model.predict(input_fn=eval_input_fn))
        predictions['true_label'] = test_y
        predictions['correct'] = (predictions['class_ids'] ==
                                  predictions['true_label'])
        predictions.to_csv(
            '{}/trial_{}_predictions.csv'.format(write_path, trial_num))


# DEFINE AN EXPERIMENT
def main_experiment(eparams, hparams):
    for trial in xrange(eparams['num_trials']):
        run_trial(eparams, hparams, trial, eparams['write_dir'])


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--out_path', help='path to output', type=str,
                        default='data/')
    # what to do arguments
    parser.add_argument('--no_train', dest='train', action='store_false')
    parser.add_argument('--train', dest='train', action='store_true')
    parser.set_defaults(train=True)
    parser.add_argument('--no_eval', dest='eval', action='store_false')
    parser.add_argument('--eval', dest='eval', action='store_true')
    parser.set_defaults(eval=True)
    parser.add_argument('--no_predict', dest='predict', action='store_false')
    parser.add_argument('--predict', dest='predict', action='store_true')
    parser.set_defaults(predict=False)
    args = parser.parse_args()

    eparams = {'write_dir': args.out_path,
               'train': args.train,
               'evaluate': args.eval,
               'predict': args.predict,
               'num_trials': 60,
               'num_epochs': 15,
               'batch_size': 128,
               'num_worlds': 16,
               'max_cells': 4,
               'items_per_bin': 16000,
               'tries_per_bin': 60000,
               'test_bin_size': 4000,
               'eval_steps': 50,
               'stop_loss': 0.02}

    hparams = {'verbs': verbs.get_all_verbs(),
               'num_classes': 2,
               'layers': [
                   {'units': 128,
                    'activation': tf.nn.elu,
                    'dropout': 0.1}]*4,
               'input_feature': 'x'}

    # TODO: parameterize this more
    main_experiment(eparams, hparams)

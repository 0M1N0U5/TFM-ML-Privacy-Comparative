# Copyright 2019, The TensorFlow Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Training a CNN on MNIST with Keras and the DP SGD optimizer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging

import utils
import loadFlags

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

def trainWithoutDP(dataset):
  FLAGS = loadFlags.load(dataset)
  FLAGS.dpsgd = False
  utils.train(FLAGS, dataset)

def trainWithDP(dataset):
  FLAGS = loadFlags.load(dataset)
  noise = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6 ,0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
  #noise = [0.9, 1.0, 1.1, 1.2]
  FLAGS.learning_rate = 0.01
  for i in noise:
    FLAGS.noise_multiplier = i
    utils.train(FLAGS, dataset)

def main(unused_argv):
  logging.set_verbosity(logging.INFO)
  #adult
  datasets = ['mnist', 'ciphar10', 'imdb']
  dataset = datasets[1]
  trainWithoutDP(dataset)
  trainWithDP(dataset)

if __name__ == '__main__':
  app.run(main)

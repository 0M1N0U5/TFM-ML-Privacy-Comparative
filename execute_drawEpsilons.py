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
import matplotlib.pyplot as plt
import numpy as np

import utils
import loadDatasets as ld
import loadModel as lm
import loadOptimizer as lo
import loadFlags

def optainEpsilonEpoch(dataset, start_epoch, max_epoch, interval_epoch):
  train_data, train_labels, test_data, test_labels = ld.load(dataset)
  trainSize = len(train_data)
  targetDelta = utils.calculate_delta(trainSize)
  FLAGS = loadFlags.load(dataset)
  epsilon_epoch_x = []
  epsilon_epoch_y = []
  for i in range(start_epoch, max_epoch, interval_epoch):
    FLAGS.epochs = i
    eps = utils.compute_epsilon(FLAGS, FLAGS.epochs * trainSize // FLAGS.batch_size, trainSize, targetDelta)
    eps = float("{:.3f}".format(eps))
    epsilon_epoch_x.append(i)
    epsilon_epoch_y.append(eps)
  return epsilon_epoch_x, epsilon_epoch_y, dataset + "_" + str(trainSize) + "_" + "epoch"

def optainEpsilonNoise(dataset, start_noise, max_noise, interval_noise):
  train_data, train_labels, test_data, test_labels = ld.load(dataset)
  trainSize = len(train_data)
  targetDelta = utils.calculate_delta(trainSize)
  FLAGS = loadFlags.load(dataset)
  epsilon_noise_x = []
  epsilon_noise_y = []
  i = start_noise
  while i <= max_noise:
    FLAGS.noise_multiplier = i
    eps = utils.compute_epsilon(FLAGS, FLAGS.epochs * trainSize // FLAGS.batch_size, trainSize, targetDelta)
    eps = float("{:.3f}".format(eps))
    epsilon_noise_x.append(i)
    epsilon_noise_y.append(eps)
    i += interval_noise
    
  return epsilon_noise_x, epsilon_noise_y, dataset + "_" + str(trainSize) + "_" + "noise"

def obtainEpsilonNoiseAndEpoch(dataset, epoch_array, noise_array):
  if len(epoch_array) != len(noise_array):
    raise ValueError('Both arrays epoch_array and noise_array must have equal length')

  train_data, train_labels, test_data, test_labels = ld.load(dataset)
  trainSize = len(train_data)
  targetDelta = utils.calculate_delta(trainSize)
  epsilon_noise_x = []
  epsilon_noise_y = []
  FLAGS = loadFlags.load(dataset)
  for i in range(len(epoch_array)):
    FLAGS.noise_multiplier = noise_array[i]
    FLAGS.epochs = epoch_array[i]
    eps = utils.compute_epsilon(FLAGS, FLAGS.epochs * trainSize // FLAGS.batch_size, trainSize, targetDelta)
    eps = float("{:.3f}".format(eps))
    epsilon_noise_x.append(str(round(noise_array[i], 2)) + "_" +str(epoch_array[i]))
    epsilon_noise_y.append(eps)

  return epsilon_noise_x, epsilon_noise_y, dataset + "_" + str(trainSize) + "_noiseAndEpoch"

def main(unused_argv):
  logging.set_verbosity(logging.INFO)
  #adult
  datasets = ['mnist', 'ciphar10']
  results_x = []
  results_y = []
  legend = []
  for dataset in datasets:
    #x, y, legendRow = optainEpsilonEpoch(dataset, 1, 16, 1)
    #results_x.append(x)
    #results_y.append(y)
    #legend.append(dataset + "_" + str(legendRow))
    #x, y, legendRow = optainEpsilonNoise(dataset, 0.1, 1, 0.05)
    #results_x.append(x)
    #results_y.append(y)
    #legend.append(dataset + "_" + str(legendRow))
    x, y, legendRow = obtainEpsilonNoiseAndEpoch(dataset, range(5, 650, 50), np.arange(0.1, 1.4, 0.1))
    results_x.append(x)
    results_y.append(y)
    legend.append(dataset + "_" + str(legendRow))

  for i in range(len(results_x)):
    plt.plot(results_x[i], results_y[i])
  
  plt.title('Epsilon evolution')
  plt.ylabel('Epsilon')
  plt.xlabel('Unit')
  plt.legend(legend, loc='upper left')
  plt.show()
  

if __name__ == '__main__':
  app.run(main)

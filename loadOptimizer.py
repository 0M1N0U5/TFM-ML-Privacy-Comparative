import tensorflow as tf
import numpy as np
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasSGDOptimizer
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasAdamOptimizer
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasAdagradOptimizer
from keras import backend as K

def load(FLAGS, dataset):
  if dataset == 'mnist':
    return __load_mnist(FLAGS)
  elif dataset == 'ciphar10':
    return __load_ciphar10(FLAGS)
  elif dataset == 'imdb':
    return __load_imdb(FLAGS)
  else:
    return None

def __load_mnist(FLAGS):
  if FLAGS.dpsgd:
    optimizer = DPKerasSGDOptimizer(
        l2_norm_clip=FLAGS.l2_norm_clip,
        noise_multiplier=FLAGS.noise_multiplier,
        num_microbatches=FLAGS.microbatches,
        learning_rate=FLAGS.learning_rate)
    # Compute vector of per-example loss rather than its mean over a minibatch.
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction=tf.losses.Reduction.NONE)
    return optimizer, loss
  else:
    optimizer = tf.keras.optimizers.SGD(learning_rate=FLAGS.learning_rate)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    return optimizer, loss

def __load_ciphar10(FLAGS):
  if FLAGS.dpsgd:
    optimizer = DPKerasAdamOptimizer(
        l2_norm_clip=FLAGS.l2_norm_clip,
        noise_multiplier=FLAGS.noise_multiplier,
        num_microbatches=FLAGS.microbatches,
        learning_rate=FLAGS.learning_rate)
    # Compute vector of per-example loss rather than its mean over a minibatch.
    loss = tf.keras.losses.CategoricalCrossentropy(reduction=tf.losses.Reduction.NONE)
    return optimizer, loss
  else:
    optimizer = tf.keras.optimizers.Adam(learning_rate=FLAGS.learning_rate)
    loss = tf.keras.losses.CategoricalCrossentropy(reduction=tf.losses.Reduction.NONE)
    return optimizer, loss

def __load_imdb(FLAGS):
  if FLAGS.dpsgd:
    optimizer = DPKerasAdamOptimizer(
        l2_norm_clip=FLAGS.l2_norm_clip,
        noise_multiplier=FLAGS.noise_multiplier,
        num_microbatches=FLAGS.microbatches,
        learning_rate=FLAGS.learning_rate)
    # Compute vector of per-example loss rather than its mean over a minibatch.
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.losses.Reduction.NONE)
    return optimizer, loss
  else:
    optimizer = tf.keras.optimizers.Adam(learning_rate=FLAGS.learning_rate)
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.losses.Reduction.NONE)
    return optimizer, loss

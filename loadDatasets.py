import tensorflow as tf
import numpy as np

def load(dataset):
  if dataset == 'mnist':
    return __load_mnist()
  elif dataset == 'ciphar10':
    return __load_ciphar10()
  elif dataset == 'imdb':
    return __load_IMDB()
  else:
    return None

def __load_mnist():
  """Loads MNIST and preprocesses to combine training and validation data."""
  train, test = tf.keras.datasets.mnist.load_data()
  tf.keras.datasets.boston_housing
  train_data, train_labels = train
  test_data, test_labels = test

  train_data = np.array(train_data, dtype=np.float32) / 255
  test_data = np.array(test_data, dtype=np.float32) / 255

  train_data = train_data.reshape((train_data.shape[0], 28, 28, 1))
  test_data = test_data.reshape((test_data.shape[0], 28, 28, 1))

  train_labels = np.array(train_labels, dtype=np.int32)
  test_labels = np.array(test_labels, dtype=np.int32)

  train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=10)
  test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=10)

  assert train_data.min() == 0.
  assert train_data.max() == 1.
  assert test_data.min() == 0.
  assert test_data.max() == 1.

  return train_data, train_labels, test_data, test_labels

def __load_ciphar10():
  # scale pixels
  def prep_pixels(train, test):
    # convert from integers to floats
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')
    # normalize to range 0-1
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0
    # return normalized images
    return train_norm, test_norm

  (trainX, trainY), (testX, testY) = tf.keras.datasets.cifar10.load_data()

  trainY = tf.keras.utils.to_categorical(trainY)
  testY = tf.keras.utils.to_categorical(testY)

  trainX, testX = prep_pixels(trainX, testX)

  return trainX, trainY, testX, testY

def __load_IMDB():
  (training_data, training_targets), (testing_data, testing_targets) = tf.keras.datasets.imdb.load_data(num_words=10000)
  data = np.concatenate((training_data, testing_data), axis=0)
  targets = np.concatenate((training_targets, testing_targets), axis=0)
  def vectorize(sequences, dimension = 10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
      results[i, sequence] = 1
    return results
  
  data = vectorize(data)
  targets = np.array(targets).astype("float32")

  testX = data[:10000]
  testY = targets[:10000]
  trainX = data[10000:]
  trainY = targets[10000:]
  return trainX, trainY, testX, testY
    
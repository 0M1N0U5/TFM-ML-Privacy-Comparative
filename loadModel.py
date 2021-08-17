import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import Dropout

def load(dataset):
  if dataset == 'mnist':
    return __load_mnist()
  elif dataset == 'ciphar10':
    return __load_ciphar10()
  elif dataset == 'imdb':
    return __load_imdb()
  else:
    return None

def load_from_disk(name, optimizer, loss):
  print("hola")


def __load_mnist():
  # Define a sequential Keras model
  model = tf.keras.Sequential([
      tf.keras.layers.Conv2D(16, 8,
                             strides=2,
                             padding='same',
                             activation='relu',
                             input_shape=(28, 28, 1)),
      tf.keras.layers.MaxPool2D(2, 1),
      tf.keras.layers.Conv2D(32, 4,
                             strides=2,
                             padding='valid',
                             activation='relu'),
      tf.keras.layers.MaxPool2D(2, 1),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(32, activation='relu'),
      tf.keras.layers.Dense(10)
  ])
  return model

  #model = tensorflow.keras.models.load_model('models/mnist-SGD-60-250-66416.54629993437.h5', compile=False)
  #optimizer, loss = lo.load(FLAGS, dataset)
  #model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
  #scores, acc = model.evaluate(test_data, test_labels, verbose=0)
  #format_float = "{:.2f}".format(acc)
  #print("Acc:", format_float)
  #exit(0)


def __load_ciphar10():
  #model =  tf.keras.Sequential()
  #shape = (32, 32, 3)
  #model.add(InputLayer(input_shape=shape))
  #model.add(Conv2D(32, (3, 3), activation='relu'))
  #model.add(MaxPooling2D())
  #model.add(Conv2D(64, (3, 3), activation='relu'))
  #model.add(MaxPooling2D())
  #model.add(Conv2D(64, (3, 3), activation='relu'))
  #model.add(MaxPooling2D())
  #model.add(Flatten())
  #model.add(Dense(128, activation='relu'))
  #model.add(Dense(10))
  #return model

	model = tf.keras.Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Dropout(0.2))
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Dropout(0.2))
	model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dropout(0.2))
	model.add(Dense(10, activation='softmax'))
	return model

def __load_imdb():
  model =  tf.keras.Sequential()
  model.add(tf.keras.layers.Dense(50, activation = "relu", input_shape=(10000, )))
  # Hidden - Layers
  model.add(tf.keras.layers.Dropout(0.3, noise_shape=None, seed=None))
  model.add(tf.keras.layers.Dense(50, activation = "relu"))
  model.add(tf.keras.layers.Dropout(0.2, noise_shape=None, seed=None))
  model.add(tf.keras.layers.Dense(50, activation = "relu"))
  # Output- Layer
  model.add(tf.keras.layers.Dense(1, activation = "sigmoid"))
  return model


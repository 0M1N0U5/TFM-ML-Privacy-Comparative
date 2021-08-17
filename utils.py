import tensorflow
from tensorflow.python.keras.backend_config import epsilon
from tensorflow_privacy.privacy.analysis.rdp_accountant import compute_rdp
from tensorflow_privacy.privacy.analysis.rdp_accountant import get_privacy_spent
import matplotlib.pyplot as plt

from tensorflow.keras.callbacks import ModelCheckpoint

import loadDatasets as ld
import loadModel as lm
import loadOptimizer as lo

def calculate_delta(trainSize):
  return 1/(10**len(str(trainSize)))

def compute_epsilon(FLAGS, steps, trainSize, targetDelta):
  """Computes epsilon value for given hyperparameters."""
  if FLAGS.noise_multiplier == 0.0:
    return float('inf')
  orders = [1 + x / 10. for x in range(1, 100)] + list(range(12, 64)) # Preguntar en reunion
  sampling_probability = FLAGS.batch_size / trainSize
  rdp = compute_rdp(q=sampling_probability,
                    noise_multiplier=FLAGS.noise_multiplier,
                    steps=steps,
                    orders=orders)
  # Delta is set to 1e-5 because MNIST has 60000 training points.
  #return get_privacy_spent(orders, rdp, target_delta=1e-5)[0]
  return get_privacy_spent(orders, rdp, target_delta=targetDelta)[0]

def saveHistory(fileName, history):
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])	
    #plt.plot(history.history['loss'])
    #plt.plot(history.history['val_loss'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    #plt.legend(['train', 'test', 'loss', 'val_loss'], loc='upper left')
    plt.savefig('models/'+fileName+'.png')
    plt.cla()
    plt.clf()

def train(FLAGS, dataset):
  if FLAGS.dpsgd and FLAGS.batch_size % FLAGS.microbatches != 0:
    raise ValueError('Number of microbatches should divide evenly batch_size')
  # Load training and test data.
  diffPriv = "NDP-0.0-" + str(FLAGS.learning_rate) + "-0.0-"
  if FLAGS.dpsgd:
    diffPriv = "DP-" + str(FLAGS.l2_norm_clip) + "-" + str(FLAGS.learning_rate) + "-" + str(FLAGS.noise_multiplier) + "-"

  train_data, train_labels, test_data, test_labels = ld.load(dataset)
  trainSize = len(train_data)
  targetDelta = calculate_delta(trainSize)
  # Load model
  model = lm.load(dataset)
  # Load optimizer and loss
  optimizer, loss = lo.load(FLAGS, dataset)

  # Compile model with Keras
  model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

  filepath = "models/checkpoint-" + dataset + "-" + diffPriv + optimizer._name + "-" + str(trainSize) + "-"+ str(FLAGS.batch_size) + "-" + str(FLAGS.microbatches) + "-{epoch:02d}-{val_accuracy:.6f}-CALCULATE_EPS.hdf5"
  checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
  callbacks_list = [checkpoint]

  # Train model with Keras
  history = model.fit(train_data, train_labels, epochs=FLAGS.epochs, 
    validation_data=(test_data, test_labels), 
    batch_size=FLAGS.batch_size, 
    callbacks=callbacks_list)

  # Compute the privacy budget expended.
  if FLAGS.dpsgd:
    eps = compute_epsilon(FLAGS, FLAGS.epochs * trainSize // FLAGS.batch_size, trainSize, targetDelta)
    FLAGS.eps = "{:.6f}".format(eps)
  else:
    print('Trained with vanilla non-private optimizer')
  scores, acc = model.evaluate(test_data, test_labels, verbose=0)
  format_float = "{:.6f}".format(acc)
  fileName = dataset + "-" + diffPriv + optimizer._name + "-" + str(trainSize) + "-" + str(FLAGS.batch_size) + "-" + str(FLAGS.microbatches) + "-" + str(FLAGS.epochs) + "-" + format_float + "-" + FLAGS.eps
  model.save('models/' + fileName + '.h5')
  text_file = open('models/' + fileName + '.FLAGS.txt', "w")
  text_file.write(FLAGS.flags_into_string())
  text_file.close()
  saveHistory(fileName, history)
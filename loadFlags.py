from absl import flags

def load(dataset):
  if dataset == 'mnist':
    return __load_mnist()
  elif dataset == 'ciphar10':
    return __load_ciphar10()
  elif dataset == 'imdb':
    return __load_imdb()
  else:
    return None

def __load_mnist():
  dpsgd = True
  learning_rate=0.001
  noise_multiplier=0.1
  l2_norm_clip=1.0
  batch_size=250
  epochs=60
  microbatches=250
  model_dir = None
  try:
    flags.DEFINE_boolean(
      'dpsgd', dpsgd, 'If True, train with DP-SGD. If False, '
      'train with vanilla SGD.')
    flags.DEFINE_float('learning_rate', learning_rate, 'Learning rate for training')
    flags.DEFINE_float('noise_multiplier', noise_multiplier, 'Ratio of the standard deviation to the clipping norm')
    flags.DEFINE_float('l2_norm_clip', l2_norm_clip, 'Clipping norm')
    flags.DEFINE_integer('batch_size', batch_size, 'Batch size')
    flags.DEFINE_integer('epochs', epochs, 'Number of epochs')
    flags.DEFINE_integer('microbatches', microbatches, 'Number of microbatches '
        '(must evenly divide batch_size)')
    flags.DEFINE_string('model_dir', model_dir, 'Model directory')
    flags.DEFINE_string('eps', "{:.6f}".format(float('inf')), "Model epsilon")

  except flags.DuplicateFlagError:
    flags.FLAGS.dpsgd = dpsgd
    flags.FLAGS.learning_rate = learning_rate
    flags.FLAGS.noise_multiplier = noise_multiplier
    flags.FLAGS.l2_norm_clip = l2_norm_clip
    flags.FLAGS.batch_size = batch_size
    flags.FLAGS.epochs = epochs
    flags.FLAGS.microbatches = microbatches
    flags.FLAGS.model_dir = model_dir
    flags.FLAGS.eps = "{:.6f}".format(float('inf'))

  return flags.FLAGS

def __load_ciphar10():
  dpsgd = True
  learning_rate=0.001
  noise_multiplier=0.1
  l2_norm_clip=1.0
  batch_size=80
  epochs=40
  microbatches=20
  model_dir = None
  try:
    flags.DEFINE_boolean(
      'dpsgd', dpsgd, 'If True, train with DP-SGD. If False, '
      'train with vanilla SGD.')
    flags.DEFINE_float('learning_rate', learning_rate, 'Learning rate for training')
    flags.DEFINE_float('noise_multiplier', noise_multiplier,
                      'Ratio of the standard deviation to the clipping norm')
    flags.DEFINE_float('l2_norm_clip', l2_norm_clip, 'Clipping norm')
    flags.DEFINE_integer('batch_size', batch_size, 'Batch size')
    flags.DEFINE_integer('epochs', epochs, 'Number of epochs') #60
    flags.DEFINE_integer(
        'microbatches', microbatches, 'Number of microbatches '
        '(must evenly divide batch_size)')
    flags.DEFINE_string('model_dir', model_dir, 'Model directory')
    flags.DEFINE_string('eps', "{:.6f}".format(float('inf')), "Model epsilon")

  except flags.DuplicateFlagError:
    flags.FLAGS.dpsgd = dpsgd
    flags.FLAGS.learning_rate = learning_rate
    flags.FLAGS.noise_multiplier = noise_multiplier
    flags.FLAGS.l2_norm_clip = l2_norm_clip
    flags.FLAGS.batch_size = batch_size
    flags.FLAGS.epochs = epochs
    flags.FLAGS.microbatches = microbatches
    flags.FLAGS.model_dir = model_dir
    flags.FLAGS.eps = "{:.6f}".format(float('inf'))

  return flags.FLAGS

def __load_imdb():
  dpsgd = True
  learning_rate=0.001
  noise_multiplier=0.1
  l2_norm_clip=1.0
  batch_size=100
  epochs=30
  microbatches=25
  model_dir = None
  try:
    flags.DEFINE_boolean(
      'dpsgd', dpsgd, 'If True, train with DP-SGD. If False, '
      'train with vanilla SGD.')
    flags.DEFINE_float('learning_rate', learning_rate, 'Learning rate for training')
    flags.DEFINE_float('noise_multiplier', noise_multiplier,
                      'Ratio of the standard deviation to the clipping norm')
    flags.DEFINE_float('l2_norm_clip', l2_norm_clip, 'Clipping norm')
    flags.DEFINE_integer('batch_size', batch_size, 'Batch size')
    flags.DEFINE_integer('epochs', epochs, 'Number of epochs') #60
    flags.DEFINE_integer(
        'microbatches', microbatches, 'Number of microbatches '
        '(must evenly divide batch_size)')
    flags.DEFINE_string('model_dir', model_dir, 'Model directory')
    flags.DEFINE_string('eps', "{:.6f}".format(float('inf')), "Model epsilon")

  except flags.DuplicateFlagError:
    flags.FLAGS.dpsgd = dpsgd
    flags.FLAGS.learning_rate = learning_rate
    flags.FLAGS.noise_multiplier = noise_multiplier
    flags.FLAGS.l2_norm_clip = l2_norm_clip
    flags.FLAGS.batch_size = batch_size
    flags.FLAGS.epochs = epochs
    flags.FLAGS.microbatches = microbatches
    flags.FLAGS.model_dir = model_dir
    flags.FLAGS.eps = "{:.6f}".format(float('inf'))

  return flags.FLAGS
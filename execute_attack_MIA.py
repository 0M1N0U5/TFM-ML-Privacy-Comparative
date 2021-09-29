from operator import index
import os
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

# evaluate the deep model on the test dataset
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack import membership_inference_attack as mia
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import AttackInputData
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import SlicingSpec
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import AttackType
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack import plotting
from scipy import special
import numpy as np
import loadOptimizer as lo
import loadDatasets as ld

class TMPFLAGS(object):
    def __init__(self):
        self.dpsgd = True
        self.learning_rate=0.001
        self.noise_multiplier=0.1
        self.l2_norm_clip=1.0
        self.batch_size=100
        self.epochs=30
        self.microbatches=25
        self.model_dir = None

def splitedNameToFlags(fSplited):
    flags = TMPFLAGS()
    flags.learning_rate = float(fSplited[4])
    flags.l2_norm_clip = float(fSplited[3])
    flags.batch_size =int(fSplited[8])
    flags.epochs = int(fSplited[10])
    flags.noise_multiplier = float(fSplited[5])
    if flags.noise_multiplier == 0.0:
        flags.dpsgd = False
    return flags

def __load_mnist(dataset, model):
    train_data, train_labels, test_data, test_labels = ld.load(dataset)
    train_labels = train_labels.astype(int)
    test_labels = test_labels.astype(int)
    logits_train = model.predict(train_data)
    logits_test = model.predict(test_data)
    prob_train = special.softmax(logits_train, axis=1)
    prob_test = special.softmax(logits_test, axis=1)
    y_train_onehot = train_labels
    y_test_onehot = test_labels
    cce = tf.keras.backend.categorical_crossentropy
    constant = tf.keras.backend.constant
    loss_train = cce(constant(y_train_onehot), constant(prob_train), from_logits=False).numpy()
    loss_test = cce(constant(y_test_onehot), constant(prob_test), from_logits=False).numpy()
    trainY = np.argmax(train_labels, axis=-1)
    testY = np.argmax(test_labels, axis=-1)

    attack_input = AttackInputData(
        logits_train = logits_train,
        logits_test = logits_test,
        loss_train = loss_train,
        loss_test = loss_test,
        labels_train = trainY,
        labels_test = testY
    )

    return attack_input

def __load_IMDB(dataset, model):
    train_data, train_labels, test_data, test_labels = ld.load(dataset)
    train_labels = train_labels.astype(int)
    test_labels = test_labels.astype(int)
    train_labels = np.reshape(train_labels, (train_labels.shape[0], 1))
    test_labels = np.reshape(test_labels, (test_labels.shape[0], 1))
    logits_train = model.predict(train_data)
    logits_test = model.predict(test_data)
    prob_train = special.softmax(logits_train, axis=1)
    prob_test = special.softmax(logits_test, axis=1)
    y_train_onehot = train_labels
    y_test_onehot = test_labels
    cce = tf.keras.backend.categorical_crossentropy
    constant = tf.keras.backend.constant
    loss_train = cce(constant(y_train_onehot), constant(prob_train), from_logits=False).numpy()
    loss_test = cce(constant(y_test_onehot), constant(prob_test), from_logits=False).numpy()
    trainY = train_labels.ravel()
    testY = test_labels.ravel()
    attack_input = AttackInputData(
        logits_train = logits_train,
        logits_test = logits_test,
        loss_train = loss_train,
        loss_test = loss_test,
        labels_train = trainY,
        labels_test = testY
    )
    return attack_input

def __load_ciphar10(dataset, model):
    train_data, train_labels, test_data, test_labels = ld.load(dataset)
    train_labels = train_labels.astype(int)
    test_labels = test_labels.astype(int)
    logits_train = model.predict(train_data)
    logits_test = model.predict(test_data)
    prob_train = special.softmax(logits_train, axis=1)
    prob_test = special.softmax(logits_test, axis=1)
    y_train_onehot = train_labels
    y_test_onehot = test_labels
    cce = tf.keras.backend.categorical_crossentropy
    constant = tf.keras.backend.constant
    loss_train = cce(constant(y_train_onehot), constant(prob_train), from_logits=False).numpy()
    loss_test = cce(constant(y_test_onehot), constant(prob_test), from_logits=False).numpy()
    uncategorical_train_labels = np.argmax(train_labels, axis=-1)
    uncategorical_test_labels = np.argmax(test_labels, axis=-1)

    attack_input = AttackInputData(
        logits_train = logits_train,
        logits_test = logits_test,
        loss_train = loss_train,
        loss_test = loss_test,
        labels_train = uncategorical_train_labels,
        labels_test = uncategorical_test_labels
    )

    return attack_input

def prepareData(dataset, model):
    if dataset == 'mnist':
        return __load_mnist(dataset, model)
    elif dataset == 'ciphar10':
        return __load_ciphar10(dataset, model)
    elif dataset == 'imdb':
        return __load_IMDB(dataset, model)
    else:
        return None

def attackModel(dataset, model):
    attack_input = prepareData(dataset, model)
    slicing_spec = SlicingSpec(
        entire_dataset = True,
        by_class = True,
        by_percentiles = False,
        by_classification_correctness = True
    )
    attack_types = [
        AttackType.LOGISTIC_REGRESSION,
        AttackType.RANDOM_FOREST,
        AttackType.THRESHOLD_ATTACK
    ]
    attacks_result = mia.run_attacks(
        attack_input=attack_input,
        slicing_spec=slicing_spec,
        attack_types=attack_types
    )
    return attacks_result

path = 'models/'
ignoreFolder = ["ciphar10"]
ignoreFolder = []
files = os.listdir(path)
done = False
for f in files:
    if f not in ignoreFolder and os.path.isdir(path + f):
        dataset = f
        modelFiles = os.listdir(path + f)
        for modelName in modelFiles:
            if modelName.endswith(".hdf5") and not os.path.exists(path + f + "/" + modelName + ".attack"):
                flags = splitedNameToFlags(modelName.split("-"))
                model = load_model(path + f + "/" + modelName, compile=False)
                optimizer, loss = lo.load(flags, dataset)
                model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
                results = attackModel(dataset, model)
                #print(results.summary(by_slices=True))
                dataframe = results.calculate_pd_dataframe()
                dataframe.to_csv(path + f + "/" + modelName + ".attack")
                print("Done:", modelName)
                #done = True
            if done: break
    if done: break
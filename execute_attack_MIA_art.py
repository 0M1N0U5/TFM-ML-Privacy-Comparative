from operator import index
import os
from utils import calculate_delta
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import loadOptimizer as lo
import loadDatasets as ld

from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import TensorFlowV2Classifier
from art.attacks.inference.membership_inference import MembershipInferenceBlackBox
from art.attacks.inference.membership_inference import LabelOnlyDecisionBoundary
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    def train_step(model, images, labels):
        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            loss = loss_object(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # Step 3: Create the ART classifier
    classifier = TensorFlowV2Classifier(
        model=model,
        loss_object=loss_object,
        train_step=train_step,
        nb_classes=10,
        input_shape=(28, 28, 1),
        clip_values=(0, 1),
    )
    return classifier

def __load_IMDB(dataset, model):
    loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    classifier = TensorFlowV2Classifier(
        model=model,
        loss_object=loss_object,
        nb_classes=2,
        input_shape=(10000,),
        clip_values=(0, 1),
    )
    return classifier

def __load_ciphar10(dataset, model):
    loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    classifier = TensorFlowV2Classifier(
        model=model,
        loss_object=loss_object,
        nb_classes=10,
        input_shape=(32, 32, 3),
        clip_values=(0, 1),
    )
    return classifier

def prepareData(dataset, model):
    if dataset == 'mnist':
        return __load_mnist(dataset, model)
    elif dataset == 'ciphar10':
        return __load_ciphar10(dataset, model)
    elif dataset == 'imdb':
        return __load_IMDB(dataset, model)
    else:
        return None

def attackModel(dataset, model, modelType):
    x_train, y_train, x_test, y_test = ld.load(dataset)
    classifier = prepareData(dataset, model)

    # black box attack - 50% dataset
    attack_x_train, attack_x_train_test, attack_y_train, attack_y_train_test = train_test_split(x_train, y_train, test_size=0.5, stratify=y_train)
    attack_x_test, attack_x_test_test, attack_y_test, attack_y_test_test = train_test_split(x_test, y_test, test_size=0.5, stratify=y_test)

    labelOnly = False
    bb_attack = 0
    if labelOnly:
        bb_attack = LabelOnlyDecisionBoundary(classifier)
        bb_attack.calibrate_distance_threshold(attack_x_train, attack_y_train,  attack_x_train_test, attack_y_train_test)
    else:
        bb_attack = MembershipInferenceBlackBox(classifier, attack_model_type=modelType)
        bb_attack.fit(attack_x_train, attack_y_train, attack_x_test, attack_y_test)
    
    # get inferred values
    inferred_train_bb = bb_attack.infer(attack_x_train_test, attack_y_train_test)
    inferred_test_bb = bb_attack.infer(attack_x_test_test, attack_y_test_test)
    # check accuracy
    train_acc = np.sum(inferred_train_bb) / len(inferred_train_bb)
    test_acc = 1 - (np.sum(inferred_test_bb) / len(inferred_test_bb))
    #acc = (train_acc * len(inferred_train_bb) + test_acc * len(inferred_test_bb)) / (len(inferred_train_bb) + len(inferred_test_bb))
    acc = (train_acc + test_acc) / 2 
    return train_acc, test_acc, acc

path = 'models/'
ignoreFolder = ["mnist", "ciphar10"]
ignoreFolder = []
modelTypes = ['rf', 'gb']
files = os.listdir(path)
done = False
for f in files:
    if f not in ignoreFolder and os.path.isdir(path + f):
        dataset = f
        modelFiles = os.listdir(path + f)
        for modelName in modelFiles:
            if modelName.endswith(".hdf5") and not os.path.exists(path + f + "/" + modelName + ".ART.attack"):
                flags = splitedNameToFlags(modelName.split("-"))
                model = load_model(path + f + "/" + modelName, compile=False)
                optimizer, loss = lo.load(flags, dataset)
                model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
                column_names = ["modelType", "train_acc", "test_acc", "mean_acc"]
                df = pd.DataFrame(columns = column_names)
                for modelType in modelTypes:
                    train_acc, test_acc, acc = attackModel(dataset, model, modelType)
                    new_row = {'modelType':modelType, 'train_acc':train_acc, 'test_acc':test_acc, 'mean_acc':acc}
                    df = df.append(new_row, ignore_index=True)
                if not df.empty:
                    df.to_csv(path + f + "/" + modelName + ".ART.attack")   
                print("Done:", modelName)
                #done = True
            if done: break
    if done: break
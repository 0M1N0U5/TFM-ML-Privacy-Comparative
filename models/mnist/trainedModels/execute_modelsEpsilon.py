import os
import utils
import loadFlags as lf

path = 'models/'

files = os.listdir(path)
def listSelector(list, indexList, separator=' '):
    result = []
    for i in range(len(list)):
        if i in indexList:
            result.append(list[i])
    return separator.join(result)

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

columns = "dataset,noise,optimizer,trainSize,batch_size,microbatches,epochs,accuracy,eps"
print(columns)
for f in files:
    if f.endswith(".hdf5"):
        fSplited = f.split("-")
        if f.find("-DP") > 0:
            #checkpoint-mnist-DP-1.2-SGD-60000-250-250-60-0.860200-CALCULATE_EPS.hdf5
            trainSize = int(fSplited[5])
            targetDelta = utils.calculate_delta(trainSize)
            flags = TMPFLAGS()
            flags.learning_rate = 0.01
            flags.l2_norm_clip = 1.0
            flags.batch_size =int(fSplited[6])
            flags.epochs = int(fSplited[8])
            flags.batch_size = int(fSplited[6])
            flags.noise_multiplier = float(fSplited[3])
            eps = utils.compute_epsilon(flags, flags.epochs * trainSize // flags.batch_size, trainSize, targetDelta)
            eps = "{:.6f}".format(eps)
            print(listSelector(fSplited, [1, 3, 4, 5, 6, 7, 8, 9], ',') + "," + eps)
        else:
            #checkpoint-mnist-SGD-60000-250-250-59-0.956700-CALCULATE_EPS.hdf5
            list1 = listSelector(fSplited, [1], ",") + ",0.0"
            list2 = listSelector(fSplited, [2, 3, 4, 5, 6, 7], ',')
            print(list1 + "," + list2 + ",inf")
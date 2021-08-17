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

columns = "dataset,l2_norm_clip,learning_rate,noise,optimizer,trainSize,batch_size,microbatches,epochs,accuracy,eps"
print(columns)
for f in files:
    if f.endswith(".hdf5"):
        fSplited = f.split("-")
        eps = "inf"
        if f.find("-DP") > 0:
            #checkpoint-mnist-DP-norm_clip-learning_rate-noise-SGD-60000-250-250-60-0.860200-CALCULATE_EPS.hdf5
            trainSize = int(fSplited[7])
            targetDelta = utils.calculate_delta(trainSize)
            flags = TMPFLAGS()
            flags.learning_rate = float(fSplited[4])
            flags.l2_norm_clip = float(fSplited[3])
            flags.batch_size =int(fSplited[8])
            flags.epochs = int(fSplited[10])
            flags.noise_multiplier = float(fSplited[5])
            eps = utils.compute_epsilon(flags, flags.epochs * trainSize // flags.batch_size, trainSize, targetDelta)
            eps = "{:.6f}".format(eps)
        print(listSelector(fSplited, [1, 3, 4, 5, 6, 7, 8, 9, 10, 11], ',') + "," + eps)
import numpy as np
import theano
import time

from fTheanoNNclassCORE import *


learnStep = 0.0001
batchSize = 32
CV_size = 1

OPTIONS = OptionsStore(learnStep=learnStep,
                       rmsProp=0.9,
                       minibatch_size=batchSize,
                       CV_size=CV_size)

L1 = LayerCNN(size_in=28224,
              size_out=25600,
              activation=FunctionModel.ReLU,
              weightDecay=False,
              sparsity=False,
              beta=2,
              dropout=False,
              kernel_shape=(16, 1, 8, 8),
              stride=4,
              pooling=False,
              pooling_shape=2,
              optimized=True)

L2 = LayerCNN(size_in=25600,
              size_out=10368,
              activation=FunctionModel.ReLU,
              weightDecay=False,
              sparsity=False,
              beta=2,
              dropout=False,
              kernel_shape=(32, 16, 4, 4),
              stride=2,
              pooling=False,
              pooling_shape=2,
              optimized=True)

L3 = LayerNN(size_in=10368,
             size_out=4,
             dropout=0.5,
             activation=FunctionModel.Sigmoid)

NN = TheanoNNclass(OPTIONS, (L1, L2, L3))
NN.trainCompile()
NN.predictCompile()

feed = np.zeros((28224, batchSize))

res = NN.predictCalc(feed).out

print res.shape
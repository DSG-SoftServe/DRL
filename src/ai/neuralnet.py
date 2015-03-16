#-----------------------------------------------#
# Neural Network realisation class based on TNNF.
#-----------------------------------------------#
import numpy as np
from TNNF import fTheanoNNclassCORE, fGraphBuilderCORE
#-----------------------------------------------#

class NeuralNet:
    def __init__(self, bSize):
        self.waccum = []   # TODO Collection?
        learnStep = 0.00001
        batchSize = bSize
        CV_size = 1
        
        OPTIONS = fTheanoNNclassCORE.OptionsStore(learnStep=learnStep,
                                                  rmsProp=0.9,
                                                  minibatch_size=batchSize,
                                                  CV_size=CV_size)

        L1 = fTheanoNNclassCORE.LayerCNN(size_in=28224,
                                         size_out=6400,
                                         activation=fTheanoNNclassCORE.FunctionModel.LReLU,   # ReLu for gs
                                         weightDecay=1e-6,
                                         sparsity=False,
                                         beta=2,
                                         dropout=False,
                                         kernel_shape=(16, 4, 8, 8),
                                         stride=4,
                                         pooling=False,
                                         pooling_shape=2,
                                         optimized=True)

        L2 = fTheanoNNclassCORE.LayerCNN(size_in=6400,
                                         size_out=2592,
                                         activation=fTheanoNNclassCORE.FunctionModel.LReLU,   # ReLu for gs
                                         weightDecay=1e-6,
                                         sparsity=False,
                                         beta=2,
                                         dropout=False,
                                         kernel_shape=(32, 16, 4, 4),
                                         stride=2,
                                         pooling=False,
                                         pooling_shape=2,
                                         optimized=True)

        L3 = fTheanoNNclassCORE.LayerNN(size_in=2592,
                                        size_out=256,
                                        #weightDecay=1e-6,
                                        #dropout=0.85,
                                        activation=fTheanoNNclassCORE.FunctionModel.ReLU)

        L4 = fTheanoNNclassCORE.LayerNN(size_in=256,
                                        size_out=4,
                                        #dropout=0.85,
                                        activation=fTheanoNNclassCORE.FunctionModel.Linear)

        NN = fTheanoNNclassCORE.TheanoNNclass(OPTIONS, (L1, L2, L3, L4))
        NN.trainCompile()
        NN.predictCompile()
        
        self.network = NN
        self.gamma = 0.97   # 0.99 or 0.9
        self.train_m = NN.trainCalc
        self.predict_r = NN.predictCalc

    def train(self, minibatch):
        batch_prestate = []
        batch_poststate = []
        batch_actions = []
        batch_reward = []

        for b in minibatch:
            batch_prestate.append(b['prestate'].reshape(-1, ))
            batch_poststate.append(b['poststate'].reshape(-1, ))
            batch_actions.append(b['action'])
            batch_reward.append(b['reward'])

        batch_prestate = np.array(batch_prestate).T
        batch_poststate = np.array(batch_poststate).T
        batch_actions = np.array(batch_actions)
        batch_reward = np.array(batch_reward)

        # Debug. Saving batch.
        #np.save('./models/AS.batch.4.84.84', batch_prestate)
        #print batch_prestate
        estQ = self.predict_r(batch_prestate).out
        # HERE WE SHOULD USE THIS FORMULA ONLY FOR NOT END GAME (5 page). So there is hack for this.
        mask1 = np.array(batch_reward < 50)
        mask2 = np.array(batch_reward > 50)
        estQ[batch_actions, range(batch_actions.shape[0])] = (batch_reward - 100 * mask2) + self.gamma * np.max(self.predict_r(batch_poststate).out, axis=0) * mask1
        tst = self.train_m(batch_prestate, estQ, iteration=1, debug=False, errorCollect=True).train_out   # TODO Draw errorCollect

    def predict_action(self, state):
        predicted_values_for_actions = self.predict_r(state.reshape(-1, 1)).out
        #print "predicted best action", predicted_values_for_actions
        return  np.argmax(predicted_values_for_actions)

    def saveModel(self, name):
        self.network.modelSaver(name)

    def loadModel(self, name):
        self.network.modelLoader(name)
        
    def printedW(self):
        w0 = self.network.paramGetter()[0]['w']
        w1 = self.network.paramGetter()[1]['w']
        print w0.shape, w1.shape 
        #print w[0, 0, :3, :3]
        w = np.sum(w0) + np.sum(w1)
        print "Weight sum ", w
        self.waccum.append(w)
        fTheanoNNclassCORE.NNsupport.errorG(self.waccum, "./graphs/wsum.png")
        if len(self.waccum) > 25:
            self.waccum = []
                
#-----------------------------------------------#

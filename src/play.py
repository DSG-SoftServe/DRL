#-----------------------------------------------#
# Class for demo. Loads model and starts game 
# without any model training.
#-----------------------------------------------#
from ale.ale import ALE
from memory.memory import Memory
from ai.neuralnet import NeuralNet
import time
import random
#-----------------------------------------------#

class Play:
    def __init__(self):
        self.size = 50000
        self.minibatch_size = 128
        self.nnet = NeuralNet(self.minibatch_size)
        self.memory = Memory(size=self.size)
        self.ale = ALE(self.memory, skip_frames=6)
        # Loading model
        self.nnet.loadModel('./models/!AS.model.weights.NN.NMem.933134')
        # For Pillow lower than 2.7.0 version
        #self.neuraln.loadModel('./models/~~p-AS.model.weights.NN.NMem.906508')
                       
    def play(self):
        while True:
            self.ale.newG()
            while not self.ale.gameover:
                # Delay for slow human perception
                time.sleep(0.05)
                # For game start we should press fire
                if random.uniform(0, 1) < 0.05 or len(self.memory.screens) < self.memory.game_step_phase:
                    action = 1
                    print "\naction:", action
                else:
                    # Random for tests
                    #action = random.choice(range(4))
                    # Neuralnet
                    action = self.nnet.predict_action(self.memory.state())
                    print '\naction NEURALNET:', action
                # Act and store action in memory
                self.ale.move(action)
            self.ale.endG()

if __name__ == '__main__':
    P = Play()
    P.play()

#-----------------------------------------------#

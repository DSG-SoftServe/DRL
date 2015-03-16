#-----------------------------------------------#
# Main class for agent training.
#-----------------------------------------------#
from ale.ale import ALE
from memory.memory import Memory
from ai.neuralnet import NeuralNet
import time
import random
import cPickle
#-----------------------------------------------#

class Main:
    def __init__(self):
        self.size = 100000   # Memory size ~ 5Gb
        self.epsilon_step_dec = 700000   # Counter for epsilon dec. step
        self.M = 0   # Episodes
        self.T = 0   # Sequence max length
        self.size_of_minibatch = 128   # Minibatch size
        self.steps = 0   # Steps counter, inc. only
        self.count = 0   # Saving counter
        self.actions_number = 4   # Left, right, up, nothing
        self.neuraln = NeuralNet(self.size_of_minibatch)
        self.memory = Memory(size=self.size)
        self.ale = ALE(self.memory, skip_frames=6)   # 4 in paper (5 for space invaders)
        # Loading model and saved memory
        self.neuraln.loadModel('./models/!AS.model.weights.NN.NMem.933134')
        #self.steps = 933134
        #f = file('./models/!AS.memory.D.933134', 'rb')
        #self.memory.D = cPickle.load(f)
        #f.close()
        
    def __epsilon__(self):
        return max((0.9 - 0.9 / self.epsilon_step_dec * self.steps), 0.1)
                        
    def play(self):
        # Maximum number of episodes
        while self.M < 50000000:
            self.ale.newG()
            # Game over or maximum sequence length
            while self.T < 50000 and not self.ale.gameover:
                # Measurement of time for iteration
                timeStart = time.time()
                # Decreasing number of random actions in time
                if random.uniform(0, 1) < self.__epsilon__() or len(self.memory.D) < self.size_of_minibatch or len(self.memory.screens) < self.memory.game_step_phase:
                    action = random.choice(range(self.actions_number))
                    print "\naction RAND:", action
                else:
                    action = self.neuraln.predict_action(self.memory.state())
                    print '\naction NEURALNET:', action
                # Act and store action in memory
                self.ale.move(action)
                # Training
                if len(self.memory.D) > self.size_of_minibatch:
                    self.neuraln.train(self.memory.minibatch(self.size_of_minibatch))
                self.steps += 1
                self.count += 1
                self.T += 1
                timeStop = time.time()
                print "Step:", self.steps
                # Time used for iteration in minutes
                print round((timeStop - timeStart) / 60, 5)
            self.ale.endG()
            # Saving model
            if self.count >= 20000:
                self.count = 0
                self.neuraln.saveModel('./models/AS.model.weights.NN.NMem.' + str(self.steps))
                self.memory.mem_save(self.steps)            
            # Sum of weights of hidden layers for debug
            self.neuraln.printedW()
            self.T = 0
            self.M += 1
            print "Episode:", self.M
            
if __name__ == '__main__':
    T = Main()
    T.play()

#-----------------------------------------------#

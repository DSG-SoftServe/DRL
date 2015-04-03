#-----------------------------------------------#
# Main class for agent training.
#-----------------------------------------------#
from ale.ale import ale
from memory.memory import memory
from ai.neuralnet import neuralnet
import time
import random
import cPickle
#-----------------------------------------------#

class main:
    def __init__(self):
        self.size = 100000   # memory size ~ 5Gb
        self.epsilon_step_dec = 700000   # Counter for epsilon dec. step
        self.M = 0   # Episodes
        self.T = 0   # Sequence max length
        self.minibatch_size = 128   # Minibatch size
        self.steps = 0   # Steps counter, inc. only
        self.count = 0   # Saving counter
        self.actions_number = 4   # Left, right, up, nothing
        self.n_net = neuralnet(self.minibatch_size)
        self.memory = memory(size=self.size)
        self.ale = ale(self.memory, frames_to_skip=6)   # 4 in paper (5 for space invaders)
        # Loading model and saved memory
        #self.n_net.loadModel('./models/!AS.model.weights.NN.NMem.933134')
        #self.steps = 933134
        #f = file('./models/!AS.memory.D.933134', 'rb')
        #self.memory.D = cPickle.load(f)
        #f.close()
        
    def __compute_epsilon__(self):
        return max((0.9 - 0.9 / self.epsilon_step_dec * self.steps), 0.1)
                        
    def run_game(self):
        # Maximum number of episodes
        while self.M < 50000000:
            self.ale.start_game()
            # Game over or maximum sequence length
            while self.T < 50000 and not self.ale.game_over:
                # Measurement of time for iteration
                timeStart = time.time()
                # Decreasing number of random actions in time
                if random.uniform(0, 1) < self.__compute_epsilon__() or len(self.memory.D) < self.minibatch_size or len(self.memory.screenshots) < self.memory.game_step_phase:
                    action = random.choice(range(self.actions_number))
                    print "\naction RAND:", action
                else:
                    action = self.n_net.predict_action(self.memory.get_actual_state())
                    print '\naction NEURALNET:', action
                # Act and store action in memory
                self.ale.move(action)
                # Training
                if len(self.memory.D) > self.minibatch_size:
                    self.n_net.train(self.memory.get_minibatch(self.minibatch_size))
                self.steps += 1
                self.count += 1
                self.T += 1
                timeStop = time.time()
                print "Step:", self.steps
                # Time used for iteration in minutes
                print round((timeStop - timeStart) / 60, 5)
            self.ale.finish_game()
            # Saving model
            if self.count >= 20000:
                self.count = 0
                self.n_net.saveModel('./models/AS.model.weights.NN.NMem.' + str(self.steps))
                self.memory.mem_save(self.steps)            
            # Sum of weights of hidden layers for debug
            self.n_net.printedW()
            self.T = 0
            self.M += 1
            print "Episode:", self.M
            
if __name__ == '__main__':
    T = main()
    T.run_game()

#-----------------------------------------------#

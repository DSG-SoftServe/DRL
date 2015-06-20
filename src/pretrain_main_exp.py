#-----------------------------------------------#
# Class for agent pre training on "pilot" data.
#-----------------------------------------------#
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
        self.minibatch_size = 128   # Minibatch size
        self.n_net = neuralnet(self.minibatch_size)
        self.memory = memory(size=self.size)
        self.steps = 0
        # Loading memory
        f = file('AS.memory.D.2079', 'rb')   # Change to yours memory dump name
        self.memory.D = cPickle.load(f)
        f.close()

    def __mapper__(self, batch):
        # -------
        # 0 1 2 3 Our
        # 0 1 3 4
        # -------
        # 0 1 2 3 4 5 Error Map
        # 0 1 3 2 5 4
        # -------
        for b in batch:
            print b['action'],
            if b['action'] == 3 or b['action'] == 4:   # b['action'] == 3 delete
                b['action'] = 0
            if b['action'] == 5:
                b['action'] = 3

    def run_game(self):
        # Maximum number of episodes
        while self.steps < 700000:
            # Measurement of time for iteration
            timeStart = time.time()
            btch = self.memory.get_minibatch(self.minibatch_size)
            #self.__mapper__(btch)   # Only if there are mistakes in ale.py self.actions_list
            self.n_net.train(btch)
            self.steps += 1
            timeStop = time.time()
            print "Step:", self.steps
            # Time used for iteration in minutes
            print round((timeStop - timeStart) / 60, 5)
            # Saving model
            if self.steps % 30000 == 0:
                self.n_net.saveModel('./models/AS.model.weights.NN.NMem.' + str(self.steps))
            # Sum of weights of hidden layers for debug
            self.n_net.printedW()

if __name__ == '__main__':
    T = main()
    T.run_game()

#-----------------------------------------------#

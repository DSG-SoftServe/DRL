#-----------------------------------------------#
# Class for demo. Loads model and starts game 
# without any model training.
#-----------------------------------------------#
from ale.ale import ale
from memory.memory import memory
from ai.neuralnet import neuralnet
import time
import random
#-----------------------------------------------#

class play:
    def __init__(self):
        self.size = 50000
        self.minibatch_size = 128
        self.n_net = neuralnet(self.minibatch_size)
        self.memory = memory(size=self.size)
        self.ale = ale(self.memory, frames_to_skip=6)   # 5 for space invaders
        # Loading model
        self.n_net.loadModel('./models/Improved_Breakout_AS.model.weights.NN.NMem.3707295')
        # For Pillow lower than 2.7.0 version
        #self.n_net.loadModel('./models/~~p-AS.model.weights.NN.NMem.906508')
                       
    def run_game(self):
        while True:
            self.ale.start_game()
            while not self.ale.game_over:
                # Delay for slow human perception
                #time.sleep(0.1)
                # For game start we should press fire
                if random.uniform(0, 1) < 0.05 or len(self.memory.screenshots) < self.memory.game_step_phase:
                    action = 1
                    print "\naction:", action
                else:
                    # Random for tests
                    #action = random.choice(range(4))
                    # Neuralnet
                    action = self.n_net.predict_action(self.memory.get_actual_state())
                    print '\naction NEURALNET:', action
                # Act and store action in memory
                self.ale.move(action)
            self.ale.finish_game()

if __name__ == '__main__':
    P = play()
    P.run_game()

#-----------------------------------------------#

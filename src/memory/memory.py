#-----------------------------------------------#
# Transitions memory class.
#-----------------------------------------------#
import numpy as np
import random
import collections
import cPickle
#-----------------------------------------------#

class memory:
    def __init__(self, size=1000000):
        self.screenshots = []
        self.actions = []
        self.rewards = []
        self.game_step_phase = 4   # Wait for collect 4 images
        self.size = size   # Size of memory
        self.D = collections.deque(maxlen=self.size)   # Transitions memory TODO Numpy instead collection
        
    def __stat__(self, mute=False):
        if not mute:
            print "S:", len(self.screenshots), "A:", len(self.actions), "R:", len(self.rewards), "D:", len(self.D)
            
    def __get_state__(self, number):
        state = self.screenshots[number - 3 : number + 1]
        side = state[0].shape[0]
        new_state = np.zeros((4, side, side))   # Add param dtype=uint8
        new_state[0, :, :] = state[0]
        new_state[1, :, :] = state[1]
        new_state[2, :, :] = state[2]
        new_state[3, :, :] = state[3]
        new_state = np.uint8(new_state)   # Not use with 0-1
        # Debug
        #np.set_printoptions(threshold=np.nan)
        #print new_state
        # Saving batch
        #np.save('./models/AS.state', new_state)
        return new_state
 
    def insert_first(self, new_screenshot):
        self.screenshots.append(new_screenshot)
        self.__stat__()

    def insert(self, action, reward, new_screenshot):
        self.actions.append(action)
        self.rewards.append(reward)
        self.screenshots.append(new_screenshot)
        # Number of last element in screenshots list
        screen_num = len(self.screenshots) - 1
        if len(self.screenshots) > self.game_step_phase:
            self.D.append({'prestate': self.__get_state__(screen_num - 1),
                           'action': action,
                           'reward': reward,
                           'poststate': self.__get_state__(screen_num)})
        self.__stat__()

    def insert_last(self):
        # Hack for checking last state or not
        self.D[len(self.D) - 1]["reward"] = self.D[len(self.D) - 1]["reward"] + 100
        # Cleaning
        self.screenshots = []
        self.actions = []
        self.rewards = []
          
    def get_minibatch(self, size):
        index = np.random.randint(0, len(self.D), size)
        transitions = [self.D[x] for x in index]   # TODO Make this with matrix way
        return transitions
        
    def get_actual_state(self):
        return self.__get_state__(len(self.screenshots) - 1)
        
    def mem_save(self, index):
        f = file('./models/AS.memory.D.' + str(index), 'wb')
        cPickle.dump(self.D, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()
                
#-----------------------------------------------#

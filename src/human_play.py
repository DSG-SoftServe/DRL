#-----------------------------------------------#
# Class for human demo. Linux only for now.
# For windows should implement:
# getkey() <-> from msvcrt import getch
# Controls: w a s d
#-----------------------------------------------#
from ale.ale import ale
from memory.memory import memory
from ai.neuralnet import neuralnet
import time
import termios
import sys
import os
import threading
#-----------------------------------------------#

TERMIOS = termios

def getkey():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    new = termios.tcgetattr(fd)
    new[3] = new[3] & ~TERMIOS.ICANON & ~TERMIOS.ECHO
    new[6][TERMIOS.VMIN] = 1
    new[6][TERMIOS.VTIME] = 0
    termios.tcsetattr(fd, TERMIOS.TCSANOW, new)
    c = None
    try:
        c = os.read(fd, 1)
    finally:
        termios.tcsetattr(fd, TERMIOS.TCSAFLUSH, old)
    return c

char = None

def keypress():
    global char
    char = getkey()

# Start thread for keypress
t = threading.Thread(target=keypress)
t.daemon = True
t.start()

class play:
    def __init__(self):
        self.size = 50000
        self.minibatch_size = 128
        self.n_net = neuralnet(self.minibatch_size)
        self.memory = memory(size=self.size)
        self.ale = ale(self.memory, frames_to_skip=6)   # 5 for space invaders

    def run_game(self):
        global char, t
        while True:
            self.ale.start_game()
            while not self.ale.game_over:
                # Delay for slow human perception
                time.sleep(0.1)   # NN plays without delay. Can you?
                # For game start we should press fire
                #-----------------------
                # Debug
                #print "Key", char
                if not(t.isAlive()):
                    t = threading.Thread(target=keypress)
                    t.daemon = True
                    t.start()
                #-----------------------
                if str(char) == "d":
                    action = 2
                elif str(char) == "a":
                    action = 3
                elif str(char) == "w":
                    action = 1
                else:
                    action = 0
                print '\naction HUMAN:', action

                # We collect all moves, so using special methods
                # we can save them to use for training process later:
                # self.memory.mem_save(self.steps)

                # Act and store action in memory
                self.ale.move(action)
                # Clearing key
                char = None
            self.ale.finish_game()

if __name__ == '__main__':
    P = play()
    # Uncomment if you don't want prints in console
    #sys.stdout = open(os.devnull, "w")
    P.run_game()
    #sys.stdout = sys.__stdout__

#-----------------------------------------------#

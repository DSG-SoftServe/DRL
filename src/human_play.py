#-----------------------------------------------#
# Class for human demo.
#-----------------------------------------------#
from ale.ale import ale
from memory.memory import memory
from ai.neuralnet import neuralnet
import time
import Tkinter as tk
#-----------------------------------------------#

def keypress(event):
    if event.keysym == 'Escape':
        root.destroy()
    print event.char

class play:
    def __init__(self):
        self.size = 50000
        self.minibatch_size = 128
        self.n_net = neuralnet(self.minibatch_size)
        self.memory = memory(size=self.size)
        self.ale = ale(self.memory, frames_to_skip=6)

    def run_game(self):
        while True:
            self.ale.start_game()
            while not self.ale.game_over:
                # Delay for slow human perception
                time.sleep(0.05)
                # For game start we should press fire
                action = 1
                # Act and store action in memory
                self.ale.move(action)
            self.ale.finish_game()

if __name__ == '__main__':
    root = tk.Tk()
    print "Press a key (Escape key to exit):"
    root.bind_all('<Key>', keypress)
    # don't show the tk window
    root.withdraw()
    root.mainloop()
    P = play()
    P.run_game()


#-----------------------------------------------#

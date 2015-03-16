#-----------------------------------------------#
# Atari emulator wrapper.
#-----------------------------------------------#
import os
import numpy as np
from time import sleep
from data.preprocessor import Preprocessor
from TNNF import fTheanoNNclassCORE, fGraphBuilderCORE
#-----------------------------------------------#

class ALE:
    def __init__(self, memory, display_screen="true", skip_frames=4, game_ROM='../libraries/ale_0_4/roms/breakout.bin'):
        # List of possible actions for agent
        self.actions = [np.uint8(0), np.uint8(1), np.uint8(3), np.uint8(4)]
        # Read and write commands and response from emulator
        self.fin = ""
        self.fout = ""
        # Storage for transitions
        self.memory = memory
        # Emulator environment variables
        self.display_screen = display_screen
        self.skip_frames = skip_frames
        self.game_ROM = game_ROM
        # Variables for emulator responses
        self.gameover = False
        self.next_image = ""
        self.c_reward = 0
        # Variables for visualization of rewards
        self.rewards_accum = []
        self.rewards_count = 0
        self.rewards_moment = 0
        self.play_games = 0
        self.p_g_a = []
        # Read and write commands and response from emulator
        os.system("mkfifo ale_fifo_out")
        os.system("mkfifo ale_fifo_in")
        # Game loading and starting
        start_command='./../libraries/ale_0_4/ale -max_num_episodes 0 -game_controller fifo_named -disable_colour_averaging true -run_length_encoding false -frame_skip ' + str(self.skip_frames) + ' -display_screen ' + self.display_screen + " " + self.game_ROM + " &"
        print start_command
        os.system(start_command)
        # Read and write commands and response from emulator
        self.fin = open('ale_fifo_out')
        self.fout = open('ale_fifo_in', 'w')
        # Initialization read
        init_read = self.fin.readline()   # Screen size [:-1] because last char is just empty space
        # Initialization parameters for emulation
        self.fout.write("1,0,0,1\n")
        self.fout.flush()
        # Data preprocessing
        self.preprocessor = Preprocessor()

    def __get_param__(self):
        self.next_image, info = self.fin.readline()[:-2].split(":")
        self.gameover = bool(int(info.split(",")[0]))
        self.c_reward = int(info.split(",")[1])
    
    def __first_fire_(self):
        self.fout.write("1,0\n")   # 1,40 for other ROMs
        self.fout.flush()
        self.fin.readline()
        
    def newG(self):
        self.__get_param__()
        print type(self.next_image), len(self.next_image)
        print type(self.preprocessor.proc(self.next_image)), len(self.preprocessor.proc(self.next_image))
        self.memory.addStart(self.preprocessor.proc(self.next_image))
        self.__first_fire_()
        self.play_games += 1
        print "Start"
        
    def endG(self):
        self.fout.write("45,45\n")
        self.fout.flush()
        self.memory.addEnd()
        print "End\n"
        
    def move(self, action_index):
        action = self.actions[action_index]
        self.fout.write(str(action)+",0\n")   # 1,40 for other ROMs
        self.fout.flush()
        self.__get_param__()
        if self.gameover:
            self.c_reward = -1
        # TODO Custom skip frames for play class
        self.memory.add(action_index, self.c_reward, self.preprocessor.proc(self.next_image))
        # Building of rewards graphic
        self.rewards_moment = self.rewards_moment + self.c_reward
        self.rewards_count += 1
        if self.rewards_count >= 1000:
            self.rewards_count = 0
            self.rewards_accum.append(self.rewards_moment)
            self.p_g_a.append(self.rewards_moment / (self.play_games + 1.0 * (self.play_games == 0)))            
            self.rewards_moment = 0
            self.play_games = 0
            fTheanoNNclassCORE.NNsupport.errorG(self.rewards_accum, "./graphs/rewards.png")
            fTheanoNNclassCORE.NNsupport.errorG(self.p_g_a, "./graphs/rgstat.png")
            if len(self.rewards_accum) > 250:
                self.rewards_accum = []
                self.p_g_a = []
        print "___________________ Reward ", self.c_reward

#-----------------------------------------------#

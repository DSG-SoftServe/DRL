#-----------------------------------------------#
# Atari emulator wrapper.
#-----------------------------------------------#
import os
import numpy as np
from time import sleep
from data.preprocessing import preprocessing
from TNNF import fTheanoNNclassCORE, fGraphBuilderCORE
#-----------------------------------------------#

class ale:
    def __init__(self, memory, display_screen="true", frames_to_skip=4, ale_game_ROM='../emulators/ale_0_4/roms/breakout.bin'):
        # List of possible actions for agent
        self.actions_list = [np.uint8(0), np.uint8(1), np.uint8(3), np.uint8(4)]   # np.uint8(2), np.uint8(5) instead 3 and 4 for Boxing ROM
        # Read and write commands and response from emulator
        self.f_in = ""
        self.f_out = ""
        # Storage for transitions
        self.memory = memory
        # Emulator environment variables
        self.display_screen = display_screen
        self.frames_to_skip = frames_to_skip
        self.ale_game_ROM = ale_game_ROM
        # Variables for emulator responses
        self.game_over = False
        self.screen_image = ""
        self.reward_from_emulator = 0
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
        start_command = './../emulators/ale_0_4/ale -max_num_episodes 0 -game_controller fifo_named -disable_colour_averaging true -run_length_encoding false -frame_skip ' + str(self.frames_to_skip) + ' -display_screen ' + self.display_screen + " " + self.ale_game_ROM + " &"
        print start_command
        os.system(start_command)
        # Read and write commands and response from emulator
        self.f_in = open('ale_fifo_out')
        self.f_out = open('ale_fifo_in', 'w')
        # Initialization read
        init_read = self.f_in.readline()   # Screen size [:-1] because last char is just empty space
        # Initialization parameters for emulation
        self.f_out.write("1,0,0,1\n")
        self.f_out.flush()
        # Data preprocessing
        self.preprocessing = preprocessing()

    def __get_param__(self):
        self.screen_image, step_information = self.f_in.readline()[:-2].split(":")
        self.game_over = bool(int(step_information.split(",")[0]))
        self.reward_from_emulator = int(step_information.split(",")[1])
    
    def __first_fire_(self):
        self.f_out.write("1,18\n")   # 1,40 if not work ; (page 9 of ALE manual)
        self.f_out.flush()
        self.f_in.readline()
        
    def start_game(self):
        self.__get_param__()
        print type(self.screen_image), len(self.screen_image)
        print type(self.preprocessing.preprocess(self.screen_image)), len(self.preprocessing.preprocess(self.screen_image))
        self.memory.insert_first(self.preprocessing.preprocess(self.screen_image))
        self.__first_fire_()
        self.play_games += 1
        print "Start"
        
    def finish_game(self):
        self.f_out.write("45,45\n")
        self.f_out.flush()
        self.memory.insert_last()
        print "End\n"
        
    def move(self, index_of_action):
        action = self.actions_list[index_of_action]
        self.f_out.write(str(action)+",18\n")   # ,40 if not work ; (page 9 of ALE manual)
        self.f_out.flush()
        self.__get_param__()
        if self.game_over:
            self.reward_from_emulator = -1
        # Hack for rewards bigger than 1 or -1
        self.reward_from_emulator = np.clip(self.reward_from_emulator, -1, 1)
        # TODO Custom skip frames for play class
        self.memory.insert(index_of_action, self.reward_from_emulator, self.preprocessing.preprocess(self.screen_image))
        # Building of rewards graphic
        self.rewards_moment = self.rewards_moment + self.reward_from_emulator
        self.rewards_count += 1
        if self.rewards_count >= 1000:
            self.rewards_count = 0
            self.rewards_accum.append(self.rewards_moment)
            self.p_g_a.append(self.rewards_moment / (self.play_games + 1.0 * (self.play_games == 0)))            
            self.rewards_moment = 0
            self.play_games = 0
            # UNCOMMENT IF YOU NEED GRAPHS
            #fTheanoNNclassCORE.NNsupport.errorG(self.rewards_accum, "./graphs/rewards.png")
            #fTheanoNNclassCORE.NNsupport.errorG(self.p_g_a, "./graphs/rgstat.png")
            if len(self.rewards_accum) > 250:
                self.rewards_accum = []
                self.p_g_a = []
        print "___________________ Reward ", self.reward_from_emulator

#-----------------------------------------------#

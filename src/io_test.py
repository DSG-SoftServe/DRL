#-----------------------------------------------#
# Test file for keyboard I / O
#-----------------------------------------------#

"""
import time

i = 0
while i < 100:
    print i
    time.sleep(0.05)
    i += 1
"""

"""
from Tkinter import *

root = Tk()

def key(event):
    print "pressed", repr(event.char)

def callback(event):
    frame.focus_set()
    print "clicked at", event.x, event.y

frame = Frame(root, width=100, height=100)
frame.bind("<Key>", key)
frame.bind("<Button-1>", callback)
frame.pack()

root.mainloop()
"""

"""
import thread
import time
import sys
import tty
import termios

def getch():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

char = None

def keypress():
    global char
    char = getch()

thread.start_new_thread(keypress, ())

while True:
    print "Key pressed is " + str(char)
    char = None
    time.sleep(5)
"""

import termios
import sys
import os
#import thread
import threading
import time

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
        #print "Print", c
    finally:
        termios.tcsetattr(fd, TERMIOS.TCSAFLUSH, old)
    return c

char = None

def keypress():
    global char
    char = getkey()

#th = thread.start_new_thread(keypress, ())
t = threading.Thread(target=keypress)
t.daemon = True
t.start()

if __name__ == '__main__':
    while 1:
        #print "Test",
        #print t.isAlive()
        print "Print", char, "\n"
        #char = None
        time.sleep(0.005)
        if not(t.isAlive()):
            #print "Please start new"
            t = threading.Thread(target=keypress)
            t.daemon = True
            t.start()
        #c = getkey()
        #print 'got', c
        #s = s + c
        #print s

#-----------------------------------------------#
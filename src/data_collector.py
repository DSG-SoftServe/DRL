#-----------------------------------------------#
# Collect dumps from folder to one file.
#-----------------------------------------------#
import os,sys
import glob
import cPickle
import collections
#-----------------------------------------------#

memlist = glob.glob("./memory_dump/*.memory.*")
print len(memlist), memlist

D = collections.deque(maxlen=100000)   # 100k as a default in train script

test = []

#-----------------------------------------------#

for memfile in memlist:
    print "\n", memfile

    f = file(memfile, 'rb')
    add = cPickle.load(f)
    print type(add), len(add)

    test.append(add[1])   # [1] just for example, you can use any number
    f.close()

    D.extend(add)
    print "D: ", type(D), len(D)

#-----------------------------------------------#

print "\nLoaded..."

# INSERT HERE MAPPER IF YOU NEED __mapper__ from pretrain_main_exp.py

f = file('./AS.memory.D.' + str(len(D)), 'wb')
cPickle.dump(D, f, protocol=cPickle.HIGHEST_PROTOCOL)
f.close()

print "Saved..."

#-----------------------------------------------#
# Testing
#for ent in test:
#  print "\n", "*****************", ent  
#print D[1]
#-----------------------------------------------#
#!/bin/bash
time THEANO_FLAGS=mode=FAST_RUN,floatX=float32,device=cpu python $*


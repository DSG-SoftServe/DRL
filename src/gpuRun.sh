#!/bin/bash
time THEANO_FLAGS=cuda.root=/usr/local/cuda-6.5,mode=FAST_RUN,floatX=float32,device=gpu python $*


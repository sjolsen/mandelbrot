#!/bin/bash
g++ -std=c++0x -O3 -I./ *.cc -c -I /usr/local/cuda/include/
nvcc --compiler-options=-std=c++0x -O3 -arch=sm_20 -I./  cuda_wrapper.cu -c

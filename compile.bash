#!/bin/bash
g++ -std=c++0x -O3 -I./ *.cc -c -I /usr/local/cuda/include/
nvcc -O3 -arch=sm_20 -I./  cuda_wrapper.cu -c
g++ *.o -o mandelbrot -L /usr/local/cuda/lib64/ -lcuda -lcudart

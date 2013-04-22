CXX = g++
CXXFLAGS = -std=c++0x -I./ -I/usr/local/cuda/include/

all: mandelbrot

clean: rm *.o mandelbrot

mandelbrot: main.o bitmap.o args.o cuda_wrapper.o
	$(CXX) -L/usr/local/cuda/lib64/ main.o bitmap.o args.o cuda_wrapper.o -o mandelbrot -lcuda -lcudart

main.o: main.cc args.hh bitmap.hh cuda_wrapper.hh
	$(CXX) $(CXXFLAGS) main.cc -c

bitmap.o: bitmap.cc bitmap.hh
	$(CXX) $(CXXFLAGS) bitmap.cc -c

args.o: args.cc args.hh
	$(CXX) $(CXXFLAGS) args.cc -c

cuda_wrapper.o: cuda_wrapper.cu cuda_wrapper.hh
	nvcc -I./ cuda_wrapper.cu -c
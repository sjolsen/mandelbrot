CXX = g++
CXXFLAGS = -std=c++0x -I./ -I/home/gluster/so1132/include -I/usr/local/cuda/include/

all: mandelbrot

clean:
	rm *.o mandelbrot

mandelbrot: main.o args.o cuda_wrapper.o
	$(CXX) -L/usr/local/cuda/lib64/ main.o  args.o cuda_wrapper.o -o mandelbrot -lcuda -lcudart `libpng-config --ldflags`

main.o: main.cc args.hh cuda_wrapper.hh
	$(CXX) $(CXXFLAGS) main.cc -c

args.o: args.cc args.hh
	$(CXX) $(CXXFLAGS) args.cc -c

cuda_wrapper.o: cuda_wrapper.cu cuda_wrapper.hh
	nvcc -I./ -I/home/gluster/so1132/include -I/usr/local/cuda/include/ cuda_wrapper.cu -c
#include <args.hh>
#include <cuda_wrapper.hh>

#include <iostream>
#include <algorithm>
#include <chrono>

#include <png++/png.hpp>
#include <png++/rgb_pixel.hpp>
#include <cuda_runtime.h>
#include <mpi.h>

#include <unistd.h>

using namespace std;
using namespace std::chrono;

#define nullptr NULL



namespace
{
	png::rgb_pixel pixel_convert (pixel p)
	{
		return png::rgb_pixel (p.red, p.green, p.blue);
	}

	template <typename Iter>
	int accumulate (Iter first, Iter last, int init)
	{
		while (first != last)
			init += *first++;
		return init;
	}

	void gdb_hook (int rank)
	{
		char hostname [128];
		size_t len = 126;
		gethostname (hostname, len);
		auto pid = getpid ();

		cout << to_string ((long long int) rank) + ": " + to_string ((long long int) pid) + string (" ") + string (hostname) + "\n";
		volatile int i = 0;
		while (!i)
			sleep (5);
	}
}



int main (int argc,
          char** argv)
{
	// Program initialization

	MPI_Init (&argc, &argv);
	atexit ((void (*) ()) MPI_Finalize); // So it just gets called automatically

	arguments args;
	try
	{
		args = arguments (argc, argv);
	}
	catch (const std::exception& e)
	{
		cerr << e.what () << endl;
		exit (EXIT_FAILURE);
	}

	// Calculate work division

	int my_rank, comm_size;
	if (MPI_Comm_rank (MPI_COMM_WORLD, &my_rank)   != MPI_SUCCESS ||
	    MPI_Comm_size (MPI_COMM_WORLD, &comm_size) != MPI_SUCCESS)
	{
		cerr << "Failed to initialize the program\n";
		exit (EXIT_FAILURE);
	}

	// if (my_rank == 0)
	// 	gdb_hook (0);

	// Calculate image parameters

	const int image_width = args.image_width;
	const int image_height = (2 * image_width) / 3;
	const mandel_float left_viewport_border = args.hcenter - args.view_width / 2;
	const mandel_float top_viewport_border = args.vcenter + args.view_width / 3; // (2/3) / 2
	const mandel_float step = args.view_width / image_width;

	const int NUM_BLOCKS = 1024;
	const int THREADS_PER_BLOCK = 512;

	// Create image

	int kernel_milliseconds = 0;

	const int partition = (image_height + comm_size - 1) / comm_size;
	const int pass_begin = partition * my_rank;
	const int pass_end = min (partition * (my_rank + 1), image_height);
	const int pass_height = pass_end - pass_begin;

	// Allocate local memory

	pixel* pass_buffer = nullptr;
	pixel* image_buffer = nullptr;
	try
	{
		if (my_rank == 0)
			image_buffer = new pixel [partition * comm_size * image_width]; // Potentially over-allocate to prevent SIGSEGV
		pass_buffer = new pixel [pass_height * image_width];
	}
	catch (const std::bad_alloc&)
	{
		cerr << "Failed to allocate local memory\n";
		exit (EXIT_FAILURE);
	}

	// Allocate device memory

	pixel* GPU_image_data = nullptr;
	if (cudaMalloc (reinterpret_cast <void**> (&GPU_image_data), pass_height * image_width * sizeof (pixel)) != cudaSuccess)
	{
		cerr << "Failed to allocate device memory\n";
		exit (EXIT_FAILURE);
	}

	// Begin computation

	auto kernel_start = system_clock::now ();

	do_image (NUM_BLOCKS, THREADS_PER_BLOCK,
	          GPU_image_data,
	          image_width,
	          pass_height,
	          left_viewport_border,
	          top_viewport_border - pass_begin * step,
	          step, args.hsample, args.vsample,
	          NUM_BLOCKS * THREADS_PER_BLOCK);

	cudaDeviceSynchronize ();
	auto kernel_end = system_clock::now ();
	kernel_milliseconds += duration_cast <milliseconds> (kernel_end - kernel_start).count ();

	// Copy back data to host memory and free device memory

	cudaMemcpy (static_cast <void*> (pass_buffer), static_cast <void*> (GPU_image_data),
	            pass_height * image_width * sizeof (pixel), cudaMemcpyDeviceToHost);
	cudaFree (static_cast <void*> (GPU_image_data));

	// Gather on master node

	if (MPI_Gather (static_cast <void*> (pass_buffer),
	                pass_height * image_width * sizeof (pixel),
	                MPI_BYTE,
	                static_cast <void*> (image_buffer),
	                partition * image_width * sizeof (pixel),
	                MPI_BYTE,
	                0,
	                MPI_COMM_WORLD) != MPI_SUCCESS)
	{
		cerr << "Failed to gather image results\n";
		exit (EXIT_FAILURE);
	}

	// Write image

	if (my_rank == 0)
	{
		png::image <png::rgb_pixel> out_image (image_width, image_height);
		for (int row = 0; row < image_height; ++row)
			transform (image_buffer + row * image_width,
			           image_buffer + (row + 1) * image_width,
			           out_image [row].begin (),
			           pixel_convert);

		auto write_start = system_clock::now ();
		out_image.write (args.filename);
		auto write_end = system_clock::now ();

		// Print statistics

		cout << "Kernel: " << kernel_milliseconds                                             << " ms\n"
		     << "Avg:    " << kernel_milliseconds / comm_size                                 << " ms\n"
		     << "Write:  " << duration_cast <milliseconds> (write_end - write_start).count () << " ms" << endl;

		delete [] image_buffer;
	}

	delete [] pass_buffer;
	exit (EXIT_SUCCESS);
}

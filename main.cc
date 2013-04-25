//#include <bitmap.hh>
#include <args.hh>
#include <cuda_wrapper.hh>

#include <iostream>
#include <fstream>
#include <algorithm>
#include <chrono>
#include <mutex>
#include <vector>

#include <png++/png.hpp>
#include <png++/rgb_pixel.hpp>
#include <cuda_runtime.h>
#include <omp.h>
#include <mpi.h>

using namespace std;
using namespace std::chrono;

#define nullptr NULL



namespace
{
	png::rgb_pixel pixel_convert (pixel p)
	{
		return png::rgb_pixel (p.red, p.green, p.blue);
	}

	struct atomic_int
	{
		int t;
		mutex m;

		atomic_int ()
			: t (0)
		{
		}

		int get ()
		{
			return t;
		}

		void set (int i)
		{
			m.lock ();
			t = i;
			m.unlock ();
		}
	};
}



int main (int argc,
          char** argv)
{
	// Program initialization

	MPI_Init (&argc, &argv);
	atexit (MPI_Finalize);

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

	int my_rank, comm_size, n_gpu;
	if (MPI_Comm_rank (MPI_COMM_WORLD, &my_rank)   != MPI_Success ||
	    MPI_Comm_size (MPI_COMM_WORLD, &comm_size) != MPI_Success ||
	    cudaGetDeviceCount (&n_gpu)                != cudaSuccess)
	{
		cerr << "Failed to initialize the program" << endl;
		exit (EXIT_FAILURE);
	}

	vector <int> gpus_by_rank (comm_size);
	if (MPI_Allgather (static_cast <void*> (&n_gpu),
	                   1,
	                   MPI_INT,
	                   static_cast <void*> (gpus_by_rank.data ()),
	                   gpus_by_rank.size (),
	                   MPI_INT,
	                   MPI_COMM_WORLD) != MPI_Success)
	{
		cerr << "Failed to initialize the program" << endl;
		exit (EXIT_FAILURE);
	}
	const int first_pass = accumulate (begin (gpus_by_rank), begin (gpus_by_rank) + my_rank, 0);
	const int n_passes = accumulate (begin (gpus_by_rank), end (gpus_by_rank), 0);

	// Calculate image parameters

	const int image_width = args.image_width;
	const int image_height = (2 * image_width) / 3;
	const mandel_float left_viewport_border = args.hcenter - args.view_width / 2;
	const mandel_float top_viewport_border = args.vcenter + args.view_width / 3; // (2/3) / 2
	const mandel_float step = args.view_width / image_width;

	const int NUM_BLOCKS = 1024;
	const int THREADS_PER_BLOCK = 512;

	// Create image

	atomic_int kernel_milliseconds;
	pixel* image_buffer = nullptr;
	if (my_rank == 0)
		try
		{
			image_buffer = new pixel [image_height * image_width];
		}
		catch (std::bad_alloc&)
		{
			cerr << "Failed to allocate image buffer" << endl;
			exit (EXIT_FAILURE);
		}

	#pragma omp parallel for default (shared) num_threads (n_gpu)
	for (int gpu = 0; gpu < n_gpu; ++gpu)
	{
		cudaSetDevice (gpu);
		const int pass = first_pass + gpu;
		const int pass_begin = (image_height / n_passes) * pass;
		const int pass_end = min ((image_height / n_passes) * (pass + 1), image_height);
		const int pass_height = pass_end - pass_begin;

		// Allocate local memory

		pixel* pass_buffer = nullptr;
		try
		{
			pass_buffer = new pixel [pass_height * image_width];
		}
		catch (const std::bad_alloc&)
		{
			cerr << "Failed to allocate local memory" << endl;
			exit (EXIT_FAILURE);
		}

		// Allocate device memory

		pixel* GPU_image_data = nullptr;
		if (cudaMalloc (reinterpret_cast <void**> (&GPU_image_data), pass_height * image_width * sizeof (pixel)) != cudaSuccess)
		{
			cerr << "Failed to allocate device memory" << endl;
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
		kernel_milliseconds.set (kernel_milliseconds.get () + duration_cast <milliseconds> (kernel_end - kernel_start).count ());

		// Copy back data to host memory

		cudaMemcpy (static_cast <void*> (pass_buffer), static_cast <void*> (GPU_image_data),
		            pass_height * image_width * sizeof (pixel), cudaMemcpyDeviceToHost);

		// Gather on master node

		if (MPI_Gather (static_cast <void*> (pass_buffer),
		                pass_height * image_width * sizeof (pixel),
		                MPI_CHAR,
		                image_buffer,
		                image_height * image_width * sizeof (pixel),
		                MPI_CHAR,
		                0,
		                MPI_COMM_WORLD) != MPI_Success)
		{
			cerr << "Failed to gather image results" << endl;
			exit (EXIT_FAILURE);
		}

		// Free memory

		cudaFree (static_cast <void*> (GPU_image_data));
		delete [] pass_buffer;
	}

	if (my_rank != 0)
		exit (EXIT_SUCCESS);

	// Write to file

	png::image <png::rgb_pixel> out_image (image_width, image_height);
	for (int row = 0; row < image_height; ++row)
		copy (image_buffer + row * image_width,
		      image_buffer + (row + 1) * image_width,
		      out_image [row].begin ());

	auto write_start = system_clock::now ();
	out_image.write (args.filename);
	auto write_end = system_clock::now ();

	// Print statistics

	cout << "Kernel: " << kernel_milliseconds.get ()                                      << " ms\n"
	     << "Avg:    " << kernel_milliseconds.get () / n_passes                           << " ms\n"
	     << "Write:  " << duration_cast <milliseconds> (write_end - write_start).count () << " ms" << endl;

	return EXIT_SUCCESS;
}

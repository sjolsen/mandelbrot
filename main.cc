//#include <bitmap.hh>
#include <args.hh>
#include <cuda_wrapper.hh>

#include <iostream>
#include <fstream>
#include <algorithm>
#include <chrono>

#include <png++/png.hpp>
#include <png++/rgb_pixel.hpp>
#include <cuda_runtime.h>

using namespace std;
using namespace std::chrono;

#define nullptr NULL



namespace
{
	png::rgb_pixel pixel_convert (pixel p)
	{
		return png::rgb_pixel (p.red, p.green, p.blue);
	}
}



int main (int argc,
          char** argv)
{
	arguments args;
	try
	{
		args = arguments (argc, argv);
	}
	catch (const std::exception& e)
	{
		cerr << e.what () << endl;
		return EXIT_FAILURE;
	}

	// Calculate image parameters

	const int image_width = args.image_width;
	const int image_height = (2 * image_width) / 3;
	const mandel_float left_viewport_border = args.hcenter - args.view_width / 2;
	const mandel_float top_viewport_border = args.vcenter + args.view_width / 3; // (2/3) / 2
	const mandel_float step = args.view_width / image_width;

	// Allocate local memory

	pixel* row_buffer = nullptr;
	try
	{
		row_buffer = new pixel [image_width];
	}
	catch (const std::bad_alloc&)
	{
		cerr << "Failed to allocate local memory" << endl;
		return EXIT_FAILURE;
	}
	png::image <png::rgb_pixel> out_image (image_width, image_height);

	// Allocate device memory

	const int NUM_BLOCKS = 1024;
	const int THREADS_PER_BLOCK = 512;
	const int n_passes = 4;

	pixel* GPU_image_data = nullptr;
	if (cudaMalloc (reinterpret_cast <void**> (&GPU_image_data), (image_height / n_passes) * image_width * sizeof (pixel)) != cudaSuccess)
	{
		cerr << "Failed to allocate device memory" << endl;
		return EXIT_FAILURE;
	}

	// Create image

	int kernel_milliseconds = 0;
	int copy_milliseconds = 0;

	for (int pass = 0; pass < n_passes; ++pass)
	{
		const int pass_begin = (image_height / n_passes) * pass;
		const int pass_end = min ((image_height / n_passes) * (pass + 1), image_height);
		const int pass_height = pass_end - pass_begin;

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

		// Copy back data directly into image

		auto copy_start = system_clock::now ();

		for (int row = 0; row < pass_height; ++row)
		{
			cudaMemcpy (static_cast <void*> (row_buffer), static_cast <void*> (GPU_image_data + image_width * row),
			            image_width * sizeof (pixel), cudaMemcpyDeviceToHost);
			transform (row_buffer, row_buffer + image_width, out_image [pass_begin + row].begin (), pixel_convert);
		}

		auto copy_end = system_clock::now ();
		copy_milliseconds += duration_cast <milliseconds> (copy_end - copy_start).count ();
	}

	// Free device memory

	cudaFree (static_cast <void*> (GPU_image_data));

	// Write to file

	auto write_start = system_clock::now ();
	out_image.write (args.filename);
	auto write_end = system_clock::now ();

	// Print statistics

	cout << "Kernel: " << kernel_milliseconds << " ms\n"
	     << "Copy:   " << copy_milliseconds   << " ms\n"
	     << "Write:  " << duration_cast <milliseconds> (write_end - write_start).count () << " ms" << endl;

	return EXIT_SUCCESS;
}

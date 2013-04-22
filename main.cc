//#include <bitmap.hh>
#include <args.hh>
#include <cuda_wrapper.hh>

#include <iostream>
#include <fstream>
#include <algorithm>

#include <png++/png.hpp>
#include <png++/rgb_pixel.hpp>
#include <cuda_runtime.h>

using namespace std;

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
	const float left_viewport_border = args.hcenter - args.view_width / 2;
	const float top_viewport_border = args.vcenter + args.view_width / 3; // (2/3) / 2
	const float step = args.view_width / image_width;

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

	// Allocate device memory

	pixel* GPU_image_data = nullptr;
	if (cudaMalloc (reinterpret_cast <void**> (&GPU_image_data), image_height * image_width * sizeof (pixel)) != cudaSuccess)
	{
		cerr << "Failed to allocate device memory" << endl;
		return EXIT_FAILURE;
	}

	// Create image

	const int NUM_BLOCKS = 256;
	const int THREADS_PER_BLOCK = 128;
	do_image (NUM_BLOCKS, THREADS_PER_BLOCK, GPU_image_data, image_width, image_height, left_viewport_border,
	          top_viewport_border, step, args.hsample, args.vsample, NUM_BLOCKS * THREADS_PER_BLOCK);

	// Copy back data directly into image

	png::image <png::rgb_pixel> out_image (image_width, image_height);

	for (int row = 0; row < image_height; ++row)
	{
		cudaMemcpy (static_cast <void*> (row_buffer), static_cast <void*> (GPU_image_data + image_width * row),
		            image_width * sizeof (pixel), cudaMemcpyDeviceToHost);
		transform (row_buffer, row_buffer + image_width, out_image [row].begin (), pixel_convert);
	}

	// Free device memory

	cudaFree (static_cast <void*> (GPU_image_data));

	// Write to file

	out_image.write (args.filename);
}

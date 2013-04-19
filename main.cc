#include <bitmap.hh>
#include <args.hh>
#include <cuda_wrapper.hh>

#include <iostream>
#include <fstream>
#include <algorithm>

#include <cuda_runtime.h>

using namespace std;

#define nullptr NULL




// namespace
// {

// 	template <int32_t H, int32_t W>
// 	void downsample (pixel (&data) [H][W],
// 	                 pixel (&dest) [H/10][W/10])
// 	{
// 		double r, g, b;
// 		for (int i = 0; i < H/10; ++i)
// 			for (int j = 0; j < W/10; ++j)
// 			{
// 				r = g = b = 0;
// 				for (int k = 0; k < 10; ++k)
// 					for (int l = 0; l < 10; ++l)
// 					{
// 						r += data [i * 10 + k][j * 10 + l].R;
// 						g += data [i * 10 + k][j * 10 + l].G;
// 						b += data [i * 10 + k][j * 10 + l].B;
// 					}
// 				dest [i][j] = pixel {r/100, g/100, b/100};
// 			}
// 	}
// }



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

	pixel** picture = nullptr;
	try
	{
		picture = new pixel* [image_height];
		for (int i = 0; i < image_height; ++i)
			picture [i] = new pixel [image_width];
	}
	catch (const std::bad_alloc&)
	{
		cerr << "Failed to allocate local memory" << endl;
		return EXIT_FAILURE;
	}

	// Allocate device memory

	pixel* GPU_escape_times = nullptr;
	if (cudaMalloc (reinterpret_cast <void**> (&GPU_escape_times), image_height * image_width * sizeof (pixel)) != cudaSuccess)
	{
		cerr << "Failed to allocate device memory" << endl;
		return EXIT_FAILURE;
	}

	// Calculate escape times and copy data back

	const int NUM_BLOCKS = 256;
	const int THREADS_PER_BLOCK = 128;
	const int cols_per_block = ((image_width + NUM_BLOCKS - 1) / NUM_BLOCKS);
	calc_escapes (NUM_BLOCKS, THREADS_PER_BLOCK, GPU_escape_times, image_width,
	              image_height, left_viewport_border, top_viewport_border, step, cols_per_block);

	for (int i = 0; i < image_height; ++i)
		cudaMemcpy (static_cast <void*> (picture [i]), static_cast <void*> (GPU_escape_times + image_width * i),
		            image_width * sizeof (pixel), cudaMemcpyDeviceToHost);

	// Free device memory

	cudaFree (static_cast <void*> (GPU_escape_times));

	// Write to file

	ofstream bigger ("biggercuda.bmp");
	write_bitmap (picture, image_height, image_width, bigger);
}

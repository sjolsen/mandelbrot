#include <cuda_wrapper.hh>

#include <cuComplex.h>

using namespace std;



namespace
{
	inline __device__ cuComplex operator * (const cuComplex& a,
	                                        const cuComplex& b)
	{
		cuComplex z;
		z.x = a.x * b.x - a.y * b.y;
		z.y = a.x * b.y + a.y * b.x;
		return z;
	}

	inline __device__ cuComplex operator + (const cuComplex& a,
	                                        const cuComplex& b)
	{
		cuComplex z;
		z.x = a.x + b.x;
		z.y = a.y + b.y;
		return z;
	}

	__device__ unsigned int escape_time (cuComplex c)
	{
		cuComplex z;
		z.x = 0;
		z.y = 0;
		for (unsigned int n = 1; n < 1024; ++n)
			if (cuCabsf (z = z*z + c) > 2.0f)
				return n;
		return 0;
	}

	__global__ void __calc_escapes (unsigned int* const times, // Using a flat array as a two-dimensinal array
	                                const int image_width,
	                                const int image_height,
	                                const float left_viewport_border,
	                                const float top_viewport_border,
	                                const float step,
	                                const int cols_per_block) // ((image_width + nblocks - 1) / nblocks)
	{
		const register int my_begin = blockIdx.x * cols_per_block;
		const register int my_end = my_begin + cols_per_block;
		register cuComplex c;

		for (int row = threadIdx.x; row < image_height; row += blockDim.x)
		{
			c.y = top_viewport_border + row * step;
			for (int column = my_begin; column < my_end && column < image_width; ++column)
			{
				c.x = left_viewport_border + column * step;
				times [row * image_width + column] = escape_time (c);
			}
		}
	}
}



void calc_escapes (const int NUM_BLOCKS,
                   const int THREADS_PER_BLOCK,
                   unsigned int* const times,
                   const int image_width,
                   const int image_height,
                   const float left_viewport_border,
                   const float top_viewport_border,
                   const float step,
                   const int cols_per_block)
{
	__calc_escapes <<<NUM_BLOCKS, THREADS_PER_BLOCK>>> (times, image_width, image_height, left_viewport_border,
	                                                    top_viewport_border, step, cols_per_block);
}

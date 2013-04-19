#include <cuda_wrapper.hh>

using namespace std;

typedef unsigned int uint32_t;



namespace
{
	__device__ unsigned int escape_time (float real_part,
	                                     float imag_part)
	{
		register float z_real = 0, z_imag = 0;
		register float z_real_sq = 0, z_imag_sq = 0;

		for (unsigned int n = 1; n < 1024; ++n)
		{
			// z = z*z + c
			z_real = z_real_sq + real_part - z_imag_sq;
			z_imag = 2 * z_real * z_imag + imag_part;
			if (z_real_sq + z_imag_sq > 4.0f) // |z| > 2
				return n;
			z_real_sq = z_real * z_real;
			z_imag_sq = z_imag * z_imag;
		}
		return 0;
	}

	__device__ pixel colorize (uint32_t n)
	{
		if (n < 256)
			return (pixel) {0, 0, static_cast <uint8_t> (n & 0xFF)};
		else if (n < 512)
			return (pixel) {static_cast <uint8_t> ((n & 0xFF) >> 1), 0, 0xFF};
		else if (n < 768)
			return (pixel) {static_cast <uint8_t> (0x80 | ((n & 0xFF) >> 1)), static_cast <uint8_t> (n & 0xFF), 0xFF};
		else
			return (pixel) {0xFF, 0xFF, static_cast <uint8_t> (~(n & 0xFF))};
	}

	__device__ void create_pixel (pixel* const target_pixel,
	                              float real_part,
	                              float imag_part,
	                              const float step)
	{
		const int hsample = 1;
		const int vsample = 1;

		register pixel sub_pixel;
		register const float sub_step = step / 16;
		register float r = 0, g = 0, b = 0;

		for (int i = 0; i < vsample; ++i)
		{
			for (int k = 0; k < hsample; ++k)
			{
				sub_pixel = colorize (escape_time (real_part, imag_part));
				r += sub_pixel.R;
				g += sub_pixel.G;
				b += sub_pixel.B;
				real_part += sub_step;
			}
			imag_part -= sub_step;
		}
		sub_pixel.R = r / (hsample * vsample);
		sub_pixel.G = g / (hsample * vsample);
		sub_pixel.B = b / (hsample * vsample);
		*target_pixel = sub_pixel;
	}

	__global__ void __do_image (pixel* const picture,
	                            const int image_width,
	                            const int image_height,
	                            const float left_viewport_border,
	                            const float top_viewport_border,
	                            const float step,
	                            const int nthreads)
	{
		register const int array_length = image_height * image_width;
		register const int my_id = threadIdx.x + blockIdx.x * blockDim.x;

		for (int point = my_id; point < array_length; point += nthreads)
			create_pixel (picture + point,
			              left_viewport_border + (point % image_width) * step,
			              top_viewport_border - (point / image_height) * step,
			              step);
	}
}



void do_image (const int NUM_BLOCKS,
               const int THREADS_PER_BLOCK,
               pixel* const picture,
               const int image_width,
               const int image_height,
               const float left_viewport_border,
               const float top_viewport_border,
               const float step,
               const int nthreads)
{
	__do_image <<<NUM_BLOCKS, THREADS_PER_BLOCK>>> (picture, image_width, image_height, left_viewport_border,
	                                                top_viewport_border, step, nthreads);
}

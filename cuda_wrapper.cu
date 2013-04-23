#include <cuda_wrapper.hh>

using namespace std;

typedef unsigned int uint32_t;

//#define MEMOIZE_COLOR

namespace
{
	__device__ unsigned int escape_time (mandel_float real_part,
	                                     mandel_float imag_part)
	{
		register mandel_float z_real = 0, z_imag = 0, z_tmp;
		register mandel_float z_real_sq = 0, z_imag_sq = 0;

		for (unsigned int n = 1; n < 1024; ++n)
		{
			// z = z*z + c
			z_tmp = z_real_sq - z_imag_sq + real_part;
			z_imag = 2 * z_real * z_imag + imag_part;
			z_real = z_tmp;
			z_real_sq = z_real * z_real;
			z_imag_sq = z_imag * z_imag;
			if (z_real_sq + z_imag_sq > 4.0f) // |z| > 2
				return n;
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

	#ifdef MEMOIZE_COLOR
	__shared__ uint8_t local_red [1024];
	__shared__ uint8_t local_green [1024];
	__shared__ uint8_t local_blue [1024];

	__device__ void load_colors ()
	{
		if (threadIdx.x == 0)
		{
			for (int i = 0; i < 1024; ++i)
			{
				pixel p = colorize (i);
				local_red [i] = p.red;
				local_green [i] = p.green;
				local_blue [i] = p.blue;
			}
		}
	}
	#endif

	__device__ void create_pixel (pixel* const target_pixel,
	                              const mandel_float real_part,
	                              const mandel_float imag_part,
	                              const mandel_float step,
	                              const int hsample,
	                              const int vsample)
	{
		const int total_sample = hsample * vsample;

		register pixel sub_pixel;
		register mandel_float sub_real, sub_imag = imag_part;
		register const mandel_float hstep = step / hsample;
		register const mandel_float vstep = step / vsample;
		register mandel_float r = 0, g = 0, b = 0;

		for (int i = 0; i < vsample; ++i)
		{
			sub_real = real_part;
			for (int k = 0; k < hsample; ++k)
			{
				#ifdef MEMOIZE_COLOR
				register unsigned int etime = escape_time (sub_real, sub_imag);
				sub_pixel = (pixel) {local_red [etime], local_green [etime], local_blue [etime]};
				#else
				sub_pixel = colorize (escape_time (sub_real, sub_imag));
				#endif
				r += sub_pixel.red;
				g += sub_pixel.green;
				b += sub_pixel.blue;
				sub_real = real_part + k * hstep;
			}
			sub_imag = imag_part + i * vstep;
		}
		sub_pixel.red = r / total_sample;
		sub_pixel.green = g / total_sample;
		sub_pixel.blue = b / total_sample;
		*target_pixel = sub_pixel;
	}

	__global__ void __do_image (pixel* const picture,
	                            const int image_width,
	                            const int image_height,
	                            const mandel_float left_viewport_border,
	                            const mandel_float top_viewport_border,
	                            const mandel_float step,
	                            const int hsample,
	                            const int vsample,
	                            const int nthreads)
	{
		register const int array_length = image_height * image_width;
		register const int my_id = threadIdx.x + blockIdx.x * blockDim.x;

		#ifdef MEMOIZE_COLOR
		load_colors ();
		__syncthreads ();
		#endif

		for (int point = my_id; point < array_length; point += nthreads)
			create_pixel (picture + point,
			              left_viewport_border + (point % image_width) * step,
			              top_viewport_border - (point / image_width) * step,
			              step, hsample, vsample);
	}
}



void do_image (const int NUM_BLOCKS,
               const int THREADS_PER_BLOCK,
               pixel* const picture,
               const int image_width,
               const int image_height,
               const mandel_float left_viewport_border,
               const mandel_float top_viewport_border,
               const mandel_float step,
               const int hsample,
               const int vsample,
               const int nthreads)
{
	__do_image <<<NUM_BLOCKS, THREADS_PER_BLOCK>>> (picture, image_width, image_height, left_viewport_border,
	                                                top_viewport_border, step, hsample, vsample, nthreads);
}

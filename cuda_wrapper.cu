#include <cuda_wrapper.hh>

#include <cuda.h>

using namespace std;

typedef unsigned int uint32_t;

#define MEMOIZE_COLOR 1
#define NAIVE_COLOR 0

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

	__device__ mandel_float lerp(mandel_float x, mandel_float a, mandel_float b)
	{
		return fma(x, b, fma(-x, a, a));
	}

	__device__ mandel_float perceptual_to_linear(mandel_float l) {
		#if NAIVE_COLOR
		return l;
		#else
		const mandel_float d = 6.0 / 29.0;
		const mandel_float d2 = pow(d, 2);
		const mandel_float t = (l + 0.16) / 1.16;
		return t > d ? pow(t, 3) : 3.0 * d2 * (t - 4.0 / 29.0);
		#endif
	}

	__device__ mandel_float linear_to_srgb(mandel_float l) {
		#if NAIVE_COLOR
		return l;
		#else
		if (l <= 0.0031308)
			return 12.92 * l;
		else
			return 1.055 * pow(l, 1/2.4) - 0.055;
		#endif
	}

	__device__ pixel<mandel_float> colorize_uncached(uint32_t n) {
		static const pixel<mandel_float> colors[] = {
			{0.0, 0.0, 0.0},  // black
			{0.4, 0.0, 0.0},  // red
			{1.0, 0.5, 0.0},  // orange
			{1.0, 1.0, 0.0},  // yellow
			{1.0, 1.0, 1.0},  // white
			{0.5, 1.0, 1.0},  // cyan
			{0.0, 0.0, 1.0},  // blue
			{0.5, 0.0, 0.5},  // purple
			{0.0, 0.0, 0.0},  // black
		};

		n = 1024 * pow(n / 1024.0, 0.7);
		mandel_float x = (n % 128) / 128.0;
		size_t i = (n / 128) % 8;
		pixel<mandel_float> a = colors[i];
		pixel<mandel_float> b = colors[i + 1];
		return {
			perceptual_to_linear(lerp(x, a.red, b.red)),
			perceptual_to_linear(lerp(x, a.green, b.green)),
			perceptual_to_linear(lerp(x, a.blue, b.blue)),
		};
	}

	#if MEMOIZE_COLOR
	__shared__ pixel<mandel_float> color_cache[1024];
	#endif

	__device__ void load_colors() {
	#if MEMOIZE_COLOR
		for (int i = threadIdx.x; i < 1024; i += blockDim.x)
			color_cache[i] = colorize_uncached(i);
		__syncthreads ();
	#endif
	}

	__device__ pixel<mandel_float> colorize(uint32_t n) {
	#if MEMOIZE_COLOR
		return color_cache[n];
	#else
		return colorize_uncached(n);
	#endif
	}

	__device__ void create_pixel (pixel<uint8_t>* const target_pixel,
	                              const mandel_float real_part,
	                              const mandel_float imag_part,
	                              const mandel_float step,
	                              const int hsample,
	                              const int vsample)
	{
		const int total_sample = hsample * vsample;

		register pixel<mandel_float> sub_pixel;
		register mandel_float sub_real, sub_imag = imag_part;
		register const mandel_float hstep = step / hsample;
		register const mandel_float vstep = step / vsample;
		register mandel_float r = 0, g = 0, b = 0;

		for (int i = 0; i < vsample; ++i)
		{
			sub_real = real_part;
			for (int k = 0; k < hsample; ++k)
			{
				sub_pixel = colorize (escape_time (sub_real, sub_imag));
				r += sub_pixel.red;
				g += sub_pixel.green;
				b += sub_pixel.blue;
				sub_real = real_part + k * hstep;
			}
			sub_imag = imag_part - i * vstep;
		}
		target_pixel->red = 0xFF * linear_to_srgb(r / total_sample);
		target_pixel->green = 0xFF * linear_to_srgb(g / total_sample);
		target_pixel->blue = 0xFF * linear_to_srgb(b / total_sample);
	}

	__global__ void __do_image (pixel<uint8_t>* const picture,
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

		load_colors ();

		for (int point = my_id; point < array_length; point += nthreads)
			create_pixel (picture + point,
			              left_viewport_border + (point % image_width) * step,
			              top_viewport_border - (point / image_width) * step,
			              step, hsample, vsample);
	}
}



void do_image (const int NUM_BLOCKS,
               const int THREADS_PER_BLOCK,
               pixel<uint8_t>* const picture,
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

#ifndef CUDA_WRAPPER_HH
#define CUDA_WRAPPER_HH

#include <pixel.hh>

typedef float mandel_float;



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
               const int cols_per_block);



#endif

#ifndef CUDA_WRAPPER_HH
#define CUDA_WRAPPER_HH

#include <pixel.hh>



void do_image (const int NUM_BLOCKS,
               const int THREADS_PER_BLOCK,
               pixel* const picture,
               const int image_width,
               const int image_height,
               const float left_viewport_border,
               const float top_viewport_border,
               const float step,
               const int cols_per_block);



#endif

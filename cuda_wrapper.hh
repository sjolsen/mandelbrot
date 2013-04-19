#ifndef CUDA_WRAPPER_HH
#define CUDA_WRAPPER_HH



void calc_escapes (const int NUM_BLOCKS,
                   const int THREADS_PER_BLOCK,
                   uint32_t* const times,
                   const int image_width,
                   const int image_height,
                   const float left_viewport_border,
                   const float top_viewport_border,
                   const float step,
                   const int cols_per_block);



#endif
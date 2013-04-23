#ifndef ARGS_HH
#define ARGS_HH

#include <string>
#include <cuda_wrapper.hh>



struct arguments
{
	std::string filename;
	int image_width;
	mandel_float view_width;
	mandel_float hcenter;
	mandel_float vcenter;
	mandel_float hsample;
	mandel_float vsample;

	arguments (const int argc,
	           const char* const* const argv);
	arguments () = default;
};



#endif

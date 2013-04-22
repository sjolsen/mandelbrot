#ifndef ARGS_HH
#define ARGS_HH

#include <string>



struct arguments
{
	std::string filename;
	int image_width;
	float view_width;
	float hcenter;
	float vcenter;
	float hsample;
	float vsample;

	arguments (const int argc,
	           const char* const* const argv);
	arguments () = default;
};



#endif

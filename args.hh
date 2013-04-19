#ifndef ARGS_HH
#define ARGS_HH



struct arguments
{
	int image_width;
	float view_width;
	float hcenter;
	float vcenter;

	arguments (const int argc,
	           const char* const* const argv);
	arguments () = default;
};



#endif

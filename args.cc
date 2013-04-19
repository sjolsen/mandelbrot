#include <args.hh>

#include <stdexcept>
#include <string>

using namespace std;



arguments::arguments (const int argc,
                      const char* const* const argv)
	: image_width (0),
	  view_width (3.0f),
	  hcenter (-0.5f),
	  vcenter (0.0f)
{
	if (argc < 2)
		throw runtime_error (string ("Usage: ") + argv [0] + string (" image_width [view_width hcenter vcenter]"));
	image_width = stoul (argv [1]);

	if (argc < 3)
		return;
	view_width = stof (argv [2]);

	if (argc < 4)
		return;
	hcenter = stof (argv [3]);

	if (argc < 5)
		return;

	vcenter = stof (argv [4]);
}

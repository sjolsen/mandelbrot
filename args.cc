#include <args.hh>

#include <stdexcept>
#include <string>

using namespace std;



arguments::arguments (const int argc,
                      const char* const* const argv)
	: image_width (0),
	  view_width (3.0f),
	  hcenter (-0.5f),
	  vcenter (0.0f),
	  hsample (1),
	  vsample (1)
{
	if (argc < 3)
		throw runtime_error (string ("Usage: ") + argv [0] + string (" image_name image_width [view_width hcenter vcenter hsample vsample]"));
	filename = argv [1];
	image_width = stoul (argv [2]);

	if (argc < 4)
		return;
	view_width = stof (argv [3]);

	if (argc < 5)
		return;
	hcenter = stof (argv [4]);

	if (argc < 6)
		return;
	vcenter = stof (argv [5]);

	if (argc < 7)
		return;
	hsample = stof (argv [6]);

	if (argc < 8)
		return;
	vsample = stof (argv [7]);
}

#include <args.hh>

#include <stdexcept>
#include <string>

using namespace std;



arguments::arguments (const int argc,
                      const char* const* const argv)
	: image_width (0),
	  view_width (3.0),
	  hcenter (-0.5),
	  vcenter (0.0),
	  hsample (1),
	  vsample (1)
{
	if (argc < 3)
		throw runtime_error (string ("Usage: ") + argv [0] + string (" image_name image_width [view_width hcenter vcenter hsample vsample]"));
	filename = argv [1];
	image_width = stoul (argv [2]);

	if (argc < 4)
		return;
	view_width = stod (argv [3]);

	if (argc < 5)
		return;
	hcenter = stod (argv [4]);

	if (argc < 6)
		return;
	vcenter = stod (argv [5]);

	if (argc < 7)
		return;
	hsample = stod (argv [6]);

	if (argc < 8)
		return;
	vsample = stod (argv [7]);
}

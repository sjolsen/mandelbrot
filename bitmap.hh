#ifndef BITMAP_HH
#define BITMAP_HH

#include <ostream>
#include <cstdint>



struct pixel
{
	uint8_t R, G, B;
};

pixel operator * (pixel p, double d);
pixel operator * (double d, pixel p);
pixel& operator += (pixel& p, const pixel& other);

std::ostream& operator << (std::ostream& os,
                           pixel p);

void write_bitmap (pixel** data,
                   std::int32_t height,
                   std::int32_t width,
                   std::ostream& os);



#endif


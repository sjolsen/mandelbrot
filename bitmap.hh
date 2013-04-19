#ifndef BITMAP_HH
#define BITMAP_HH

#include <pixel.hh>

#include <ostream>
#include <cstdint>



std::ostream& operator << (std::ostream& os,
                           pixel p);

void write_bitmap (pixel** data,
                   std::int32_t height,
                   std::int32_t width,
                   std::ostream& os);



#endif

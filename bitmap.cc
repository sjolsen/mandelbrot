#include <bitmap.hh>

#include <iterator>
#include <algorithm>

using namespace std;



namespace
{
	// ALL MULTI-BYTE VALUES ARE LITTLE-ENDIAN

	const uint8_t BMP_HDR [14] = {'B', 'M',      // Bitmap file
	                              0,   0, 0, 0,  // File size
	                              0,   0, 0, 0,  // Reserved bytes
	                              54,  0, 0, 0}; // Image offset

	// BITMAPINFOHEADER
	const uint8_t DIB_HDR [40] = {40,   0,    0, 0,  // Header size
	                              0,    0,    0, 0,  // Image width in px (signed int)
	                              0,    0,    0, 0,  // Image height in px (signed int)
	                              1,    0,           // Color planes
	                              24,   0,           // BPP
	                              0,    0,    0, 0,  // No compression
	                              0,    0,    0, 0,  // Image size
	                              0x23, 0x2E, 0, 0,  // Horizontal resolution in px/m (signed int) (300 dpi)
	                              0x23, 0x2E, 0, 0,  // Vertical resolution in px/m (signed int) (300 dpi)
	                              0,    0,    0, 0,  // Colors in palette (0 for unspecified)
	                              0,    0,    0, 0}; // "Important" colors

	#ifdef BIG_ENDIAN_HOST
	template <typename IntType>
	void write_little_endian (IntType value,
	                          uint8_t* dest)
	{
		for (auto i = 0; i < sizeof (IntType); ++i)
			dest [i] = *(reinterpret_cast <uint8_t*> (&value) + (sizeof (IntType) - 1) - i);
	}
	#else
	template <typename IntType>
	void write_little_endian (IntType value,
	                          uint8_t* dest)
	{
		*(reinterpret_cast <IntType*> (dest)) = value;
	}
	#endif

	template <typename T, size_t N>
	T* begin (T (&array) [N])
	{
		return array;
	}

	template <typename T, size_t N>
	T* end (T (&array) [N])
	{
		return array + N;
	}
}



ostream& operator << (ostream& os,
                      pixel p)
{
	os.put (p.B);
	os.put (p.G);
	os.put (p.R);
	return os;
}



void write_bitmap (pixel** data,
                   int32_t height,
                   int32_t width,
                   ostream& os)
{
	uint32_t ROW_SIZE = ((24 * width + 31) / 32) * 4;
	uint32_t IMAGE_SIZE = ROW_SIZE * height;
	uint32_t FILE_SIZE = 54 + IMAGE_SIZE;

	uint8_t bmp_hdr [14];
	uint8_t dib_hdr [40];

	copy (begin (BMP_HDR), end (BMP_HDR), begin (bmp_hdr));
	copy (begin (DIB_HDR), end (DIB_HDR), begin (dib_hdr));

	write_little_endian (FILE_SIZE, bmp_hdr + 2);
	write_little_endian (width, dib_hdr + 4);
	write_little_endian (height, dib_hdr + 8);

	copy (begin (bmp_hdr), end (bmp_hdr), ostreambuf_iterator <char> (os));
	copy (begin (dib_hdr), end (dib_hdr), ostreambuf_iterator <char> (os));

	for (int32_t row = height - 1; row >= 0; --row)
	{
		copy (data [row], data [row] + width, ostream_iterator <pixel> (os));
		fill_n (ostreambuf_iterator <char> (os), (width * 3) % 4, 0);
	}
}

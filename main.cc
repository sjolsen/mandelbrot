#include <fstream>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <iterator>

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
};

struct pixel
{
	uint8_t R, G, B;
};

ostream& operator << (ostream& os,
                      pixel p)
{
	os.put (p.B);
	os.put (p.G);
	os.put (p.R);
	return os;
}

template <int32_t H, int32_t W>
void write_data_to_stream (pixel (&data) [H][W],
                           ostream& os)
{
	constexpr uint32_t ROW_SIZE = ((24 * W + 31) / 32) * 4;
	constexpr uint32_t IMAGE_SIZE = ROW_SIZE * H;
	constexpr uint32_t FILE_SIZE = 54 + IMAGE_SIZE;

	uint8_t bmp_hdr [14];
	uint8_t dib_hdr [40];

	copy (begin (BMP_HDR), end (BMP_HDR), begin (bmp_hdr));
	copy (begin (DIB_HDR), end (DIB_HDR), begin (dib_hdr));

	write_little_endian (FILE_SIZE, bmp_hdr + 2);
	write_little_endian (W, dib_hdr + 4);
	write_little_endian (H, dib_hdr + 8);

	copy (begin (bmp_hdr), end (bmp_hdr), ostreambuf_iterator <char> (os));
	copy (begin (dib_hdr), end (dib_hdr), ostreambuf_iterator <char> (os));

	for (int32_t row = H - 1; row >= 0; --row)
	{
		copy (data [row], data [row] + W, ostream_iterator <pixel> (os));
		fill_n (ostreambuf_iterator <char> (os), (W * 3) % 4, 0);
	}
}

namespace
{
	constexpr double r_dist (int i, int j)
	{
		return sqrt ((i - 250) * (i - 250) + (j - 250) * (j- 250));
	}
}

int main ()
{
	pixel pic [500][500] = {};

	for (int i = 0; i < 500; ++i)
		for (int j = 0; j < 500; ++j)
			if (/*r_dist (i, j) < 200 || */r_dist (i, j) > 205)
				pic [i][j] = pixel {0xFF, 0xFF, 0xFF};

	ofstream fout ("test.bmp");
	write_data_to_stream (pic, fout);
}

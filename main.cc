#include <fstream>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <iterator>
#include <complex>

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
		return sqrt ((i - 2500) * (i - 2500) + (j - 2500) * (j- 2500));
	}

	inline pixel operator * (pixel p, double d)
	{
		return pixel {p.R * d, p.G * d, p.B * d};
	}

	inline pixel operator * (double d, pixel p)
	{
		return pixel {p.R * d, p.G * d, p.B * d};
	}

	inline pixel& operator += (pixel& p, const pixel& other)
	{
		p.R += other.R;
		p.G += other.G;
		p.B += other.B;
		return p;
	}

	template <int32_t H, int32_t W>
	void downsample (pixel (&data) [H][W],
	                 pixel (&dest) [H/10][W/10])
	{
		double r, g, b;
		for (int i = 0; i < H/10; ++i)
			for (int j = 0; j < W/10; ++j)
			{
				r = g = b = 0;
				for (int k = 0; k < 10; ++k)
					for (int l = 0; l < 10; ++l)
					{
						r += data [i * 10 + k][j * 10 + l].R;
						g += data [i * 10 + k][j * 10 + l].G;
						b += data [i * 10 + k][j * 10 + l].B;
					}
				dest [i][j] = pixel {r/100, g/100, b/100};
			}
	}
}

uint8_t escape_time (complex <double> c)
{
	complex <double> z = 0;
	uint8_t n = 0;
	for (; n != 255; ++n)
		if (abs (z = z*z + c) > 2)
			break;
	return n;
}

int main ()
{
	pixel (&pic) [10000][15000] = *reinterpret_cast <pixel (*) [10000][15000]> (new pixel [150000000]);
	pixel (&pic2) [1000][1500] = *reinterpret_cast <pixel (*) [1000][1500]> (new pixel [150000000]);

	#pragma omp parallel for num_threads (4)
	for (int i = 0; i < 10000; ++i)
		for (int j = 0; j < 15000; ++j)
		{
			auto n = escape_time (complex <double> ((j / 5000.) - 2, 1 - (i / 5000.)));
			pic [i][j] = pixel {n, n, n};
		}

	{
		ofstream bigger ("bigger.bmp");
		write_data_to_stream (pic, bigger);
	}

	downsample (pic, pic2);

	{
		ofstream big ("big.bmp");
		write_data_to_stream (pic2, big);
	}
}

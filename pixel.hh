#ifndef PIXEL_HH
#define PIXEL_HH

template <typename T>
struct pixel {
	T red, green, blue;

	template <typename U>
	pixel<U> convert() const {
		return pixel<U> {
			static_cast<U>(red),
			static_cast<U>(green),
			static_cast<U>(blue),
		};
	}
};

#endif

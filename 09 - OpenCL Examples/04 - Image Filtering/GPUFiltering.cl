unsigned char GetPixel(
	__global unsigned char * src,
	unsigned int w, unsigned int h, int x, int y, int c)
{
	x = x < 0 ? 0 : x;
	x = x < w ? x : w - 1;
	y = y < 0 ? 0 : y;
	y = y < h ? y : h - 1;
	return src[3 * (w * y + x) + c];
}

__kernel
void greyscale_kernel(
	__global unsigned char * src,
	__global unsigned char * dest,
	unsigned int w, unsigned int h)
{
	// определяем координаты пикселя
	unsigned int x = get_global_id(0);
	unsigned int y = get_global_id(1);

	if (y < h && x < w)
	{
		float luma = 0
			+ 0.2126f * GetPixel(src, w, h, x, y, 0)
			+ 0.7152f * GetPixel(src, w, h, x, y, 1)
			+ 0.0722f * GetPixel(src, w, h, x, y, 2);
		
		unsigned char l = (unsigned char)luma;
		dest[3 * (w * y + x) + 0] = l;
		dest[3 * (w * y + x) + 1] = l;
		dest[3 * (w * y + x) + 2] = l;
	}
}

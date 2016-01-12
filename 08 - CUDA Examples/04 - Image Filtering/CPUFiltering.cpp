void greyscaleCPU(unsigned char * src, unsigned char * dest, unsigned int w, unsigned int h)
{
	for (unsigned int y = 0; y < h; ++y)
		for (unsigned int x = 0; x < w; ++x)
		{
			unsigned char r = src[3 * (w * y + x) + 0];
			unsigned char g = src[3 * (w * y + x) + 1];
			unsigned char b = src[3 * (w * y + x) + 2];

			unsigned char l = static_cast<unsigned char>(0.2126f * r + 0.7152f * g + 0.0722f * b);
			dest[3 * (w * y + x) + 0] = l;
			dest[3 * (w * y + x) + 1] = l;
			dest[3 * (w * y + x) + 2] = l;
		}
}

extern "C"
void CPUFiltering(unsigned char * src, unsigned char * dest, unsigned int w, unsigned int h)
{
	greyscaleCPU(src, dest, w, h);
}

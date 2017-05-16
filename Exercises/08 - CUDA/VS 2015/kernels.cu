const unsigned char BLOCK_DIM_X = 16;
const unsigned char BLOCK_DIM_Y = 16;

__device__
unsigned char GetPixel(unsigned char * src, unsigned int w, unsigned int h, int x, int y, int c)
{
	x = x < 0 ? 0 : x;
	x = x < w ? x : w - 1;
	y = y < 0 ? 0 : y;
	y = y < h ? y : h - 1;
	return src[3 * (w * y + x) + c];
}

__global__
void filter_kernel(unsigned char * src, unsigned char * dest, unsigned int w, unsigned int h)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < w && y < h)
	{
		dest[3 * (w * y + x) + 0] = GetPixel(src, w, h, x, y, 0);
		dest[3 * (w * y + x) + 1] = GetPixel(src, w, h, x, y, 1);
		dest[3 * (w * y + x) + 2] = GetPixel(src, w, h, x, y, 2);
	}
}

extern "C"
void Filter(unsigned char * src, unsigned char * dest, unsigned int w, unsigned int h)
{
	// код фильтра
	dim3 BlockDim1(BLOCK_DIM_X, BLOCK_DIM_Y, 1);
	dim3 GridDim1((w - 1)/BlockDim1.x + 1, (h - 1)/BlockDim1.y + 1, 1);
	filter_kernel<<<GridDim1, BlockDim1>>>(src, dest, w, h);
}

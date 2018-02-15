__global__
void mult_matrix_kernel_simple(float *A, float *B, float *C, unsigned int N, unsigned int L, unsigned int M)
{
	// определяем место потока в массиве
	unsigned int tx = threadIdx.x;
	unsigned int ty = threadIdx.y;
	unsigned int bx = blockIdx.x;
	unsigned int by = blockIdx.y;

	// определяем рассчетный элемент матрицы
	unsigned int row = by * blockDim.y + ty;
	unsigned int col = bx * blockDim.x + tx;

	if (row < N && col < M)
	{
		float sum = 0.0f;
		for (unsigned int i = 0; i < L; ++i)
			sum += A[row*L + i]*B[i*M + col];
		C[row*M + col] = sum;
	}
}

__global__
void mult_matrix_kernel_shared(float *A, float *B, float *C, unsigned int N, unsigned int L, unsigned int M)
{
	// разделяемая память для хранения блока элементов исходных матриц
	__shared__ float ds_A[16][16];
	__shared__ float ds_B[16][16];

	// определяем место потока в массиве
	unsigned int tx = threadIdx.x;
	unsigned int ty = threadIdx.y;
	unsigned int bx = blockIdx.x;
	unsigned int by = blockIdx.y;

	// определяем рассчетный элемент матрицы
	unsigned int row = by * blockDim.y + ty;
	unsigned int col = bx * blockDim.x + tx;

	float sum = 0;
	unsigned int nCycles = (L - 1)/16 + 1;
	for (int i = 0; i < nCycles; ++i)
	{
		// копируем элемент исходной матрицы A
		if (row < N && i*blockDim.x + tx < L)
			ds_A[ty][tx] = A[row*N + i*blockDim.x + tx];
		else
			ds_A[ty][tx] = 0;

		// копируем элемент исходной матрицы B
		if (i*blockDim.x + ty < L && col < M)
			ds_B[ty][tx] = B[(i*blockDim.x + ty)*M + col];
		else
			ds_B[ty][tx] = 0;

		__syncthreads();

		// накопление частичной суммы
		for (int j = 0; j < blockDim.x; ++j)
			sum += ds_A[ty][j] * ds_B[j][tx];

		__syncthreads();
	}

	// запись элемента результирующей матрицы
	if (row < N && col < M)
		C[row*M + col] = sum;
}

extern "C"
void multMatrixSimple(float *A, float *B, float *C, unsigned int N, unsigned int L, unsigned int M)
{
	dim3 BlockDim(16, 16, 1);
	dim3 GridDim((N - 1)/BlockDim.x + 1, (M - 1)/BlockDim.y + 1, 1);
	mult_matrix_kernel_simple<<<GridDim, BlockDim>>>(A, B, C, N, L, M);
}

extern "C"
void multMatrixShared(float *A, float *B, float *C, unsigned int N, unsigned int L, unsigned int M)
{
	dim3 BlockDim(16, 16, 1);
	dim3 GridDim((N - 1)/BlockDim.x + 1, (M - 1)/BlockDim.y + 1, 1);
	mult_matrix_kernel_shared<<<GridDim, BlockDim>>>(A, B, C, N, L, M);
}

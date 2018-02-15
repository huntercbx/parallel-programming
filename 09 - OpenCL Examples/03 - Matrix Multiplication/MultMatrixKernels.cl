__kernel
void mult_matrix_kernel_simple(__global float *A, __global float *B, __global float *C,
	unsigned int N, unsigned int L, unsigned int M)
{
	// определяем рассчетный элемент матрицы
	unsigned int row = get_global_id(0);
	unsigned int col = get_global_id(1);

	if (row < N && col < M)
	{
		float sum = 0.0f;
		for (unsigned int i = 0; i < L; ++i)
			sum += A[row*L + i]*B[i*M + col];
		C[row*M + col] = sum;
	}
}

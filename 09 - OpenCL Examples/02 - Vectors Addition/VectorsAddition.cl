// OpenCL Kernel
__kernel
void add_vectors_kernel(__global int* A, __global int* B, __global int* C)
{
	unsigned int tx = get_global_id(0);
	C[tx] = A[tx] + B[tx];
}

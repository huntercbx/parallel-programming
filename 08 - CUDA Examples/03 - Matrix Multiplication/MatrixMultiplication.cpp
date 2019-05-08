////////////////////////////////////////////////////////////////////////////////
// Данная программа генерирует две матрицы и перемножает их.
// Все элементы первой матрицы N * L равны 1.0f
// Все элементы второй матрицы L * M равны 2.0f
// В этом случае элемент результирующей матрицы равен L * 2.0f
////////////////////////////////////////////////////////////////////////////////
#include <cstdio>

// заголовочные файлы CUDA
#include <cuda.h>
#include <cuda_runtime.h>

// Макрос для проверки на ошибку, при вызове функций CUDA
#define CUDA_CHECK(call) \
	if((call) != cudaSuccess) { \
		cudaError_t err = cudaGetLastError(); \
		printf( "CUDA error calling \"%s\", error code is %d\n", #call, err); \
		exit(-1); }

extern "C"
void multMatrixSimple(float *A, float *B, float *C, unsigned int N, unsigned int L, unsigned int M);

extern "C"
void multMatrixShared(float *A, float *B, float *C, unsigned int N, unsigned int L, unsigned int M);

int main(int argc, char* argv[])
{
	// проверка на наличие устройств с поддержкой CUDA
	int deviceCount = 0;
	CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
	if (deviceCount == 0)
	{
		printf("There are no available device(s) that support CUDA\n");
		return 0;
	}

	// используем первое устройство
	cudaSetDevice(0);

	// генерация исходных матриц
	const size_t N = 1000;//211;
	const size_t L = 1000;//101;
	const size_t M = 1000;//151;

	// размеры матриц в байтах
	size_t sizeA = N*L*sizeof(float);
	size_t sizeB = L*M*sizeof(float);
	size_t sizeC = N*M*sizeof(float);

	// выделение памяти под матрицы в ОЗУ
	float *host_A = (float *)malloc(sizeA);
	float *host_B = (float *)malloc(sizeB);
	float *host_C = (float *)malloc(sizeC);

	for (size_t row = 0; row < N; ++row)
		for (size_t col = 0; col < L; ++col)
			host_A[row*L + col] = 1.0f;

	for (size_t row = 0; row < L; ++row)
		for (size_t col = 0; col < M; ++col)
			host_B[row*M + col] = 1.0f;

	cudaEvent_t evt1, evt2, evt3, evt4, evt5, evt6;
	cudaEventCreate(&evt1);
	cudaEventCreate(&evt2);
	cudaEventCreate(&evt3);
	cudaEventCreate(&evt4);
	cudaEventCreate(&evt5);
	cudaEventCreate(&evt6);

	// выделение видеопамяти под матрицы
	cudaEventRecord(evt1, 0);
	float *dev_A, *dev_B, *dev_C;
	CUDA_CHECK(cudaMalloc((void**)&dev_A, sizeA));
	CUDA_CHECK(cudaMalloc((void**)&dev_B, sizeB));
	CUDA_CHECK(cudaMalloc((void**)&dev_C, sizeC));

	// копирование исходных матриц в видеопамять
	cudaEventRecord(evt2, 0);
	CUDA_CHECK(cudaMemcpy(dev_A, host_A, sizeA, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(dev_B, host_B, sizeB, cudaMemcpyHostToDevice));

	// запуск вычислений на GPU
	cudaEventRecord(evt3, 0);
	multMatrixShared(dev_A, dev_B, dev_C, N, L, M);

	// копирование результирующей матрицы из видеопамять в ОЗУ
	cudaEventRecord(evt4, 0);
	CUDA_CHECK(cudaMemcpy(host_C, dev_C, sizeC, cudaMemcpyDeviceToHost));

	// освобождаем видеопамять
	cudaEventRecord(evt5, 0);
	CUDA_CHECK(cudaFree(dev_A));
	CUDA_CHECK(cudaFree(dev_B));
	CUDA_CHECK(cudaFree(dev_C));

	// окончание работы
	cudaEventRecord(evt6, 0);
	cudaEventSynchronize(evt6);

	printf("Matricies dimensions:                [%zd x %zd] * [%zd x %zd]\n", N, L, L, M);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, evt1, evt2);
	printf("Video memory allocation:             %g ms\n", elapsedTime);
	cudaEventElapsedTime(&elapsedTime, evt2, evt3);
	printf("Copying source data to video memory: %g ms\n", elapsedTime);
	cudaEventElapsedTime(&elapsedTime, evt3, evt4);
	printf("Calculation:                         %g ms\n", elapsedTime);
	cudaEventElapsedTime(&elapsedTime, evt4, evt5);
	printf("Copying results from video memory:   %g ms\n", elapsedTime);
	cudaEventElapsedTime(&elapsedTime, evt5, evt6);
	printf("Video memory free:                   %g ms\n", elapsedTime);
	cudaEventElapsedTime(&elapsedTime, evt1, evt6);
	printf("Total time:                          %g ms\n", elapsedTime);

	cudaEventDestroy(evt1);
	cudaEventDestroy(evt2);
	cudaEventDestroy(evt3);
	cudaEventDestroy(evt4);
	cudaEventDestroy(evt5);
	cudaEventDestroy(evt6);

	printf("Resulting matrix element:            %g\n", host_C[N]);

	free(host_A);
	free(host_B);
	free(host_C);

	return 0;
}

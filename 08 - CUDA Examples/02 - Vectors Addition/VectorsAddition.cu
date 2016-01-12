#include <cstdio>

// заголовочные файлы CUDA
#include <cuda.h>
#include <cuda_runtime.h>

// размер векторов
const size_t N = 10;

// Макрос для проверки на ошибку, при вызове функций CUDA
#define CUDA_CHECK(call) \
	if((call) != cudaSuccess) { \
		cudaError_t err = cudaGetLastError(); \
		printf( "CUDA error calling \"%s\", error code is %d\n", #call, err); \
		exit(-1); }

// Функция для выполнения на видеокарте
__global__ void add_vectors_kernel(int *A, int *B, int *C)
{
	// индекс обрабатываемого элемента
	int tx = blockIdx.x;
	if (tx < N)
		C[tx] = A[tx] + B[tx];
}

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

	// инициализация исходных векторов
	int host_A[N], host_B[N], host_C[N];
	for (size_t i = 0; i < N; ++i)
	{
		host_A[i] = static_cast<int>(i);
		host_B[i] = static_cast<int>(N*i);
	}

	// выделение видеопамяти под вектора
	int *dev_A, *dev_B, *dev_C;
	cudaMalloc((void**)&dev_A, N * sizeof(int));
	cudaMalloc((void**)&dev_B, N * sizeof(int));
	cudaMalloc((void**)&dev_C, N * sizeof(int));

	// копирование исходных векторов в видеопамять
	cudaMemcpy(dev_A, host_A, N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_B, host_B, N * sizeof(int), cudaMemcpyHostToDevice);

	// запуск вычислений на GPU
	add_vectors_kernel<<<N,1>>>(dev_A, dev_B, dev_C);

	// копирование результирующего вектора из видеопамяти в ОЗУ
	cudaMemcpy(host_C, dev_C, N * sizeof(int), cudaMemcpyDeviceToHost);

	// освобождение видеопамяти
	cudaFree(dev_A);
	cudaFree(dev_B);
	cudaFree(dev_C);

	// вывод резудьтирующего вектора на экран
	for (size_t i = 0; i < N; ++i)
		printf("%d + %d = %d\n", host_A[i], host_B[i], host_C[i]);

	return 0;
}

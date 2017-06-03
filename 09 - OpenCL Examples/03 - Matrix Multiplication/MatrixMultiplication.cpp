#include <vector>
#include <CL/cl.h>
#include "../utils/utils.h"

using namespace std;

int main(int argc, char* argv[])
{
	// Проверка на наличие устройств с поддержкой OpenCL
	cl_uint num_platforms;
	OPENCL_CHECK(clGetPlatformIDs(1, NULL, &num_platforms));
	if (num_platforms == 0)
	{
		printf("There are no available device(s) that support OpenCL\n");
		exit(EXIT_FAILURE);
	}
	cl_platform_id *platforms = new cl_platform_id[num_platforms];
	OPENCL_CHECK(clGetPlatformIDs(num_platforms, platforms, nullptr));

	// Используем первую платформу и получаем список устройств
	cl_uint num_devices;
	OPENCL_CHECK(clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, 0, nullptr, &num_devices));
	cl_device_id *devices = new cl_device_id[num_devices];
	OPENCL_CHECK(clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, num_devices, devices, nullptr));
	cl_device_id device_id = devices[0];

	// Создание контекста вычислений
	cl_int err;
	cl_context context = clCreateContext(nullptr, 1, &device_id, nullptr, nullptr, &err);
	if (!context)
	{
		printf("Error: Failed to create a compute context!\n");
		return EXIT_FAILURE;
	}

	// Создание очереди команд
	cl_command_queue commands = clCreateCommandQueue(context, device_id, 0, &err);
	if (!commands)
	{
		printf("Error: Failed to create a command queue!\n");
		return EXIT_FAILURE;
	}

	// Генерация исходных матриц
	const unsigned int N = 1000;//211;
	const unsigned int L = 1000;//101;
	const unsigned int M = 1000;//151;

	// Размеры матриц в байтах
	size_t sizeA = N*L * sizeof(float);
	size_t sizeB = L*M * sizeof(float);
	size_t sizeC = N*M * sizeof(float);

	// Выделение памяти под матрицы в ОЗУ
	float *host_A = (float *)malloc(sizeA);
	float *host_B = (float *)malloc(sizeB);
	float *host_C = (float *)malloc(sizeC);

	for (size_t row = 0; row < N; ++row)
		for (size_t col = 0; col < L; ++col)
			host_A[row*L + col] = 1.0f;

	for (size_t row = 0; row < L; ++row)
		for (size_t col = 0; col < M; ++col)
			host_B[row*M + col] = 2.0f;

	// Читаем и строим программу
	cl_program program = CreateAndBuildProgram("MultMatrixKernels.cl", context, device_id);

	// Создание ядра в програме для запуска
	cl_kernel kernel = clCreateKernel(program, "mult_matrix_kernel_simple", &err);
	if (!kernel || err != CL_SUCCESS)
	{
		printf("Error: Failed to create compute kernel!\n");
		return EXIT_FAILURE;
	}

	// Выделение видеопамяти под матрицы и копирование исходных матриц в память устройства
	cl_mem dev_A = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeA, host_A, &err);
	cl_mem dev_B = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeB, host_B, &err);
	cl_mem dev_C = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeC, nullptr, &err);
	if (!dev_A || !dev_B || !dev_C)
	{
		printf("Error: Failed to allocate device memory!\n");
		return EXIT_FAILURE;
	}

	// Передача параметров ядра
	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&dev_A);
	err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&dev_B);
	err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&dev_C);
	err |= clSetKernelArg(kernel, 3, sizeof(unsigned int), (void *)&N);
	err |= clSetKernelArg(kernel, 4, sizeof(unsigned int), (void *)&L);
	err |= clSetKernelArg(kernel, 5, sizeof(unsigned int), (void *)&M);
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to set kernel arguments! %d\n", err);
		return EXIT_FAILURE;
	}

	size_t localWorkSize[2] = { 16, 16 };
	size_t globalWorkSize[2] = {
		((N - 1) / localWorkSize[0] + 1) * localWorkSize[0],
		((M - 1) / localWorkSize[1] + 1) * localWorkSize[1]
	};
	OPENCL_CHECK(clEnqueueNDRangeKernel(commands, kernel, 2, nullptr, globalWorkSize, localWorkSize, 0, nullptr, nullptr));

	// Копирование результирующей матрицы из памяти устройства в ОЗУ
	OPENCL_CHECK(clEnqueueReadBuffer(commands, dev_C, CL_TRUE, 0, sizeC, host_C, 0, nullptr, nullptr));

	// Ожидание окончания рассчетов
	OPENCL_CHECK(clFinish(commands));

	printf("Matricies dimensions:                [%d x %d] * [%d x %d]\n", N, L, L, M);
	printf("Resulting matrix element:            %g\n", host_C[N]);

	return 0;
}

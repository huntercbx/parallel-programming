#include <vector>
#include <CL/cl.h>
#include "../utils/utils.h"

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

	// Создание очереди команд (с включенным профилированием)
	cl_command_queue commands = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &err);
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

	// Запуск ядра с измерением времени его выполнения
	cl_event kernelExecutionEvent;
	const cl_uint workDimensions = 2;
	size_t localWorkSize[workDimensions] = { 16, 16 };
	size_t globalWorkSize[workDimensions] = {
		((N - 1) / localWorkSize[0] + 1) * localWorkSize[0],
		((M - 1) / localWorkSize[1] + 1) * localWorkSize[1]
	};
	OPENCL_CHECK(clEnqueueNDRangeKernel(commands, kernel,
		workDimensions, nullptr, globalWorkSize, localWorkSize,
		0, nullptr, &kernelExecutionEvent));
	OPENCL_CHECK(clWaitForEvents(1, &kernelExecutionEvent));

	// Копирование результирующей матрицы из памяти устройства в ОЗУ
	cl_event readBufferEvent;
	OPENCL_CHECK(clEnqueueReadBuffer(commands, dev_C, CL_TRUE, 0, sizeC, host_C, 0, nullptr, &readBufferEvent));
	OPENCL_CHECK(clWaitForEvents(1, &readBufferEvent));

	// Ожидание окончания расчетов
	OPENCL_CHECK(clFinish(commands));

	cl_ulong time_start, time_end;
	OPENCL_CHECK(clGetEventProfilingInfo(kernelExecutionEvent, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, nullptr));
	OPENCL_CHECK(clGetEventProfilingInfo(kernelExecutionEvent, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, nullptr));
	double kernelExecutionTime = (time_end - time_start)/1e9;
	OPENCL_CHECK(clGetEventProfilingInfo(readBufferEvent, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, nullptr));
	OPENCL_CHECK(clGetEventProfilingInfo(readBufferEvent, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, nullptr));
	double readBufferTime = (time_end - time_start) / 1e9;

	// Освобождаем ресурсы
	OPENCL_CHECK(clReleaseKernel(kernel));
	OPENCL_CHECK(clReleaseProgram(program));
	OPENCL_CHECK(clReleaseMemObject(dev_A));
	OPENCL_CHECK(clReleaseMemObject(dev_B));
	OPENCL_CHECK(clReleaseMemObject(dev_C));
	OPENCL_CHECK(clReleaseCommandQueue(commands));
	OPENCL_CHECK(clReleaseDevice(device_id));
	OPENCL_CHECK(clReleaseContext(context));

	printf("Matricies dimensions:                [%d x %d] * [%d x %d]\n", N, L, L, M);
	printf("Resulting matrix element:            %g\n", host_C[N]);
	printf("Kernel execution time:               %g s\n", kernelExecutionTime);
	printf("Copying result time:                 %g s\n", readBufferTime);

	return 0;
}

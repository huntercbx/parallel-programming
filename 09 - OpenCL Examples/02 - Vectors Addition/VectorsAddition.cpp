#include <vector>
#include <CL/cl.h>

#include "../utils/utils.h"

// размер векторов
const unsigned int N = 10;

int main(int argc, char* argv[])
{
	// проверка на наличие устройств с поддержкой OpenCL
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
		printf("Error: Failed to create a command commands!\n");
		return EXIT_FAILURE;
	}

	// инициализация исходных векторов
	int host_A[N], host_B[N], host_C[N];
	for (size_t i = 0; i < N; ++i)
	{
		host_A[i] = static_cast<int>(i);
		host_B[i] = static_cast<int>(N*i);
	}

	// Читаем и строим программу ядра
	cl_program program = CreateAndBuildProgram("VectorsAddition.cl", context, device_id);

	// Create the compute kernel in the program we wish to run
	cl_kernel kernel = clCreateKernel(program, "add_vectors_kernel", &err);
	if (!kernel || err != CL_SUCCESS)
	{
		printf("Error: Failed to create compute kernel!\n");
		return EXIT_FAILURE;
	}

	// выделение видеопамяти под вектора и копирование исходных векторов в память устройства
	const size_t mem_size = N * sizeof(float);
	cl_mem dev_A = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, mem_size, host_A, &err);
	cl_mem dev_B = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, mem_size, host_B, &err);
	cl_mem dev_C = clCreateBuffer(context, CL_MEM_WRITE_ONLY, mem_size, nullptr, &err);
	if (!dev_A || !dev_B || !dev_C)
	{
		printf("Error: Failed to allocate device memory!\n");
		return EXIT_FAILURE;
	}

	// передача параметров ядра
	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&dev_A);
	err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&dev_B);
	err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&dev_C);
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to set kernel arguments! %d\n", err);
		return EXIT_FAILURE;
	}

	size_t globalWorkSize[1] = { N };
	OPENCL_CHECK(clEnqueueNDRangeKernel(commands, kernel, 1, nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr));

	// копирование результирующего вектора из видеопамяти в ОЗУ
	OPENCL_CHECK(clEnqueueReadBuffer(commands, dev_C, CL_TRUE, 0, mem_size, host_C, 0, nullptr, nullptr));

	// Ожидание окончания рассчетов
	OPENCL_CHECK(clFinish(commands));

	// Освобождаем ресурсы
	OPENCL_CHECK(clReleaseKernel(kernel));
	OPENCL_CHECK(clReleaseProgram(program));
	OPENCL_CHECK(clReleaseMemObject(dev_A));
	OPENCL_CHECK(clReleaseMemObject(dev_B));
	OPENCL_CHECK(clReleaseMemObject(dev_C));
	OPENCL_CHECK(clReleaseCommandQueue(commands));
	OPENCL_CHECK(clReleaseDevice(device_id));
	OPENCL_CHECK(clReleaseContext(context));
	
	// вывод резудьтирующего вектора на экран
	for (size_t i = 0; i < N; ++i)
		printf("%d + %d = %d\n", host_A[i], host_B[i], host_C[i]);

	return 0;
}

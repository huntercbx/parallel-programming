#include <cstdio>
#include <CL/cl.h>
#include "../utils/utils.h"

// заголовочный файл для работы с PPM изображениями
#include "PPMImage.h"

int main(int argc, char* argv[])
{
	// имена входного и выходного файла должны передаваться через параметры командной строки
	if (argc != 3)
	{
		printf("Usage: %s input_file output_file \n", argv[0]);
		exit(-1);
	}

	// загружаем изображение из файла
	PPMImage original_image(argv[1]);
	if (original_image.image == nullptr) return -1;

	// копируем настройки изображения и выделяеям память под результат
	PPMImage result_image;
	result_image.width = original_image.width;
	result_image.height = original_image.height;
	result_image.max_color_val = original_image.max_color_val;
	result_image.allocate_memory();

	unsigned int img_size = original_image.get_required_mem_size();
	unsigned int width = original_image.width;
	unsigned int height = original_image.height;

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

	// Читаем и строим программу
	cl_program program = CreateAndBuildProgram("GPUFiltering.cl", context, device_id);

	// Создание ядра в програме для запуска
	cl_kernel kernel = clCreateKernel(program, "greyscale_kernel", &err);
	if (!kernel || err != CL_SUCCESS)
	{
		printf("Error: Failed to create compute kernel!\n");
		return EXIT_FAILURE;
	}

	// Выделение видеопамяти под изображения и копирование исходного изображения в память устройства
	cl_mem dev_origin = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, img_size, original_image.image, &err);
	cl_mem dev_result = clCreateBuffer(context, CL_MEM_WRITE_ONLY, img_size, nullptr, &err);
	if (!dev_origin || !dev_result)
	{
		printf("Error: Failed to allocate device memory!\n");
		return EXIT_FAILURE;
	}

	// Передача параметров ядра
	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&dev_origin);
	err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&dev_result);
	err |= clSetKernelArg(kernel, 2, sizeof(unsigned int), (void *)&width);
	err |= clSetKernelArg(kernel, 3, sizeof(unsigned int), (void *)&height);
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
		((width - 1) / localWorkSize[0] + 1) * localWorkSize[0],
		((height - 1) / localWorkSize[1] + 1) * localWorkSize[1]
	};
	OPENCL_CHECK(clEnqueueNDRangeKernel(commands, kernel,
		workDimensions, nullptr, globalWorkSize, localWorkSize,
		0, nullptr, &kernelExecutionEvent));
	OPENCL_CHECK(clWaitForEvents(1, &kernelExecutionEvent));

	// Копирование результирующей матрицы из памяти устройства в ОЗУ
	cl_event readBufferEvent;
	OPENCL_CHECK(clEnqueueReadBuffer(commands, dev_result, CL_TRUE, 0, img_size, result_image.image, 0, nullptr, &readBufferEvent));
	OPENCL_CHECK(clWaitForEvents(1, &readBufferEvent));

	// Ожидание окончания расчетов
	OPENCL_CHECK(clFinish(commands));

	cl_ulong time_start, time_end;
	OPENCL_CHECK(clGetEventProfilingInfo(kernelExecutionEvent, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, nullptr));
	OPENCL_CHECK(clGetEventProfilingInfo(kernelExecutionEvent, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, nullptr));
	double kernelExecutionTime = (time_end - time_start) / 1e9;
	OPENCL_CHECK(clGetEventProfilingInfo(readBufferEvent, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, nullptr));
	OPENCL_CHECK(clGetEventProfilingInfo(readBufferEvent, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, nullptr));
	double readBufferTime = (time_end - time_start) / 1e9;

	// Освобождаем ресурсы
	OPENCL_CHECK(clReleaseKernel(kernel));
	OPENCL_CHECK(clReleaseProgram(program));
	OPENCL_CHECK(clReleaseMemObject(dev_origin));
	OPENCL_CHECK(clReleaseMemObject(dev_result));
	OPENCL_CHECK(clReleaseCommandQueue(commands));
	OPENCL_CHECK(clReleaseDevice(device_id));
	OPENCL_CHECK(clReleaseContext(context));

	printf("Image size:                          %d x %d\n", width, height);
	printf("Kernel execution time:               %g s\n", kernelExecutionTime);
	printf("Copying result time:                 %g s\n", readBufferTime);

	// сохраняем обработанное изображение в файл
	result_image.save(argv[2]);

	return 0;
}

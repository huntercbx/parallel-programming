#include <stdio.h>
#include <vector>
#include <CL\cl.h>
#include "utils.h"

// Upload the OpenCL C source code to output argument source
// The memory resource is implicitly allocated in the function
// and should be deallocated by the caller
int ReadSourceFromFile(const char* fileName, char** source, size_t* sourceSize)
{
	int errorCode = CL_SUCCESS;

	FILE* file = nullptr;
	fopen_s(&file, fileName, "rb");
	if (file == nullptr)
	{
		printf("Error: could not open file \"%s\"\n", fileName);
		return CL_INVALID_VALUE;
	}

	fseek(file, 0, SEEK_END);
	*sourceSize = ftell(file);
	fseek(file, 0, SEEK_SET);

	*source = new char[*sourceSize];
	if (*source == nullptr)
	{
		printf("Error: could not allocate memory\n");
		return CL_OUT_OF_HOST_MEMORY;
	}

	fread(*source, 1, *sourceSize, file);
	return CL_SUCCESS;
}

cl_program CreateAndBuildProgram(const char* fileName, cl_context& context, cl_device_id& deviceId)
{
	cl_int err;

	// Читаем исходный код ядра из файла
	char* source = nullptr;
	size_t src_size = 0;
	OPENCL_CHECK(ReadSourceFromFile(fileName, &source, &src_size));

	// And now after you obtained a regular C string call clCreateProgramWithSource to create OpenCL program object.
	cl_program program = clCreateProgramWithSource(context, 1, (const char**)&source, &src_size, &err);
	if (CL_SUCCESS != err)
	{
		printf("Error: clCreateProgramWithSource returned %d.\n", err);
		exit(EXIT_FAILURE);
	}

	// Построение программы
	err = clBuildProgram(program, 1, &deviceId, "", nullptr, nullptr);
	if (CL_SUCCESS != err)
	{
		printf("Error: clBuildProgram() for source program returned %d.\n", err);

		// В случае ошибки выводим лог сборки
		if (err == CL_BUILD_PROGRAM_FAILURE)
		{
			size_t log_size = 0;
			clGetProgramBuildInfo(program, deviceId, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);

			std::vector<char> build_log(log_size);
			clGetProgramBuildInfo(program, deviceId, CL_PROGRAM_BUILD_LOG, log_size, &build_log[0], nullptr);

			printf("Error happened during the build of OpenCL program.\nBuild log:%s", &build_log[0]);
			exit(EXIT_FAILURE);
		}
	}

	if (source)
		delete[] source;

	return program;
}

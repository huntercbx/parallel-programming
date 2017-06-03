#pragma once

#include <cstdlib>
#include <CL\cl.h>

// Макрос для проверки на ошибку, при вызове функций OpenCL
#define OPENCL_CHECK(call)                                                \
{                                                                         \
    cl_int err = call;                                                    \
    if (err != CL_SUCCESS)                                                \
    {                                                                     \
        printf( "Error calling \"%s\" (error code = %d) \n", #call, err); \
        exit(EXIT_FAILURE);                                               \
    }                                                                     \
}

// Чтение исходного кода ядра OpenCL из файла
int ReadSourceFromFile(const char* fileName, char** source, size_t* sourceSize);

// Создание и построение программы из исходника
cl_program CreateAndBuildProgram(const char* fileName, cl_context& context, cl_device_id& deviceId);


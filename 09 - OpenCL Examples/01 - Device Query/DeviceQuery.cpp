#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <CL/cl.h>

// Макрос для проверки на ошибку, при вызове функций OPENCL
#define OPENCL_CHECK(call)                                                \
{                                                                         \
    cl_int err = call;                                                    \
    if (err != CL_SUCCESS)                                                \
    {                                                                     \
        printf( "Error calling \"%s\" (error code = %d) \n", #call, err); \
        exit(-1);                                                         \
    }                                                                     \
}

std::string GetPlatformInfo(cl_platform_id platform, cl_platform_info param_name)
{
	size_t size;
	OPENCL_CHECK(clGetPlatformInfo(platform, param_name, 0, nullptr, &size));

	char *data = new char[size];
	OPENCL_CHECK(clGetPlatformInfo(platform, param_name, size, data, nullptr));

	std::string str(data);
	delete[] data;
	return str;
}

template <typename T>
T GetDeviceInfo(cl_device_id device, cl_device_info param_name)
{
	T result;
	OPENCL_CHECK(clGetDeviceInfo(device, param_name, sizeof(T), &result, nullptr));
	return result;
}

template <>
std::string GetDeviceInfo(cl_device_id device, cl_device_info param_name)
{
	size_t size;
	OPENCL_CHECK(clGetDeviceInfo(device, param_name, 0, nullptr, &size));

	char *data = new char[size];
	OPENCL_CHECK(clGetDeviceInfo(device, param_name, size, data, nullptr));

	std::string str(data);
	delete[] data;
	return str;
}

template <typename T>
std::vector<T> GetDeviceVectorInfo(cl_device_id device, cl_device_info param_name)
{
	size_t size;
	OPENCL_CHECK(clGetDeviceInfo(device, param_name, 0, nullptr, &size));

	std::vector<T> v(size);
	OPENCL_CHECK(clGetDeviceInfo(device, param_name, size, v.data(), nullptr));

	return v;
}


int main(int argc, char* argv[])
{
	// get platform list
	cl_uint num_platforms;
	OPENCL_CHECK(clGetPlatformIDs(1, NULL, &num_platforms));
	if (num_platforms == 0)
	{
		printf("There are no compatible platforms found\n");
		exit(-1);
	}
	cl_platform_id *platforms = new cl_platform_id[num_platforms];
	OPENCL_CHECK(clGetPlatformIDs(num_platforms, platforms, nullptr));

	for (cl_uint i = 0; i<num_platforms; ++i)
	{
		std::string str;
		printf("Platform %d information:\n", i);

		str = GetPlatformInfo(platforms[i], CL_PLATFORM_NAME);
		printf("   CL_PLATFORM_NAME       = %s\n", str.c_str());

		str = GetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR);
		printf("   CL_PLATFORM_VENDOR     = %s\n", str.c_str());

		str = GetPlatformInfo(platforms[i], CL_PLATFORM_VERSION);
		printf("   CL_PLATFORM_VERSION    = %s\n", str.c_str());

		str = GetPlatformInfo(platforms[i], CL_PLATFORM_PROFILE);
		printf("   CL_PLATFORM_PROFILE    = %s\n", str.c_str());

		str = GetPlatformInfo(platforms[i], CL_PLATFORM_EXTENSIONS);
		printf("   CL_PLATFORM_EXTENSIONS = %s\n", str.c_str());

		// get device list
		cl_uint num_devices;
		OPENCL_CHECK(clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, nullptr, &num_devices));
		cl_device_id *devices = new cl_device_id[num_devices];
		OPENCL_CHECK(clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, num_devices, devices, nullptr));

		for (cl_uint j = 0; j < num_devices; ++j)
		{
			printf("   Device %d information:\n", j);

			str = GetDeviceInfo<std::string>(devices[j], CL_DEVICE_NAME);
			printf("      CL_DEVICE_NAME                       = %s\n", str.c_str());

			str = GetDeviceInfo<std::string>(devices[j], CL_DEVICE_VENDOR);
			printf("      CL_DEVICE_VENDOR                     = %s\n", str.c_str());

			str = GetDeviceInfo<std::string>(devices[j], CL_DRIVER_VERSION);
			printf("      CL_DRIVER_VERSION                    = %s\n", str.c_str());

			auto global_mem_size = GetDeviceInfo<cl_ulong>(devices[j], CL_DEVICE_GLOBAL_MEM_SIZE);
			printf("      CL_DEVICE_GLOBAL_MEM_SIZE            = %I64d MB\n", global_mem_size / 1024 / 1024);

			auto local_mem_size = GetDeviceInfo<cl_ulong>(devices[j], CL_DEVICE_LOCAL_MEM_SIZE);
			printf("      CL_DEVICE_LOCAL_MEM_SIZE             = %I64d KB\n", local_mem_size / 1024);

			auto compute_units = GetDeviceInfo<cl_int>(devices[j], CL_DEVICE_MAX_COMPUTE_UNITS);
			printf("      CL_DEVICE_MAX_COMPUTE_UNITS          = %d\n", compute_units);

			auto max_work_group_size = GetDeviceInfo<size_t>(devices[j], CL_DEVICE_MAX_WORK_GROUP_SIZE);
			printf("      CL_DEVICE_MAX_WORK_GROUP_SIZE        = %d\n", max_work_group_size);

			auto max_work_dimensions = GetDeviceInfo<cl_uint>(devices[j], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS);
			auto max_work_sizes = GetDeviceVectorInfo<size_t>(devices[j], CL_DEVICE_MAX_WORK_ITEM_SIZES);
			printf("      CL_DEVICE_MAX_WORK_ITEM_SIZES        = ");
			for (size_t k = 0, ke = max_work_dimensions; k < ke; ++k)
				printf("%s%d", k == 0 ? "" : " x ", max_work_sizes[k]);
			printf("\n");

			auto timerResolution = GetDeviceInfo<size_t>(devices[j], CL_DEVICE_PROFILING_TIMER_RESOLUTION);
			printf("      CL_DEVICE_PROFILING_TIMER_RESOLUTION = %d ns\n", timerResolution);

		}

		delete[] devices;
		printf("\n");
	}

	delete[] platforms;
	return 0;
}

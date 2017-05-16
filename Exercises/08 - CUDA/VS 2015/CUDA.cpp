#include <stdio.h>

// заголовочные файлы CUDA
#include <cuda.h>
#include <cuda_runtime.h>

// заголовочный файл для работы с PPM изображениями
#include "PPMImage.h"

// Макрос для проверки на ошибку, при вызове функций CUDA
#define CUDA_CHECK(call) \
	if((call) != cudaSuccess) { \
		cudaError_t err = cudaGetLastError(); \
		printf( "CUDA error calling \"%s\", error code is %d\n", #call, err); \
		printf( "Error %d: %s\n", err, cudaGetErrorString(err)); \
		exit(-1); }


extern "C"
void Filter(unsigned char * src, unsigned char * dest, unsigned int w, unsigned int h);

int main(int argc, char* argv[])
{
	unsigned char *dev_origin, *dev_result;

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

	cudaEvent_t evt1, evt2, evt3, evt4, evt5, evt6;
	cudaEventCreate(&evt1);
	cudaEventCreate(&evt2);
	cudaEventCreate(&evt3);
	cudaEventCreate(&evt4);
	cudaEventCreate(&evt5);
	cudaEventCreate(&evt6);

	// выделение видеопамяти
	cudaEventRecord(evt1, 0);
	size_t img_size = original_image.get_required_mem_size();
	CUDA_CHECK(cudaMalloc((void**)&dev_origin, img_size));
	CUDA_CHECK(cudaMalloc((void**)&dev_result, img_size));

	// копирование исходного изображения в видеопамять
	cudaEventRecord(evt2, 0);
	CUDA_CHECK(cudaMemcpy(dev_origin, original_image.image, img_size, cudaMemcpyHostToDevice));

	// обработка изображения
	cudaEventRecord(evt3, 0);
	Filter(dev_origin, dev_result, original_image.width, original_image.height);

	// копирование обработанного изображения из видеопамяти
	cudaEventRecord(evt4, 0);
	CUDA_CHECK(cudaMemcpy(result_image.image, dev_result, img_size, cudaMemcpyDeviceToHost));

	// освобождаем видеопамять
	cudaEventRecord(evt5, 0);
	CUDA_CHECK(cudaFree(dev_origin));
	CUDA_CHECK(cudaFree(dev_result));

	// окончание работы
	cudaEventRecord(evt6, 0);
	cudaEventSynchronize(evt6);

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

	// сохраняем обработанное изображение в файл
	result_image.save(argv[2]);

	return 0;
}

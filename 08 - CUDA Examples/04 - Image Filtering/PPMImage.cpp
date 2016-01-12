#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "PPMImage.h"

// Конструктор
PPMImage::PPMImage() : width(0), height(0), max_color_val(0), image(0)
{}

PPMImage::PPMImage(const char * filename) : image(0)
{
	load(filename);
}

// Деструктор
PPMImage::~PPMImage()
{
	// освобождаем память
	if (image) free(image);
}

// Возвращает размер памяти в байтах, необходимой для хранения растра
unsigned int PPMImage::get_required_mem_size()
{
	return 3 * width * height * (max_color_val < 256 ? 1 : 2);
}

// Выделение памяти под изображение
void PPMImage::allocate_memory()
{
	if (image)
		free(image);

	image = (unsigned char*)malloc(get_required_mem_size());
}

// Загрузка из файла
bool PPMImage::load(const char * filename)
{
	FILE *f = fopen(filename, "rb");
	if (f == 0)
	{
		printf("File '%s' not found\n", filename);
		return false;
	}

	char type[80];
	fscanf(f, "%s", type);
	if (strcmp(type, "P6"))
	{
		printf("Invalid PPM file\n");
		fclose(f);
		return false;
	}

	fscanf(f, "%u", &width);
	fscanf(f, "%u", &height);
	fscanf(f, "%u\n", &max_color_val);
	if (max_color_val == 0 || max_color_val > 255)
	{
		printf("Unsupported color depth: %d\n", max_color_val);
		fclose(f);
		return false;
	}

	allocate_memory();
	fread(image, 1, get_required_mem_size(), f);

	fclose(f);
	return true;
}

// Сохранение в файл
bool PPMImage::save(const char * filename)
{
	FILE *f = fopen(filename, "wb");
	if (f == 0)
	{
		printf("Can not save file %s\n", filename);
		return false;
	}

	fprintf(f, "P6\n");
	fprintf(f, "%u %u %u\n", width, height, max_color_val);

	fwrite(image, 1, get_required_mem_size(), f);

	fclose(f);
	return true;
}

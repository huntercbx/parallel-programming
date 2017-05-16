#pragma once

// Структура для работы с PPM (Portable PixelMap) изображениями
struct PPMImage
{
	unsigned int    width;          // ширина изображения
	unsigned int    height;         // высота изображения
	unsigned int    max_color_val;  // максимальное значение для цвета
	unsigned char*  image;          // растр изображения

	// Конструктор
	PPMImage();
	PPMImage(const char * filename);
	
	// Деструктор
	~PPMImage();

	// Возвращает размер памяти в байтах, необходимой для хранения растра
	unsigned int get_required_mem_size();

	// Выделение памяти под изображение
	void allocate_memory();

	// Загрузка из файла
	bool load(const char * filename);

	// Сохранение в файл
	bool save(const char * filename);
};

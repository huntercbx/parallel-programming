////////////////////////////////////////////////////////////////////////////////
// Данная программа показывает пример преобразования исходной картинки в
//    чёрно-белую с использованием двухмерного итерационного пространства
//    библиотеки TBB.
//
// Результом работы является чёрно-белое изображение.
//
// Программа принимает на вход два аргумента:
//      - имя исходного изображения (по умолчанию "image.jpg")
//      - количество потоков (по умолчанию автоматически)
////////////////////////////////////////////////////////////////////////////////

#include <iostream>

#include "tbb/tick_count.h"
#include "tbb/tbb_thread.h"
#include "tbb/task_scheduler_init.h"
#include "tbb/blocked_range2d.h"
#include "tbb/parallel_for.h"

#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace tbb;
using namespace cv;

const string OUTPUT_IMAGE_NAME_1 = "output_1.jpg";	// имя выходного изображения 1
const string OUTPUT_IMAGE_NAME_2 = "output_2.jpg";	// имя выходного изображения 2
const size_t GRAIN_SIZE = 1000;						// размер "порции" вычислений

////////////////////////////////////////////////////////////////////////////////
// Функциональный класс для преобразования изображения в оттенки серого
////////////////////////////////////////////////////////////////////////////////
class GrayConverterSerial
{
public:
	GrayConverterSerial(Mat& img) : m_image(img) {}

	void operator()()
	{
		for (int i = 0; i < m_image.rows; ++i)
		{
			for (int j = 0; j < m_image.cols; ++j)
			{
				Vec3b color = m_image.at<Vec3b>(i, j);
				float luma = 0;
				luma += 0.2126f * color[0];
				luma += 0.7152f * color[1];
				luma += 0.0722f * color[2];

				color[0] = color[1] = color[2] = (unsigned char)luma;
				m_image.at<Vec3b>(i, j) = color;
			}
		}
	}

private:
	Mat& m_image;
};

////////////////////////////////////////////////////////////////////////////////
// Функциональный класс для преобразования изображения в оттенки серого
////////////////////////////////////////////////////////////////////////////////
class GrayConverterParallel
{
public:
	GrayConverterParallel(Mat& img) : m_image(img) {}

    void operator()(const blocked_range2d<int, int>& range) const
    {
		#ifdef _DEBUG
			stringstream ss;
			ss << "Thread " << this_tbb_thread::get_id()
				<< " [" << range.rows().begin() << ", " << range.rows().end() << "]"
				<< " [" << range.cols().begin() << ", " << range.cols().end() << "]" << endl;
			cout << ss.str();
		#endif

        for (int i = range.rows().begin(); i != range.rows().end(); ++i)
        {
            for (int j = range.cols().begin(); j < range.cols().end(); ++j)
            {
				Vec3b color = m_image.at<Vec3b>(i, j);
				float luma = 0;
				luma += 0.2126f * color[0];
				luma += 0.7152f * color[1];
				luma += 0.0722f * color[2];

                color[0] = color[1] = color[2] = (unsigned char)luma;
                m_image.at<Vec3b>(i, j) = color;
            }
        }
    }

private:
    Mat& m_image;
};

////////////////////////////////////////////////////////////////////////////////
// Основная программа
////////////////////////////////////////////////////////////////////////////////
int main(int argc, const char* argv[])
{
	// инициализация библиотеки TBB
	task_scheduler_init init;

	// проверка аргументов комадной строки
	if (argc < 2)
	{
		cout << "Usage: \"" << argv[0] << "\" source" << endl;
		return -1;
	}

	// открытие исходного изображения
	cout << "Input image name: " << argv[1] << endl;
	Mat image = imread(argv[1]);

	// проверка: загружено ли изображение
	if (image.data == nullptr)
	{
		cout << "Could not load image." << endl;
		return -1;
	}

	// параметры выходного изображения
	vector<int> imageParams = { CV_IMWRITE_JPEG_QUALITY, 100 };

	// копируем изображение для обработки
	auto image1 = image.clone();
	auto image2 = image.clone();

	// применение последовательного алгоритма
	cout << "Launching serial version" << endl;
	tick_count t0 = tick_count::now();
	{
		GrayConverterSerial converterSerial(image1);
		converterSerial();
	}
	tick_count t1 = tick_count::now();
	imwrite(OUTPUT_IMAGE_NAME_1, image1, imageParams);

	// применение алгоритма parallel_reduce
	cout << "Launching parallel version" << endl;
	tick_count t2 = tick_count::now();
	{
		GrayConverterParallel converterParallel(image2);
		blocked_range2d<int, int> range(0, image.rows, GRAIN_SIZE, 0, image.cols, GRAIN_SIZE);
		parallel_for(range, converterParallel);
	}
	tick_count t3 = tick_count::now();
	imwrite(OUTPUT_IMAGE_NAME_2, image2, imageParams);

	double speedup = (t1 - t0).seconds() / (t3 - t2).seconds();
	cout << "Execution time (serial)   = " << (t1 - t0).seconds() << " seconds" << endl;
	cout << "Execution time (parallel) = " << (t3 - t2).seconds() << " seconds" << endl;
	cout << "Speedup                   = " << speedup << endl;

    return 0;
}

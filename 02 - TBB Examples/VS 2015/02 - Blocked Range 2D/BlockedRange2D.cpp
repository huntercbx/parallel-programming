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

#include <stdio.h>

#include <tbb/tick_count.h>
#include <tbb/tbb_thread.h>
#include <tbb/task_scheduler_init.h>
#include <tbb/blocked_range2d.h>
#include <tbb/parallel_for.h>

#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace tbb;
using namespace cv;

const string INPUT_IMAGE_NAME = "image.jpg";    // имя входного изображения
const size_t GRAIN_SIZE = 1505;                 // размер "порции" вычислений

////////////////////////////////////////////////////////////////////////////////
// Функциональный класс для преобразования изображения в оттенки серого
////////////////////////////////////////////////////////////////////////////////
class GrayConverter
{
public:
	GrayConverter(Mat& img) : m_image(img) {}

	void operator()(const blocked_range2d<int, int>& range) const
	{
		printf("thread_id = %d; rows = [%d; %d); cols = [%d; %d)\n",
			this_tbb_thread::get_id(),
			range.rows().begin(),
			range.rows().end(),
			range.cols().begin(),
			range.cols().end());

		for (int i = range.rows().begin(); i != range.rows().end(); ++i)
		{
			for (int j = range.cols().begin(); j < range.cols().end(); ++j)
			{
				Vec3b color = m_image.at<Vec3b>(i, j);
				uchar gray = (uchar)(0.11f * color[0] + 0.59f * color[1] + 0.3f * color[2]);

				color[0] = color[1] = color[2] = gray;
				m_image.at<Vec3b>(i, j) = color;
			}
		}
	}

private:
	Mat & m_image;
};

////////////////////////////////////////////////////////////////////////////////
// Основная программа
////////////////////////////////////////////////////////////////////////////////
int main(int argc, const char* argv[])
{
	// инициализация библиотеки TBB
	int threads_count = (argc >= 3) ? atoi(argv[2]) : task_scheduler_init::automatic;
	printf("Threads count: %i.\n", threads_count);
	task_scheduler_init init(threads_count);

	// названия входного и выходного изображений
	String in_image_name = (argc >= 2) ? argv[1] : INPUT_IMAGE_NAME;
	String out_image_name = "gray_" + in_image_name;

	// открытие исходного изображения
	printf("Input image name: \"%s\".\n", in_image_name.c_str());
	Mat image = imread(in_image_name);
    
	// проверка: загружено ли изображение
	if (image.data == nullptr) {
		printf("Could not load image.\n");
		return -1;
	}

	// преобразовние изображения в чёрно-белое и измерение времени работы
	tick_count t1 = tick_count::now();
	blocked_range2d<int, int> range(0, image.rows, 0, image.cols);
	parallel_for(range, GrayConverter(image));
	tick_count t2 = tick_count::now();

	// вывести затраченное время
	printf("Elapsed time: %lf.\n", (t2 - t1).seconds());

	// сохранить изображение 
	vector<int> params = { CV_IMWRITE_JPEG_QUALITY, 100 };
	bool success = imwrite(out_image_name, image, params);

	// вывод результатов работы программы
	if (success)
		printf("Output image name: \"%s\".\n", out_image_name.c_str());
	else
	{
		printf("Error while saving image.\n");
		return -1;
	}

	return 0;
}

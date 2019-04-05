////////////////////////////////////////////////////////////////////////////////
// Данная программа иллюстрирует построение конвейера с использованием TBB
//    на примере обработки видеофайла.
//
// Конвейер состоит из 4-х этапов:
//    1. Чтение исходного кадра из потока
//    2. Применение первого фильтра (перевод в оттенки серого)
//    3. Применение второго фильтра (фильтр Собеля для обнаружения границ)
//    4. Запись обработаного кадра в поток
//
// Результом работы является видеофайл.
//
// Программа принимает на вход два аргумента:
//      - имя исходного видеофайла
//      - имя выходного видеофайла (необязательный параметр)
////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <sstream>
#include <fstream>

#include <tbb/pipeline.h>
#include <tbb/tick_count.h>
#include <tbb/task_scheduler_init.h>
#include <tbb/concurrent_vector.h>
#include <tbb/compat/thread>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;

// время запуска основной програмы
tbb::tick_count t0;

////////////////////////////////////////////////////////////////////////////////
// Матрица свертки фильтра Собеля по X
////////////////////////////////////////////////////////////////////////////////
const auto SOBEL_KERNEL_X = (Mat_<float>(3, 3) <<
	 -3.0f, 0.0f,  3.0f,
	-10.0f, 0.0f, 10.0f,
	 -3.0f, 0.0f,  3.0f
	) / (3.0f * 3.0f);

////////////////////////////////////////////////////////////////////////////////
// Матрица свертки фильтра Собеля по Y
////////////////////////////////////////////////////////////////////////////////
const auto SOBEL_KERNEL_Y = (Mat_<float>(3, 3) <<
	-3.0f, -10.0f, -3.0f,
	 0.0f,   0.0f,  0.0f,
	 3.0f,  10.0f,  3.0f
	) / (3.0f * 3.0f);

////////////////////////////////////////////////////////////////////////////////
// Журнал работы программы, по окончании работы записывается в файл журнала.
// Запись в файл во время работы основоного алгоритма не производится в целях
// производительности.
////////////////////////////////////////////////////////////////////////////////
tbb::concurrent_vector<string> program_log;

////////////////////////////////////////////////////////////////////////////////
// Запись события в журнал
////////////////////////////////////////////////////////////////////////////////
void WriteLog(const string& message)
{
	std::ostringstream s;
	s << (tbb::tick_count::now() - t0).seconds() << ", "
		<< this_thread::get_id() << ", "
		<< message;

	program_log.push_back(s.str());
}

////////////////////////////////////////////////////////////////////////////////
// Запись журнала в файл
////////////////////////////////////////////////////////////////////////////////
void SaveLogToFile(const string& filename)
{
	ofstream file(filename);
	if (!file.is_open())
	{
		cout << "Can not write to file \"" << filename << "\"" << endl;
		exit(EXIT_FAILURE);
	}
	cout << "Saving log to the file \"" << filename << "\" " << endl;

	for (const auto& log_record : program_log)
		file << log_record << endl;

	file.close();
}

////////////////////////////////////////////////////////////////////////////////
// Этап 1 - чтение исходного кадра из видеопотока
////////////////////////////////////////////////////////////////////////////////
struct ReadFrameFunc
{
	// конструктор
	ReadFrameFunc(VideoCapture& source) : m_source(source)
	{}

	Mat operator()(tbb::flow_control& fc) const
	{
		WriteLog("ReadFrameFunc, begin");

		Mat frame;
		m_source >> frame;

		// если достигнут конец файла, останавливаем работу конвейера
		if (frame.empty())
			fc.stop();

		WriteLog("ReadFrameFunc, end");

		return frame;
	}

private:
	VideoCapture& m_source; // источник видео
};

////////////////////////////////////////////////////////////////////////////////
// Этап 2 - первый фильтр (перевод в оттенки серого)
////////////////////////////////////////////////////////////////////////////////
struct Filter1
{
	Mat operator()(Mat frame) const
	{
		WriteLog("Filter1, begin");

		Mat_<Vec3b> frame_ = frame;
		for (int i = 0; i < frame.rows; ++i)
		{
			for (int j = 0; j < frame.cols; ++j)
			{
				float luma = 0;
				luma += 0.2126f * frame_(i, j)[0];
				luma += 0.7152f * frame_(i, j)[1];
				luma += 0.0722f * frame_(i, j)[2];

				unsigned char l = (unsigned char)luma;

				frame_(i, j)[0] = l;
				frame_(i, j)[1] = l;
				frame_(i, j)[2] = l;
			}
		}

		WriteLog("Filter1, end");

		return frame;
	}
};

////////////////////////////////////////////////////////////////////////////////
// Этап 3 - второй фильтр (фильтр Собеля для обнаружения границ)
////////////////////////////////////////////////////////////////////////////////
struct Filter2
{
	Mat operator()(Mat frame) const
	{
		WriteLog("Filter2, begin");

		Mat g_x, g_y;
		filter2D(frame, g_x, -1, SOBEL_KERNEL_X);
		filter2D(frame, g_y, -1, SOBEL_KERNEL_Y);

		Mat_<Vec3b> frame_ = frame;
		Mat_<Vec3b> g_x_ = g_x;
		Mat_<Vec3b> g_y_ = g_y;
		for (int i = 0; i < frame.rows; ++i)
		{
			for (int j = 0; j < frame.cols; ++j)
			{
				frame_(i, j)[0] = static_cast<uchar>(sqrt(static_cast<float>(g_x_(i, j)[0] * g_x_(i, j)[0] + g_y_(i, j)[0] * g_y_(i, j)[0])));
				frame_(i, j)[1] = static_cast<uchar>(sqrt(static_cast<float>(g_x_(i, j)[1] * g_x_(i, j)[1] + g_y_(i, j)[1] * g_y_(i, j)[1])));
				frame_(i, j)[2] = static_cast<uchar>(sqrt(static_cast<float>(g_x_(i, j)[2] * g_x_(i, j)[2] + g_y_(i, j)[2] * g_y_(i, j)[2])));
			}
		}

		WriteLog("Filter2, end");

		return frame;
	}
};

////////////////////////////////////////////////////////////////////////////////
// Этап 4 - запись обработаного кадра в видеопоток
////////////////////////////////////////////////////////////////////////////////
struct WriteFrameFunc
{
	// конструктор
	WriteFrameFunc(VideoWriter& destination) : m_destination(destination)
	{}

	void operator()(Mat frame) const
	{
		WriteLog("WriteFrameFunc, begin");

		m_destination << frame;

		WriteLog("WriteFrameFunc, end");
	}

private:
	VideoWriter& m_destination; // приемник видео (выходной поток)
};

////////////////////////////////////////////////////////////////////////////////
// Основная программа
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{
	// время запуска основной програмы
	t0 = tbb::tick_count::now();

	// инициализация планировщика задач TBB
	tbb::task_scheduler_init init;

	// проверка аргументов комадной строки
	if (argc < 2)
	{
		cout << "Usage: \"" << argv[0] << "\" source [destination]" << endl;
		cout << "   source      - source videofile name, supported formats: avi, mp4, mkv" << endl;
		cout << "   destination - resulting videofile name (optional)" << endl;
		cout << "If destination is not specified then output file will be named \"output\"" << endl;
		cout << "   and it will have the same file extension as source videofile" << endl;
		return -1;
	}

	// открытие входного файла
	VideoCapture sourceVideo(argv[1]);
	if (!sourceVideo.isOpened())
	{
		cout << "Could not read \"" << argv[1] << "\"" << endl;
		return -1;
	}

	// получение четырехсимвольного тип кодека, размера кадра и частоты кадров в секунду
	int fourcc = static_cast<int>(sourceVideo.get(CV_CAP_PROP_FOURCC));
	int frame_width = static_cast<int>(sourceVideo.get(CV_CAP_PROP_FRAME_WIDTH));
	int frame_height = static_cast<int>(sourceVideo.get(CV_CAP_PROP_FRAME_HEIGHT));
	double fps = sourceVideo.get(CV_CAP_PROP_FPS);

	// создание выходного потока
	string destination;
	if (argc >= 3)
		destination = argv[2];
	else
	{
		destination = "output";
		string source_name(argv[1]);
		auto pos = source_name.find(".");
		if (pos != string::npos)
			destination += source_name.substr(pos, source_name.size());
	}
	VideoWriter outputVideo;
	outputVideo.open(destination, fourcc, fps, Size(frame_width, frame_height), true);
	if (!outputVideo.isOpened())
	{
		cout << "Could not write to \"" << destination << "\"" << endl;
		return -1;
	}

	// выводим сообщение о начале обработки видео
	cout << "Start video processing ("
		<< frame_width << "x" << frame_height << "px, "
		<< fps << " fps)" << endl;

	// создание отдельных этапов конвейера и их объединение
	tbb::filter_t<void, Mat> f1(tbb::filter::serial_in_order, ReadFrameFunc(sourceVideo));
	tbb::filter_t<Mat, Mat>  f2(tbb::filter::parallel, Filter1());
	tbb::filter_t<Mat, Mat>  f3(tbb::filter::parallel, Filter2());
	tbb::filter_t<Mat, void> f4(tbb::filter::serial_in_order, WriteFrameFunc(outputVideo));
	tbb::filter_t<void, void> f = f1 & f2 & f3 & f4;

	// запуск конвейера и вывод затраченного времени
	tbb::tick_count t1 = tbb::tick_count::now();
	tbb::parallel_pipeline(16, f);
	tbb::tick_count t2 = tbb::tick_count::now();
	cout << "Elapsed time = " << (t2 - t1).seconds() << " seconds" << endl;

	// запись журнала в файл
	string log_name(argv[0]);
	auto pos = log_name.find(".");
	if (pos != string::npos)
		log_name = log_name.substr(0, pos);
	SaveLogToFile(log_name + ".log");

	return 0;
}

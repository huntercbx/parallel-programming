// 08 - Unsharp Mask.cpp: определяет точку входа для консольного приложения.
//

#include <iostream>
#include <sstream>
#include <fstream>

#include <tbb/pipeline.h>
#include <tbb/concurrent_vector.h>

#include <tbb/tick_count.h>
#include <tbb/compat/thread>
#include <tbb/flow_graph.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;
using namespace tbb;

////////////////////////////////////////////////////////////////////////////////
// Чтение исходного кадра из видеопотока
////////////////////////////////////////////////////////////////////////////////
struct ReadFrame
{
	// конструктор
	ReadFrame(VideoCapture& source) : m_source(source)
	{}

	bool operator()(Mat& frame) const
	{
		m_source >> frame;

		// если достигнут конец файла, останавливаем работу конвейера
		if (frame.empty())
			return false;

		return true;
	}

private:
	VideoCapture & m_source; // источник видео
};

////////////////////////////////////////////////////////////////////////////////
// Запись обработаного кадра в видеопоток
////////////////////////////////////////////////////////////////////////////////
struct WriteFrame
{
	// конструктор
	WriteFrame(VideoWriter& destination) : m_destination(destination)
	{}

	int operator()(Mat frame) const
	{
		m_destination << frame;
		return 0;
	}

private:
	VideoWriter & m_destination; // приемник видео (выходной поток)
};

////////////////////////////////////////////////////////////////////////////////
// Размытие по Гауссу
////////////////////////////////////////////////////////////////////////////////
struct Gaussian
{
	Mat operator()(Mat frame) const
	{
		Mat gaussian;
		int ksize = 3;
		double sigma = 0.3 *((ksize - 1) *0.5 - 1) + 0.8;
		GaussianBlur(frame, gaussian, Size(ksize, ksize), sigma);
		return gaussian;
	}
};

////////////////////////////////////////////////////////////////////////////////
// Разница между двумя изображениями
////////////////////////////////////////////////////////////////////////////////
struct Difference
{
	Mat operator()(tuple < Mat, Mat> v) const
	{
		auto starting = get<0>(v);
		auto gaussian = get<1>(v);
		Mat result = starting - gaussian;
		return result;
	}
};

////////////////////////////////////////////////////////////////////////////////
// Сумма двух изображений
////////////////////////////////////////////////////////////////////////////////
struct Sum
{
	Mat operator()(tuple < Mat, Mat> v) const
	{
		auto x1 = get<0>(v);
		auto x2 = get<1>(v);
		Mat result = x1 + x2;
		return result;
	}
};

////////////////////////////////////////////////////////////////////////////////
// Умножение изображения на коэффициент
////////////////////////////////////////////////////////////////////////////////
struct Scale
{
	Scale(int k) : coef(k) {}

	Mat operator()(Mat frame) const
	{
		frame = frame * coef;
		return frame;
	}
private:
	int coef;
};

int main(int argc, char *argv[]) {

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
	tbb::tick_count t1 = tbb::tick_count::now();

	flow::graph g;

	flow::source_node<Mat> read(g, ReadFrame(sourceVideo), false);
	flow::broadcast_node<Mat> broadcast(g);
	flow::function_node<Mat, Mat> gaussian(g, flow::serial, Gaussian());
	flow::function_node<tuple<Mat, Mat>, Mat> sum(g, flow::unlimited, Sum());
	flow::function_node<Mat, int> write(g, flow::serial, WriteFrame(outputVideo));
	flow::function_node<tuple<Mat, Mat>, Mat> difference(g, flow::unlimited, Difference());
	flow::function_node<Mat, Mat> scale(g, flow::unlimited, Scale(5));
	flow::join_node< flow::tuple<Mat, Mat>, flow::queueing> join_gausse_start(g);
	flow::join_node< flow::tuple<Mat, Mat>, flow::queueing> join_mult_start(g);

	make_edge(read, broadcast);
	make_edge(broadcast, get< 0 >(join_gausse_start.input_ports()));
	make_edge(broadcast, gaussian);
	make_edge(broadcast, get<0>(join_mult_start.input_ports()));
	make_edge(gaussian, get<1>(join_gausse_start.input_ports()));
	make_edge(join_gausse_start, difference);
	make_edge(difference, scale);
	make_edge(scale, get< 1 >(join_mult_start.input_ports()));
	make_edge(join_mult_start, sum);
	make_edge(sum, write);

	// активиация источника и ожидание окончания обработки
	read.activate();
	g.wait_for_all();

	tbb::tick_count t2 = tbb::tick_count::now();
	cout << "Elapsed time = " << (t2 - t1).seconds() << " seconds" << endl;
	return 0;
}

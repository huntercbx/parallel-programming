////////////////////////////////////////////////////////////////////////////////
// Демонстрация использования параллельного алгоритма parallel_reduce
// для нахождения индекса минимального элемента массива
//
// С данной программой в отладочной конфигурации можно провести эксперименты:
// 1. Задать значение третьего параметра grainsize в blocked_range<size_t>(0, N)
//     и посмотреть как это повлияет на разбивку исходного диапазона
////////////////////////////////////////////////////////////////////////////////
#include <climits>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <vector>

#include <tbb/task_scheduler_init.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>
#include <tbb/tick_count.h>
#include <tbb/compat/thread>

using namespace std;
using namespace tbb;

////////////////////////////////////////////////////////////////////////////////
// Базовый функциональный объект для поиска индекса минимального элемента массива
////////////////////////////////////////////////////////////////////////////////
class MinIndexFinderBase
{
public:
	// конструткор
	MinIndexFinderBase(const vector<float>& v) :
		vec(v),
		min_el_val(v[0]),
		min_el_idx(0)
	{}

	// вывод результатов работы алгоритма
	void PrintResult() const
	{
		cout << "Index=" << min_el_idx << "; "
		     << "Value=" << min_el_val << endl;
	}

protected:
	const vector<float>&    vec;        // ссылка на исходный массив
	size_t                  min_el_idx; // индекс минимального элемента массива
	float                   min_el_val; // значение минимального элемента массива
};

////////////////////////////////////////////////////////////////////////////////
// Функциональный объект для поиска индекса минимального элемента массива
//  (последовательная версия)
////////////////////////////////////////////////////////////////////////////////
class MinIndexFinderSerial : public MinIndexFinderBase
{
public:
	MinIndexFinderSerial(const vector<float>& v) : MinIndexFinderBase(v)
	{}

	void operator()()
	{
		for(size_t i = 0; i < vec.size(); ++i)
		{
			if (vec[i] < min_el_val)
			{
				min_el_val = vec[i];
				min_el_idx = i;
			}
		}
	}
};

////////////////////////////////////////////////////////////////////////////////
// Функциональный объект для поиска индекса минимального элемента массива
//  (параллельная версия)
////////////////////////////////////////////////////////////////////////////////
class MinIndexFinderParallel : public MinIndexFinderBase
{
public:
	// конструтор
	MinIndexFinderParallel(const vector<float>& v) : MinIndexFinderBase(v)
	{}

	// расщепляющий конструтор (splitting constructor)
	MinIndexFinderParallel(MinIndexFinderParallel& other, split) :
		MinIndexFinderBase(other.vec)
	{
		min_el_val = other.min_el_val;
		min_el_idx = other.min_el_idx;
	}

	void operator()(const blocked_range<size_t>& r)
	{
		#ifdef _DEBUG
			// вывод идентификатора потока и диапазона
			stringstream ss;
			ss << "Thread " << this_thread::get_id() << " [" << r.begin() << ", " << r.end() << "]" << endl;
			cout << ss.str();
		#endif

		for(size_t i = r.begin(); i != r.end(); ++i)
		{
			if (vec[i] < min_el_val)
			{
				min_el_val = vec[i];
				min_el_idx = i;
			}
		}
	}

	// опреация редукции (reduce operation)
	void join(const MinIndexFinderParallel& other)
	{
		if (other.min_el_val < min_el_val)
		{
			min_el_val = other.min_el_val;
			min_el_idx = other.min_el_idx;
		}
	}
};

////////////////////////////////////////////////////////////////////////////////
// Основная программа
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
	// определение размера решаемой задачи из параметров командной строки
	const size_t N = (argc == 2) ? atol(argv[1]) : 100000000;

	vector<float> v(N);

	// инициализация планировщика задач TBB
	task_scheduler_init init;

	// генерация исходного массива данных
	cout << "Generating data (" << N << " elements)" << endl;
	for (size_t i = 0; i < N; ++i)
		v[i] = rand()/static_cast<float>(RAND_MAX);

	// применение последовательного алгоритма
	cout << "Launching serial version" << endl;
	tick_count t0 = tick_count::now();
	MinIndexFinderSerial mif_serial(v);
	mif_serial();
	tick_count t1 = tick_count::now();
	mif_serial.PrintResult();

	// применение алгоритма parallel_reduce
	cout << "Launching parallel version" << endl;
	tick_count t2 = tick_count::now();
	MinIndexFinderParallel mif_parallel(v);
	parallel_reduce(blocked_range<size_t>(0, N, 1000000), mif_parallel);
	tick_count t3 = tick_count::now();
	mif_parallel.PrintResult();

	double speedup = (t1-t0).seconds()/(t3-t2).seconds();
	cout << "Execution time (serial)   = " << (t1-t0).seconds() << " seconds" << endl;
	cout << "Execution time (parallel) = " << (t3-t2).seconds() << " seconds" << endl;
	cout << "Speedup                   = " << speedup << endl;

	return 0;
}

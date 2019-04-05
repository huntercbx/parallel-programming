////////////////////////////////////////////////////////////////////////////////
// Демонстрация использования параллельного алгоритма parallel_for
// для вычисления значении функции для всех элементов массива
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
#include <tbb/parallel_for.h>
#include <tbb/tick_count.h>
#include <tbb/compat/thread>

using namespace std;
using namespace tbb;

////////////////////////////////////////////////////////////////////////////////
// Функция возведения числа в квадрат
////////////////////////////////////////////////////////////////////////////////
float y(float x)
{
	return sin(x)*cos(x) + cos(2*x);
}

////////////////////////////////////////////////////////////////////////////////
// Основная программа
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
	// определение размера решаемой задачи из параметров командной строки
	const size_t N = (argc == 2) ? atol(argv[1]) : 100000000;
	vector<float> v1(N), v2(N);

	// инициализация планировщика задач TBB
	task_scheduler_init init;

	// генерация исходного массива данных
	cout << "Generating data (" << N << " elements)" << endl;
	for (size_t i = 0; i < N; ++i)
		v1[i] = v2[i] = rand()/static_cast<float>(RAND_MAX);

	// применение последовательного алгоритма
	cout << "Launching serial version" << endl;
	tick_count t0 = tick_count::now();
	for(size_t i = 0; i < v1.size(); ++i)
		v1[i] = y(v1[i]);
	tick_count t1 = tick_count::now();

	// применение алгоритма parallel_for
	cout << "Launching parallel version" << endl;
	tick_count t2 = tick_count::now();
	parallel_for(
		blocked_range<size_t>(0, N, 1000000),
		[&v2] (const blocked_range<size_t>& r)
		{
			#ifdef _DEBUG
				// вывод идентификатора потока и диапазона
				stringstream ss;
				ss << "Thread " << this_thread::get_id() << " [" << r.begin() << ", " << r.end() << "]" << endl;
				cout << ss.str();
			#endif

			for(size_t i = r.begin(); i != r.end(); ++i)
				v2[i] = y(v2[i]);
		}
	);
	tick_count t3 = tick_count::now();

	double speedup = (t1-t0).seconds()/(t3-t2).seconds();
	cout << "Execution time (serial)   = " << (t1-t0).seconds() << " seconds" << endl;
	cout << "Execution time (parallel) = " << (t3-t2).seconds() << " seconds" << endl;
	cout << "Speedup                   = " << speedup << endl;

	return 0;
}

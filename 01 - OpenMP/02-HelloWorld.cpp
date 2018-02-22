////////////////////////////////////////////////////////////////////////////////
/// Программа печатает 'Hello World' из разных потоков.
///
/// При изучении программы следует обратить внимание на следующие моменты:
/// 1. Операции потокового ввода-вывода (оператор <<) не являются
/// потокобезопасными, при их использовании из разных потоков необходимы
/// дополнительные механизмы синхронизации - например, критическая секция
///
/// С программой рекомендуется произвести следующие эксперименты:
/// 1. Закомментировать '#pragma omp critical' и сравнить результаты работы
////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <omp.h>

using namespace std;

int main(int argc, char *argv[])
{
	int nthreads;

	#pragma omp parallel
	{
		int th_id = omp_get_thread_num();
		#pragma omp critical
		{
			cout << "Hello World from thread " << th_id << '\n';
		}

		#pragma omp barrier

		#pragma omp master
		{
			nthreads = omp_get_num_threads();
			cout << "There are " << nthreads << " threads" << '\n';
		}
	}

	return 0;
}

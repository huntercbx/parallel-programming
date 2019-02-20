////////////////////////////////////////////////////////////////////////////////
/// Программа печатает 'Hello World' из разных потоков.
///
/// При изучении программы следует обратить внимание на следующие моменты:
/// 1. Переменная th_id будет у каждого потока своя - private(th_id)
/// 2. Переменная nthreads будет общей для всех потоков - shared(nthreads)
/// 3. Количество созданных потоков будет равно 10 - num_threads(10)
///
/// С программой рекомендуется произвести следующие эксперименты:
/// 1. Заменить константу 10 в num_threads(10) на любое другое число
/// 2. Закоментировать '#pragma omp barrier'
/// 3. Заменить '#pragma omp master' на 'if (th_id == 0)'
/// 3. Заменить '#pragma omp master' на '#pragma omp single'
////////////////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <omp.h>

int main(int argc, char *argv[])
{
	int th_id, nthreads;
	#pragma omp parallel \
		num_threads(10) \
		private(th_id) \
		shared(nthreads)
	{
		th_id = omp_get_thread_num();
		printf("Hello World from thread %d\n", th_id);

		#pragma omp barrier

		#pragma omp master
		//	if (th_id == 0)
		{
			nthreads = omp_get_num_threads();
			printf("There are %d threads (thread %d)\n", nthreads, th_id);
		}
	}
	return 0;
}

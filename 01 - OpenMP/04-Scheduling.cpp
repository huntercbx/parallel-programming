
////////////////////////////////////////////////////////////////////////////////
/// Программа демонстрирует распределение итераций цикла между двумя потоками
///   при различных дисциплинах диспетчеризации.
///
/// При изучении программы следует обратить внимание на следующие моменты:
/// 1. 
///
/// С программой рекомендуется произвести следующие эксперименты:
/// 1. 
////////////////////////////////////////////////////////////////////////////////

#include <omp.h>
#include <cstdio>
#include <cstdlib>

const unsigned int N = 20;  // количество итераций цикла

////////////////////////////////////////////////////////////////////////////////
// функция эмулирования вычислительного процесса
////////////////////////////////////////////////////////////////////////////////
void process_data(unsigned int n)
{
	for (unsigned int i=0; i < n; ++i)
	{
		long test = 1;
		for (long l = 1; l < 1000000; ++l)
			test = test % l;
	}
}

////////////////////////////////////////////////////////////////////////////////
// функция вызывающаяся для каждой итерации цикла
////////////////////////////////////////////////////////////////////////////////
void iteration_body(unsigned int i, unsigned int task_size)
{
	int th_id = omp_get_thread_num();
	#pragma omp critical
	printf("iteration %2d, thread %d, task size %3d\n", i, th_id, task_size);
	process_data(task_size);
}

int main(int argc, char *argv[])
{
	// генерируем размеры задач случайным образом
	unsigned int task_sizes[N];
	for (size_t i = 1; i < N; ++i)
		task_sizes[i] = 100 + rand() % 500;

	printf("Default schedule\n");
	#pragma omp parallel for num_threads(2)
	for (size_t i = 1; i < N; ++i)
		iteration_body(i, task_sizes[i]);
	printf("\n");

	printf("Schedule (static, 3)\n");
	#pragma omp parallel for num_threads(2) \
	schedule(static, 3)
	for (size_t i = 1; i < N; ++i)
		iteration_body(i, task_sizes[i]);
	printf("\n");

	printf("Schedule (dynamic, 3)\n");
	#pragma omp parallel for num_threads(2) \
	schedule(dynamic, 3)
	for (size_t i = 1; i < N; ++i)
		iteration_body(i, task_sizes[i]);
	printf("\n");

	return 0;
}

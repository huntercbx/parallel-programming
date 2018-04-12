////////////////////////////////////////////////////////////////////////////////
// Данный пример создает несколько дочерних потоков с указанной дисциплиной
// диспетчеризации и ждет их завершения. Если дисциплина диспетчеризации не
// указана, то используется значение SCHED_OTHER. Для каждого потока при
// завершении выводится длительность времени, прошедшее с момента его запуска.
//
// Рекомендуется создавать количество потоков больше или равно количеству ядер
// (процессоров).
//
// Программа принимает на вход два аргумента:
//	  - дисциплина диспетчеризации - FIFO или RR (необязательный параметр)
//	  - количество создаваемых дочерних потоков
//
// Следует обратить внимание на следующие моменты:
//		- передача аргументов в функцию потока
//		- использование барьера для ожидания запуска всех потоков
//		- на порядок запуска/завершения работы потоков при различных дисциплинах
//			диспетчеризации (для этого следует запустить программу несколько раз
//			с одинаковыми параметрами и сравнить результаты).
//
////////////////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <pthread.h>
#include <string.h>
#include <stdlib.h>

pthread_barrier_t	barrier;			// барьер

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
// функция потока
////////////////////////////////////////////////////////////////////////////////
void * thread_funtion(void *x)
{
	// ждем запуска всех потоков
	pthread_barrier_wait(&barrier);

	int i;
	i = *(int*)x;
	printf("Thread start: %d\n", i);

	// эмулируем вычислительный процесс
	clock_t t1 = clock();
	process_data(500);
	clock_t t2 = clock();

	printf("Thread end: %d (execution time = %g seconds)\n", i, static_cast<float>(t2-t1)/CLOCKS_PER_SEC);
	pthread_exit(0);
}

int main(int argc, char *argv[])
{
	int				policy = SCHED_OTHER;
	sched_param		param;
	unsigned int	n_threads;
	int				res;

	// проверка параметров командной строки
	if (argc == 3)
	{
		if (strcasecmp(argv[1], "FIFO") == 0)
			policy = SCHED_FIFO;
		else if (strcasecmp(argv[1], "RR") == 0)
			policy = SCHED_RR;
		else
			policy = SCHED_OTHER;

		n_threads = atoi(argv[2]);
	}
	else if (argc == 2)
		n_threads = atoi(argv[1]);
	else
	{
		printf("Usage: %s [FIFO|RR] n_threads\n", argv[0]);
		return 0;
	}

	pthread_t		threads[n_threads];		// идентификаторы потоков
	int				treads_num[n_threads];	// аргументы потоков
	pthread_attr_t	pthread_attr;			// атрибуты потоков

	// инициализируем аттрибуты потоков
	res = pthread_attr_init(&pthread_attr);
	if (res != 0)
	{
		printf("pthread_attr_init failed (%d)\n", res);
		return 0;
	}

	// разрешаем дочерним потоки иметь собственную дисциплину диспетчеризации,
	// которая может отличатся от дисциплины основного потока
	res = pthread_attr_setinheritsched(&pthread_attr, PTHREAD_EXPLICIT_SCHED);
	if (res != 0)
	{
		printf("pthread_attr_setinheritsched failed (%d)\n", res);
		return 0;
	}

	// устанавливаем дисциплину диспетчеризации дочерних потоков
	res = pthread_attr_setschedpolicy(&pthread_attr, policy);
	if (res != 0)
	{
		printf("pthread_attr_setschedpolicy failed (%d)\n", res);
		return 0;
	}

	param.sched_priority = (sched_get_priority_min(policy) + sched_get_priority_max(policy))/2;
	res = pthread_attr_setschedparam(&pthread_attr, &param);
	if (res != 0)
	{
		printf("pthread_attr_setschedparam failed (%d)\n", res);
		return 0;
	}

	// инициализируем барьер
	res = pthread_barrier_init(&barrier, NULL, 1 + n_threads);
	if (res != 0)
	{
		printf("pthread_barrier_init failed (%d)\n", res);
		return 0;
	}

	// создание потоков
	for (int i = 0; i < n_threads; ++i)
	{
		treads_num[i] = i;
		int res = pthread_create(
			&threads[i],			// идентификатор потока
			&pthread_attr,			// аттрибуты потока
			&thread_funtion,		// функция потока
			&treads_num[i]);		// аргумент, передаваемый в функцию потока (данные)
		if (res != 0)
		{
			printf("pthread_create failed (%d)\n", res);
			return 0;
		}
	}

	// ждем запуска всех потоков
	pthread_barrier_wait(&barrier);

	// ожидание завершения потоков
	for (int i = 0; i < n_threads; ++i)
	{
		int res = pthread_join(
			threads[i],				// идентификатор потока
			NULL);					// указатель на возвращаемое значение
		if (res != 0)
			printf("pthread_join failed (%d)\n", res);
	}

	// удаление структур
	pthread_attr_destroy(&pthread_attr);
	pthread_barrier_destroy(&barrier);

	return 0;
}
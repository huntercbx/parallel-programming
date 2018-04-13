////////////////////////////////////////////////////////////////////////////////
// Моделирование "инверсии приоритетов" - ситуации когда высокоприоритетный
// поток косвенно вытесняется потоком с более низким приоритетом.
//
// Помимо этого пример иллюстрирует использование барьера синхронизации для
// предотвращения запуска других потоков раньше чем низкоприоритетный поток
// захватит мьютекс (иначе мьютекс сразу будет захвачен высокоприоритетным
// потоком).
//
// Создаются следующие потоки:
//		- низкоприоритетный поток low_priority_thread
//		- высокоприоритетный поток hi_priority_thread
//		- потоки со средним приоритетом mid_priority_thread
//			(их количество должно быть не меньше количества процессоров (ядер))
//
// В операционной системе Linux можно провести следующие эксперименты:
//		- поменять протокол работы мьютекса (PTHREAD_PRIO_PROTECT,
//			PTHREAD_PRIO_NONE, PTHREAD_PRIO_INHERIT)
//		- для протокол работы мьютекса PTHREAD_PRIO_PROTECT поменять значение
//			маскимального приоритета мьютекса
//
////////////////////////////////////////////////////////////////////////////////

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include "error_check.h"

pthread_barrier_t   barrier;                        // барьер
pthread_mutex_t     mutex;                          // мьютекс
pthread_mutexattr_t mutex_attr;                     // атрибуты мьютекса

const int           N_MID_PRIORITY_THREADS  = 4;    // количество потоков со средним приоритетом

////////////////////////////////////////////////////////////////////////////////
// вывод информации о диспетчеризации указанного потока
////////////////////////////////////////////////////////////////////////////////
void print_sched_info(pthread_t thread)
{
	// получаем параметры диспетчеризации для потока
	int             policy;
	sched_param     param;
	int             res;
	POSIX_CHECK(res = pthread_getschedparam(thread, &policy, &param));
	if (res == 0)
	{
		switch (policy)
		{
		case SCHED_FIFO:
			printf("policy = SCHED_FIFO");
			break;
		case SCHED_RR:
			printf("policy = SCHED_RR");
			break;
		case SCHED_OTHER:
			printf("policy = SCHED_OTHER");
			break;
		default:
			printf("policy = ???");
			break;
		}
		printf(", priority = %d\n", param.sched_priority);
	}
}

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
// функция низкоприоритетного потока
////////////////////////////////////////////////////////////////////////////////
void * low_priority_thread(void * )
{
	printf("Low priority thread started\n");

	// захватываем мьютекс
	int res;
	POSIX_CHECK(res = pthread_mutex_lock(&mutex));
	if (res == 0)
	{
		printf("Low priority thread succsessfully locked mutex\n");

		// ждем пока другие потоки будут созданы
		POSIX_CHECK(pthread_barrier_wait(&barrier));

		// эмулируем вычислительный процесс
		process_data(1000);

		// разблокируем мьютекс
		printf("Low priority thread unlocking mutex\n");
		POSIX_CHECK(pthread_mutex_unlock(&mutex));
	}

	printf("Low priority thread ended\n");
	pthread_exit(0);
}

////////////////////////////////////////////////////////////////////////////////
// функция высокоприоритетного потока
////////////////////////////////////////////////////////////////////////////////
void * high_priority_thread(void * )
{
	// ждем у барьера пока низкоприоритетный поток захватит мьютекс
	// и все остальные потоки будут созданы
	POSIX_CHECK(pthread_barrier_wait(&barrier));

	printf("High priority thread started\n");

	clock_t t1 = clock();

	// захватываем мьютекс
	int res;
	POSIX_CHECK(res = pthread_mutex_lock(&mutex));
	if (res == 0)
	{
		clock_t t2 = clock();
		printf("High priority thread succsessfully lock mutex\n");
		printf("Waiting of mutex for %g seconds\n", static_cast<float>(t2-t1)/CLOCKS_PER_SEC);

		// эмулируем вычислительный процесс
		process_data(1000);

		// разблокируем мьютекс
		printf("High priority thread unlocking mutex\n");
		POSIX_CHECK(pthread_mutex_unlock(&mutex));
	}

	printf("High priority thread ended\n");
	pthread_exit(0);
}

////////////////////////////////////////////////////////////////////////////////
// функция потока со средним приоритетом
////////////////////////////////////////////////////////////////////////////////
void * mid_priority_thread(void *x)
{
	// ждем у барьера пока низкоприоритетный поток захватит мьютекс
	// и все остальные потоки будут созданы
	POSIX_CHECK(pthread_barrier_wait(&barrier));

	int i;
	i = *(int*)x;
	if (N_MID_PRIORITY_THREADS > 1)
		printf("Mid priority thread started: %d\n", i);
	else
		printf("Mid priority thread started\n");

	// эмулируем вычислительный процесс
	process_data(2000);

	if (N_MID_PRIORITY_THREADS > 1)
		printf("Mid priority thread ended: %d\n", i);
	else
		printf("Mid priority thread ended\n");

	pthread_exit(0);
}

int main(int argc, char *argv[])
{
	pthread_t       threads[N_MID_PRIORITY_THREADS+2];  // идентификаторы потоков
	pthread_attr_t  pthread_attr;                       // аттрибуты потоков

	int             policy;                             // политика диспетчеризации
	struct          sched_param param;                  // параметры диспетчеризации
	int             min_priority;                       // минимальный приоритет для выбранной политики
	int             max_priority;                       // максимальный приоритет для выбранной политики
	int             low_priority;                       // приоритет низкоприоритетного потока
	int             high_priority;                      // приоритет высокоприоритетного потока
	int             mid_priority;                       // приоритеты потоков со средним приоритетом

#ifdef PTHREAD_WIN32
	// в реализации для Windows
	// получаем настройки диспетчеризации основного потока
	POSIX_CHECK_EXIT(pthread_getschedparam(pthread_self(), &policy, &param));
#else
	// в реализации для Linux
	// выбираем карусельную политику диспетчеризации
	policy = SCHED_RR;
#endif

	// получаем минимальный и максимальный приоритеты потоков
	// для выбранной политики диспетчеризации
	min_priority  = sched_get_priority_min(policy);
	max_priority  = sched_get_priority_max(policy);

	// вычисляем приоритеты для создаваемых потоков
	mid_priority  = (min_priority + max_priority)/2;
	low_priority  = (mid_priority + min_priority)/2;
	high_priority = (mid_priority + max_priority)/2;

	// выводим информацию о приоритетах
	printf("min priority  = %d\n", min_priority);
	printf("max priority  = %d\n", max_priority);
	printf("low priority  = %d\n", low_priority);
	printf("mid priority  = %d\n", mid_priority);
	printf("high priority = %d\n\n", high_priority);

	// выводим информацию о диспетчеризации основного потока
	printf("Main thread: ");
	print_sched_info(pthread_self());

	// инициализируем аттрибуты потоков
	POSIX_CHECK_EXIT(pthread_attr_init(&pthread_attr));

	// разрешаем дочерним потоки иметь собственную дисциплину диспетчеризации,
	// которая может отличатся от дисциплины основного потока
	POSIX_CHECK_EXIT(pthread_attr_setinheritsched(&pthread_attr, PTHREAD_EXPLICIT_SCHED));

	// устанавливаем дисциплину диспетчеризации дочерних потоков
	POSIX_CHECK_EXIT(pthread_attr_setschedpolicy(&pthread_attr, policy));

	// инициализируем атрибуты  мьютекса
	POSIX_CHECK_EXIT(pthread_mutexattr_init(&mutex_attr));

#ifndef PTHREAD_WIN32
	// указываем протокол работы мьютекса
	POSIX_CHECK_EXIT(pthread_mutexattr_setprotocol(&mutex_attr, PTHREAD_PRIO_INHERIT));

	// указываем максимальный приоритет мьютекса
	POSIX_CHECK_EXIT(pthread_mutexattr_setprioceiling(&mutex_attr, high_priority));
#endif

	// инициализируем мьютекс
	POSIX_CHECK_EXIT(pthread_mutex_init(&mutex, &mutex_attr));

	// инициализируем барьер
	POSIX_CHECK_EXIT(pthread_barrier_init(&barrier, NULL, 3 + N_MID_PRIORITY_THREADS));

	// создаем низкоприоритетный поток
	param.sched_priority = low_priority;
	POSIX_CHECK_EXIT(pthread_attr_setschedparam(&pthread_attr, &param));
	POSIX_CHECK_EXIT(pthread_create(
		&threads[0],            // идентификатор потока
		&pthread_attr,          // аттрибуты потока
		&low_priority_thread,   // функция потока
		NULL));                 // функция потока без аргумента

		// выводим информацию о диспетчеризации низкоприоритетного потока
	printf("Low priority thread created: ");
	print_sched_info(threads[0]);

	// создаем высокоприоритетный поток
	param.sched_priority = high_priority;
	POSIX_CHECK_EXIT(pthread_attr_setschedparam(&pthread_attr, &param));
	POSIX_CHECK_EXIT(pthread_create(
		&threads[1],            // идентификатор потока
		&pthread_attr,          // аттрибуты потока
		&high_priority_thread,  // функция потока
		NULL));                 // функция потока без аргумента

		// выводим информацию о диспетчеризации высокоприоритетного потока
	printf("High priority thread created: ");
	print_sched_info(threads[1]);

	// создаем потоки со средним приоритетом
	int thread_params[N_MID_PRIORITY_THREADS];
	param.sched_priority = mid_priority;
	POSIX_CHECK_EXIT(pthread_attr_setschedparam(&pthread_attr, &param));
	for (int i = 0; i < N_MID_PRIORITY_THREADS; ++i)
	{
		thread_params[i] = i;
		POSIX_CHECK_EXIT(pthread_create(
			&threads[2+i],          // идентификатор потока
			&pthread_attr,          // аттрибуты потока
			&mid_priority_thread,   // функция потока
			&thread_params[i]));    // аргумент, передаваемый в функцию потока

			// выводим информацию о диспетчеризации потока со средним приоритетом
		printf("Mid priority thread created: ");
		print_sched_info(threads[2+i]);
	}

	// ждем захвата мьютекса низкоприоритетным потоком
	POSIX_CHECK(pthread_barrier_wait(&barrier));

	// ожидание завершения потоков
	for (int i = 0; i < N_MID_PRIORITY_THREADS+2; ++i)
	{
		POSIX_CHECK(pthread_join(
			threads[i],             // идентификатор потока
			NULL));                  // указатель на возвращаемое значение
	}

	// удаление структур
	POSIX_CHECK(pthread_attr_destroy(&pthread_attr));
	POSIX_CHECK(pthread_mutexattr_destroy(&mutex_attr));
	POSIX_CHECK(pthread_mutex_destroy(&mutex));
	POSIX_CHECK(pthread_barrier_destroy(&barrier));

	return 0;
}
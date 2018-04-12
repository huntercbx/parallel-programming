////////////////////////////////////////////////////////////////////////////////
// Данный пример создает несколько дочерних потоков для поиска простых чисел.
// Результами работы являются количество найденных чисел и максимальное
// найденное простое число. Для предотварщения одновременного обновления
// результатов из разных потоков используется ждущая блокировка.
//
// Программа принимает на вход два аргумента:
//      - правую границу диапозона для поиска простых чисел
//      - количество создаваемых потоков
//
// Рекомендуется запустить программу дважды - с числом потоков равным числу
// процессоров и превышающим его в несколько раз. При этом диапазон для поиска
// следует выбрать таким, чтобы работа программы занимала не менее 15 сек.
// Обратить внимание на полученное время работы программы.
// Как можно объяснить полученные результаты?
////////////////////////////////////////////////////////////////////////////////

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "error_check.h"

pthread_spinlock_t  spinlock;                   // ждущая блокировка
unsigned long       n_prime_numbers     = 0;    // количество найденных простых чисел
unsigned long       last_prime_number   = 2;    // максимальное простое число

// Структура передаваемая в поток в качестве параметра
struct search_params
{
	unsigned long start;    // начало поддиапазона
	unsigned long end;      // конец поддиапазона
};

////////////////////////////////////////////////////////////////////////////////
// Функция для проверки, является ли заданное число простым
////////////////////////////////////////////////////////////////////////////////
bool is_prime(unsigned long x)
{
	bool ret = true;
	unsigned long last = (long)floor(sqrt(x));
	for (unsigned long i = 2; i <= last && ret; ++i)
		ret = (x % i) != 0;
	return ret;
}

////////////////////////////////////////////////////////////////////////////////
// Функция потока
////////////////////////////////////////////////////////////////////////////////
void * thread_funtion(void * arg)
{
	search_params* params = (search_params*) arg;
	printf("Searching thread started [%lu, %lu]\n", params->start, params->end);

	for (unsigned long number = params->start; number <= params->end; ++number)
	{
		if (is_prime(number))
		{
			// захват ждущей блокировки
			POSIX_CHECK(pthread_spin_lock(&spinlock));

			// обновление информации о найденных простых числах
			++n_prime_numbers;
			if (number > last_prime_number)
				last_prime_number = number;

			// освобождение ждущей блокировки
			POSIX_CHECK(pthread_spin_unlock(&spinlock));
		}
	}
	pthread_exit(0);
}

////////////////////////////////////////////////////////////////////////////////
// Основная программа
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char *argv[])
{
	// проверка параметров командной строки
	if (argc != 3)
	{
		printf("Usage: %s max_number n_threads\n", argv[0]);
		return 0;
	}

	const unsigned long max_number  = atol(argv[1]);    // максимальное число
	const unsigned int  n_threads   = atoi(argv[2]);    // количество потоков
	const unsigned long N = max_number/n_threads;       // количество чисел для проверки одним потоком
	pthread_t           threads[n_threads];             // идентификаторы потоков
	search_params       treads_arg[n_threads];          // аргументы потоков

	// создание ждущей блокировки
	POSIX_CHECK_EXIT(pthread_spin_init(&spinlock, PTHREAD_PROCESS_PRIVATE));

	// запоминаем время начала вычислений
	printf("Searching prime numbers in interval [2, %lu] using %d threads\n", max_number, n_threads);
	clock_t t1 = clock();

	for (unsigned int i = 0; i < n_threads; i++)
	{
		// распределение задач между потоками
		treads_arg[i].start = (i*N > 2) ? i*N : 2;
		treads_arg[i].end = (i == n_threads-1) ? max_number : ((i+1)*N - 1);

		POSIX_CHECK_EXIT( pthread_create(
			&threads[i],            // идентификатор потока
			NULL,                   // аттрибуты потока
			&thread_funtion,        // функция потока
			&treads_arg[i]));       // аргумент, передаваемый в функцию потока (данные)
	}

	// ожидание завершения потоков
	for (int i = 0; i < n_threads; ++i)
	{
		POSIX_CHECK( pthread_join(
			threads[i],             // идентификатор потока
			NULL));                 // указатель на возвращаемое значение
	}

	// запоминаем время окончания вычислений
	clock_t t2 = clock();

	printf("Found %lu prime numbers, last prime number = %lu\n", n_prime_numbers, last_prime_number);
	printf("Execution time: %g s\n", static_cast<float>(t2-t1)/CLOCKS_PER_SEC);

	pthread_spin_destroy(&spinlock);

	return 0;
}

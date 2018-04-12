////////////////////////////////////////////////////////////////////////////////
// Решение проблемы читателей-писателей с использованием блокировки
// чтения-записи.
//
// Программа принимает на вход два аргумента:
//    - количество создаваемых потоков-читателей
//    - количество создаваемых потоков-писателей
//
// По результатам работы программы подсчитывается количество произведенных
// операций чтения/записи за время TOTAL_TIME_MS
////////////////////////////////////////////////////////////////////////////////

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "error_check.h"

const unsigned int READ_TIME_MS     = 1;    // Время чтения в мс (эмуляция долгих операций)
const unsigned int WRITE_TIME_MS    = 10;   // Время записи в мс (эмуляция долгих операций)
const unsigned int TOTAL_TIME_MS    = 5000; // Общее время работы программы

// блокировка чтения-записи (со статической инициализацией)
pthread_rwlock_t rwlock	= PTHREAD_RWLOCK_INITIALIZER;

// флаг для остановки работы всех потоков
bool abort_all_threads = false;

////////////////////////////////////////////////////////////////////////////////
// Замена функции Sleep из WinAPI
////////////////////////////////////////////////////////////////////////////////
#ifndef PTHREAD_WIN32
void Sleep(unsigned int msTime)
{
	timespec	time;
	time.tv_sec	= msTime/1000;
	time.tv_nsec = (msTime - time.tv_sec*1000)*1000*1000;
	int res = nanosleep(&time, NULL);
}
#endif

////////////////////////////////////////////////////////////////////////////////
// функция потока-читателя
////////////////////////////////////////////////////////////////////////////////
void * reader_thread(void * arg)
{
	// количество операций чтения, выполненных данных потоком
	unsigned int *n_read = (unsigned int *)arg;

	int res;

	while (!abort_all_threads)
	{
		// блокируем на чтение
		POSIX_CHECK(res = pthread_rwlock_rdlock(&rwlock))
		if (res == 0)
		{
			++(*n_read);
			Sleep(READ_TIME_MS);

			POSIX_CHECK(pthread_rwlock_unlock(&rwlock));
		}
	}

	pthread_exit(0);
}

////////////////////////////////////////////////////////////////////////////////
// функция потока-писателя
////////////////////////////////////////////////////////////////////////////////
void * writer_thread(void * arg)
{
	// количество операций записи, выполненных данных потоком
	unsigned int *n_write = (unsigned int *)arg;

	int res;

	while (!abort_all_threads)
	{
		// блокируем на запись
		POSIX_CHECK(res = pthread_rwlock_wrlock(&rwlock));
		if (res == 0)
		{
			++(*n_write);
			Sleep(WRITE_TIME_MS);

			POSIX_CHECK(pthread_rwlock_unlock(&rwlock));
		}
	}

	pthread_exit(0);
}

////////////////////////////////////////////////////////////////////////////////
// Основная программа
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
	// проверка параметров командной строки
	if (argc != 3)
	{
		printf("Usage: %s n_readers n_writers\n", argv[0]);
		return 0;
	}

	const unsigned int n_readers = atoi(argv[1]);   // Количество задач-читателей
	const unsigned int n_writers = atoi(argv[2]);   // Количество задач-писателей
	pthread_t          reader_threads[n_readers];   // идентификаторы потоков-читателей
	pthread_t          writer_threads[n_writers];   // идентификаторы потоков-писателей
	unsigned int       n_reads[n_readers];          // Количество произведенных операций чтения каждым потоком
	unsigned int       n_writes[n_writers];         // Количество произведенных операций записи каждым потоком

	// создание потоков-читателей
	for (unsigned int i = 0; i < n_readers; ++i)
	{
		// инициализируем нулем количество операций чтения, произведенных потоком
		n_reads[i] = 0;

		POSIX_CHECK_EXIT( pthread_create(
			&reader_threads[i],     // идентификатор потока
			NULL,                   // аттрибуты потока
			&reader_thread,         // функция потока
			&n_reads[i]));          // аргумент, передаваемый в функцию потока (данные)

		printf("Reader thread %d created\n", i);
	}

	// создание потоков-писателей
	for (unsigned int i = 0; i < n_writers; ++i)
	{
		// инициализируем нулем количество операций записи, произведенных потоком
		n_writes[i] = 0;

		POSIX_CHECK_EXIT( pthread_create(
			&writer_threads[i],     // идентификатор потока
			NULL,                   // аттрибуты потока
			&writer_thread,         // функция потока
			&n_writes[i]));         // аргумент, передаваемый в функцию потока (данные)

		printf("Writer thread %d created\n", i);
	}

	// Ждем указанное время для получения результатов и выставляем флаг для завершения всех потоков
	Sleep(TOTAL_TIME_MS);
	abort_all_threads = true;

	// Ожидаем заверешения потоков-читателей и подсчитываем количество операций чтения
	unsigned int total_reads = 0;
	for (unsigned int i = 0; i < n_readers; ++i)
	{
		POSIX_CHECK( pthread_join(
			reader_threads[i],      // идентификатор потока
			NULL));                 // указатель на возвращаемое значение

		printf("Number of reads by thread %d: %d\n", i, n_reads[i]);
		total_reads += n_reads[i];
	}
	printf("Total number of reads: %d\n", total_reads);

	// Ожидаем заверешения потоков-писателей и подсчитываем количество операций записи
	unsigned int total_writes = 0;
	for (unsigned int i = 0; i < n_writers; ++i)
	{
		POSIX_CHECK( pthread_join(
			writer_threads[i],      // идентификатор потока
			NULL));                 // указатель на возвращаемое значение

		printf("Number of writes by thread %d: %d\n", i, n_writes[i]);
		total_writes += n_writes[i];
	}
	printf("Total number of writes: %d\n", total_writes);

	return 0;
}

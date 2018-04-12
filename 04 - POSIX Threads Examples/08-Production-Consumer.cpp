////////////////////////////////////////////////////////////////////////////////
// Данный пример демонстрирует использование условных переменных для решения
// классической задачи "Производитель-Потребитель" с использованием очереди
// на основе циклического буфера.
//
// Условные переменные используются для проверки двух условий - на непустую и
// непереполненную очередь. В случае пустой очереди поток-потребитель будет
// ждать пока в очереди не появятся новые элементы, а в случае полной очереди
// поток-производитель приостоновит свою работу до появления свободного места
// в очереди.
//
// Работа потока-производителя останавливается из основого потока программы.
// После остановки потока-производителя поток-потребитель продолжит работу
// до тех пор пока не обработает все оставшиеся элементы из очереди.
//
// С данным примером можно провести следующий эксперимент - поменять местами
// максимальные паузы между циклами производства.
//
// ВНИМАНИЕ! Программа специально содержит логическую ошибку и в некоторых
// случаях возможно ее "зависание"
////////////////////////////////////////////////////////////////////////////////

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include "error_check.h"

double          start                   = 0;    // время запуска программы в секундах
const int       PRODUCER_SLEEP_TIME_MS  = 1000; // максимальная пауза между циклами производства
const int       CONSUMER_SLEEP_TIME_MS  = 3000; // максимальная пауза между циклами потребления

// переменные для организации циклического буфера на основе массива
const int       BUFFER_SIZE             = 8;    // размер циклического буфера
int             buffer[BUFFER_SIZE];            // циклический буфер для очереди
unsigned int    queue_start             = 0;    // индекс начала буфера
unsigned int    queue_end               = 0;    // индекс конца буфера

// флаг для остановки потока-производителя
bool            stop_production         = false;

// мьютекс для доступа к очереди (со статической инициализацией)
pthread_mutex_t queue_mutex             = PTHREAD_MUTEX_INITIALIZER;

// условные переменные для непереполненого и непустого буфера (со статической инициализацией)
pthread_cond_t  queue_not_full_cond     = PTHREAD_COND_INITIALIZER;
pthread_cond_t  queue_not_empty_cond    = PTHREAD_COND_INITIALIZER;

// обеспечении единой последовательности случайных чисел для всех потоков
pthread_mutex_t     rnd_mutex   = PTHREAD_MUTEX_INITIALIZER;    // мьютекс для ГПСЧ
int                 rnd_seed    = 1;                            // Значение SEED для ГПСЧ

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
// Функция генерации псевдослучайного числа с общей для всех потоков
// последовательностью
////////////////////////////////////////////////////////////////////////////////
int my_rand()
{
	POSIX_CHECK(pthread_mutex_lock(&rnd_mutex));
	srand(rnd_seed);
	int res = rnd_seed = rand();
	POSIX_CHECK(pthread_mutex_unlock(&rnd_mutex));
	return res;
}

////////////////////////////////////////////////////////////////////////////////
// функция для определения размера очереди
////////////////////////////////////////////////////////////////////////////////
unsigned int get_queue_size()
{
	return (queue_start <= queue_end) ?
		queue_end - queue_start :
		queue_end + BUFFER_SIZE - queue_start;
}

////////////////////////////////////////////////////////////////////////////////
// функция для вывода в лог сообщения
////////////////////////////////////////////////////////////////////////////////
void write_log(const char* thread_name, const char* message)
{
	timeval tv;
	gettimeofday(&tv, NULL);
	double seconds = tv.tv_sec + (double)tv.tv_usec/1000000.0;

	printf("%10.6f [%s] %s\n", seconds - start, thread_name, message);
}

////////////////////////////////////////////////////////////////////////////////
// функция потока-производителя
////////////////////////////////////////////////////////////////////////////////
void * producer(void *)
{
	char message[80];

	write_log("PRODUCER", "Thread started");

	// Цикл производства
	while (true)
	{
		// проверка на остановку производства
		if (stop_production) break;

		POSIX_CHECK(pthread_mutex_lock(&queue_mutex));

		// если очередь переполнена - ждем
		while (get_queue_size() == BUFFER_SIZE - 1)
		{
			write_log("PRODUCER", "Thread is waiting (queue is full)");
			POSIX_CHECK(pthread_cond_wait(&queue_not_full_cond, &queue_mutex));
		}

		// добавляем новый элемент в очередь
		int item = my_rand() % 100000;
		buffer[queue_end] = item;
		queue_end = (queue_end + 1) % BUFFER_SIZE;

		sprintf(message, "Item (%5d) produced,  queue size = %2d (%2d, %2d)", item, get_queue_size(), queue_start, queue_end);
		write_log("PRODUCER", message);
		POSIX_CHECK(pthread_mutex_unlock(&queue_mutex));

		// посылаем сигнал для пробуждения потока-потребителя в случае,
		// если он ждет появления новых элементов в очереди
		POSIX_CHECK(pthread_cond_signal(&queue_not_empty_cond));

		Sleep (my_rand() % PRODUCER_SLEEP_TIME_MS);
	}

	write_log("PRODUCER", "Thread exit");
	pthread_exit(0);
}

////////////////////////////////////////////////////////////////////////////////
// функция потока-потребителя
////////////////////////////////////////////////////////////////////////////////
void * consumer(void *)
{
	char message[80];

	write_log("CONSUMER", "Thread started");

	// Цикл потребления
	while (true)
	{
		POSIX_CHECK(pthread_mutex_lock(&queue_mutex));

		// проверка на окончание работы
		// выполняется после блокировки мьютекса, чтобы избежать добавления новых
		// элементов в очередь в момент проверки
		if (stop_production && get_queue_size() == 0) break;

		// если очередь пуста - ждем
		while (get_queue_size() == 0)
		{
			write_log("CONSUMER", "Thread is waiting (queue is empty)");
			POSIX_CHECK(pthread_cond_wait(&queue_not_empty_cond, &queue_mutex));
		}

		// удаляем обработанный элемент из очереди
		int item = buffer[queue_start];
		queue_start = (queue_start + 1) % BUFFER_SIZE;

		sprintf (message, "Item (%5d) processed, queue size = %2d (%2d, %2d)", item, get_queue_size(), queue_start, queue_end);
		write_log("CONSUMER", message);
		POSIX_CHECK(pthread_mutex_unlock(&queue_mutex));

		// посылаем сигнал для пробуждения потока-производителя в случае,
		// если он ждет освобождения места в очереди
		POSIX_CHECK(pthread_cond_signal(&queue_not_full_cond));

		Sleep (my_rand() % CONSUMER_SLEEP_TIME_MS);
	}

	write_log("CONSUMER", "Thread exit");
	pthread_exit(0);
}

////////////////////////////////////////////////////////////////////////////////
// Основная программа
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char *argv[])
{
	// запоминаем время запуска программы
	timeval tv;
	gettimeofday(&tv, NULL);
	start = tv.tv_sec + (double)tv.tv_usec/1000000.0;

	printf("Press enter to stop\n");

	pthread_t threads[2];       // идентификаторы потоков

	// создание потока-производителя
	POSIX_CHECK_EXIT( pthread_create(
		&threads[0],            // идентификатор потока
		NULL,                   // аттрибуты потока
		&producer,              // функция потока
		NULL));                 // функция потока без аргумента

	// создание потока-потребителя
	POSIX_CHECK_EXIT( pthread_create(
		&threads[1],            // идентификатор потока
		NULL,                   // аттрибуты потока
		&consumer,              // функция потока
		NULL));                 // функция потока без аргумента

	getchar();
	stop_production = true;

	// ожидание завершения потоков
	for (int i = 0; i < 2; ++i)
	{
		POSIX_CHECK( pthread_join(
			threads[i],         // идентификатор потока
			NULL));             // указатель на возвращаемое значение
	}

	return 0;
}

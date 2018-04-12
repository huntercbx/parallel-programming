////////////////////////////////////////////////////////////////////////////////
// Данный пример иллюстрирует создание и использование переменных в локальном
// хранилище потока.
// Для каждого дочернего потока создается своя копия данных, доступ к которым
// можно получить по ключу.
//
// В данном примере также демонстрируется использование однократной
// инициализации для создания ключа доступа к данным потока.
//
// Помимо этого можно обратить внимание на механизм генерации псевдослучайных
// чисел из разных потоков, ориентированный на общую для всех потоков
// последовательность чисел.
////////////////////////////////////////////////////////////////////////////////

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

const unsigned int  N_THREADS   = 4;                    // количество создаваемых потоков
pthread_key_t       data_key;                           // ключ для доступа к данным потока
pthread_once_t      key_once    = PTHREAD_ONCE_INIT;    // однократная инициализация

// обеспечении единой последовательности случайных чисел для всех потоков
pthread_mutex_t     mutex   = PTHREAD_MUTEX_INITIALIZER;    // мьютекс для ГПСЧ
int                 seed    = 1;                            // Значение SEED для ГПСЧ

// структура для хранения данных потока
struct thread_data
{
	unsigned int	n;
	float			n_factorial;
};

#ifndef PTHREAD_WIN32
////////////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////////////
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
// функция-деструктор, вызваемая при завершении потока
////////////////////////////////////////////////////////////////////////////////
void data_destructor(void * ptr)
{
	thread_data *data = (thread_data *) ptr;
	printf("Data destrutor called for (%d, %g)\n", data->n, data->n_factorial);
	pthread_setspecific(data_key, NULL);
	free(data);
}

////////////////////////////////////////////////////////////////////////////////
// функция создания ключа для доступа к данным потока
////////////////////////////////////////////////////////////////////////////////
void create_key()
{
	// создаем ключ для доступа к данным потока
	int res = pthread_key_create(&data_key, data_destructor);
	if (res != 0)
		printf("pthread_key_create failed (%d)\n", res);
	else
		printf("Key was successfully created\n");
}

////////////////////////////////////////////////////////////////////////////////
// функция вычисление факториала
////////////////////////////////////////////////////////////////////////////////
void calc_factorial()
{
	thread_data *data = (thread_data *) pthread_getspecific(data_key);

	// calculate factorial
	data->n_factorial = 1;
	for (unsigned int i = 1; i <= data->n; ++i)
		data->n_factorial *= i;
}

////////////////////////////////////////////////////////////////////////////////
// вывод данных потока
////////////////////////////////////////////////////////////////////////////////
void print_thread_data()
{
	thread_data *data = (thread_data *) pthread_getspecific(data_key);
	printf("%d! = %g\n", data->n, data->n_factorial);
}

////////////////////////////////////////////////////////////////////////////////
// функция потока
////////////////////////////////////////////////////////////////////////////////
void * thread_funtion(void *)
{
	printf("Child thread started\n");

	Sleep(1);

	// создаем ключ для доступа к данным потока
	pthread_once(&key_once, create_key);

	if (pthread_getspecific(data_key) == NULL)
	{
		// генерация случайного числа
		pthread_mutex_lock(&mutex);
		srand(seed);
		int k = seed = rand();
		pthread_mutex_unlock(&mutex);

		// создаем и инициализируем данные потока
		thread_data *data = (thread_data *) malloc(sizeof(thread_data));
		data->n = 1 + (k % 10);
		data->n_factorial = 0;

		// устанавливаем соответсвие данных потока и ключа доступа
		int res = pthread_setspecific(data_key, data);
		if (res != 0)
			printf("pthread_setspecific failed (%d)\n", res);
    }

	calc_factorial();
	print_thread_data();

	Sleep(1);

	printf("Child thread terminated\n");
	pthread_exit(0);
}

int main(int argc, char *argv[])
{
	pthread_t threads[N_THREADS];	// идентификаторы потоков

	// создание потоков
	for (unsigned int i = 0; i < N_THREADS; ++i)
	{
		int res = pthread_create(
			&threads[i],			// идентификатор потока
			NULL,					// аттрибуты потока
			&thread_funtion,		// функция потока
			NULL);					// функция потока без аргумента
		if (res != 0)
			printf("pthread_create failed (%d)\n", res);
	}

	// ожидание завершения потоков
	for (int i = 0; i < N_THREADS; ++i)
	{
		int res = pthread_join(
			threads[i],				// идентификатор потока
			NULL);					// указатель на возвращаемое значение
		if (res != 0)
			printf("pthread_join failed (%d)\n", res);
	}

	// удаляем ключ для доступа к данным потока
	pthread_key_delete(data_key);

	return 0;
}
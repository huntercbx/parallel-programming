////////////////////////////////////////////////////////////////////////////////
// Данный пример демонстирует прерывание выполнения дочернего потока.
// Дочерний поток блокирует мьютекс и выделяет память - при прерывании работы
// потока эти ситуации должны быть корректно обработаны, для чего в примере
// устанавливаются соответсвующие обработчики.
// Также следует обратить внимение, что работа потока может быть прервана
// только в специальных точках программы.
//
// С данным примером рекомендуется провести следующие эксперименты:
//	1. Разрешить прервать дочерний поток (allow_abort = true) только после
//		основного цикла эмулирования вычислительного процесса. Обратить внимание
//		 на то, как были вызваны обработчики прерывания потока.
//	2. Удалить обработчик освобожения мьютекса. Программа после этого окажется в
//		ситуации взаимной блокировки и может быть завершена по Ctrl-Break.
//	3. Изменить тип мьютекса на PTHREAD_MUTEX_ROBUST и посмотреть как изменится
//		поведение программы.
////////////////////////////////////////////////////////////////////////////////
#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>
#include <errno.h>

pthread_mutex_t		mutex;					// мьютекс
pthread_mutexattr_t	mutex_attr;				// атрибуты мьютекса
bool				aborted		= false;	// флаг для проверки причины завершения потока
bool				allow_abort	= false;	// флаг для предовращения прерывания дочернего процесса до захвата мьютекса
const unsigned int	N_LOOPS		= 500;		// количество внешних циклов при эмулировании вычислений

////////////////////////////////////////////////////////////////////////////////
// Обработчик прерывании работы дочернего потока
// функция освобождает мьютекс переданный ей в качестве аргумента
////////////////////////////////////////////////////////////////////////////////
void release_mutex_func(void * arg)
{
	if (aborted)
		printf("Child thread was canceled - ");
	printf("Unlock mutex\n");

	pthread_mutex_t* m = (pthread_mutex_t *) arg;
	int res = pthread_mutex_unlock(m);
	if (res != 0)
		printf("pthread_mutex_unlock failed (%d)\n", res);
}

////////////////////////////////////////////////////////////////////////////////
// Обработчик прерывании работы дочернего потока
// функция освобождает блок памяти, переданный ей в качестве аргумента
////////////////////////////////////////////////////////////////////////////////
void release_memory_func (void * arg)
{
	if (aborted)
		printf("Child thread was canceled - ");
	printf("Release memory\n");
	free (arg);
}

////////////////////////////////////////////////////////////////////////////////
// функция потока
////////////////////////////////////////////////////////////////////////////////
void * thread_func(void * )
{
	printf("Child thread started\n");

	// захватываем мьютекс
	int res = pthread_mutex_lock(&mutex);
	if (res != 0)
		printf("pthread_mutex_lock failed (%d)\n", res);
	else
	{
		printf("Child thread locked mutex\n");
		// устанавливаем обработчик для разблокирования мьютекса
		pthread_cleanup_push(&release_mutex_func, &mutex);

		// выделяем память
		long * buffer = (long *) malloc(N_LOOPS*sizeof(long));
		printf("Child thread allocate a memory\n");
		// устанавливаем обработчик для освобождения памяти
		pthread_cleanup_push(&release_memory_func, buffer);

		// теперь можно прерывать выполенние потока
		allow_abort = true;

		// эмулируем вычислительный процесс
		for (unsigned int i=0; i < N_LOOPS; ++i)
		{
			long k = 1;
			for (long j = 1; j < 1000000; ++j)
				k = k % j;

			buffer[i] = k;

			// проверка на прерывание работы потока
			aborted = true;
			pthread_testcancel();
			aborted = false;
		}

		// вызываем обработчик осбождения памяти и удаляем его
		pthread_cleanup_pop(1);

		// проверка на прерывание работы потока
		aborted = true;
		pthread_testcancel();
		aborted = false;

		// вызываем обработчик разблокирования мьютекса и удаляем его
		pthread_cleanup_pop(1);
	}

	printf("Thread ended\n");
	pthread_exit(0);
}

int main(int argc, char *argv[])
{
	// инициализируем атрибуты мьютекса
	int res = pthread_mutexattr_init(&mutex_attr);
	if (res != 0)
	{
		printf("pthread_mutexattr_init failed (%d)\n", res);
		return 0;
	}
	res = pthread_mutexattr_setrobust(&mutex_attr, PTHREAD_MUTEX_STALLED);
	if (res != 0)
	{
		printf("pthread_mutexattr_setrobust failed (%d)\n", res);
		return 0;
	}

	// инициализируем мьютекс
	res = pthread_mutex_init(&mutex, &mutex_attr);
	if (res != 0)
	{
		printf("pthread_mutex_init failed (%d)\n", res);
		return 0;
	}

	// создаем поток
	pthread_t		thread_id;	// идентификатор потока
	res = pthread_create(
		&thread_id,				// идентификатор потока
		NULL,					// аттрибуты потока по умолчанию
		&thread_func,			// функция потока
		NULL);					// функция потока без аргумента
	if (res != 0)
	{
		printf("pthread_create failed (%d)\n", res);
		return 0;
	}

	// ждем чтобы не прервать работу дочернего потока раньше времени
	while (!allow_abort) {}

	// прерываем работу дочернего потока
	res = pthread_cancel(thread_id);
	if (res != 0)
	{
		printf("pthread_cancel failed (%d)\n", res);
		return 0;
	}

	// захватываем мьютекс
	res = pthread_mutex_lock(&mutex);
	if (res == 0)
	{
		printf("Main thread succsessfully lock mutex\n");

		// разблокируем мьютекс
		res = pthread_mutex_unlock(&mutex);
		if (res != 0)
			printf("pthread_mutex_lock failed (%d)\n", res);
		else
			printf("Main thread succsessfully unlock mutex\n");
	}
	else if (res == EOWNERDEAD)
		printf("Mutex state is inconsistent\n");
	else
		printf("pthread_mutex_lock failed (%d)\n", res);

	pthread_mutex_destroy(&mutex);
	pthread_mutexattr_destroy(&mutex_attr);
}
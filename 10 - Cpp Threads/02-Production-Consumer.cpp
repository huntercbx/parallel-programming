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

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <chrono>
#include <mutex>
#include <condition_variable>
#include <thread>

// время запуска программы
std::chrono::time_point<std::chrono::system_clock> start;

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
std::mutex queue_mutex;

// условные переменные для непереполненого и непустого буфера (со статической инициализацией)
std::condition_variable queue_not_full_cond;
std::condition_variable queue_not_empty_cond;

// обеспечении единой последовательности случайных чисел для всех потоков
std::mutex      rnd_mutex;                      // мьютекс для ГПСЧ
int             rnd_seed = 1;                   // Значение SEED для ГПСЧ

////////////////////////////////////////////////////////////////////////////////
// Функция генерации псевдослучайного числа с общей для всех потоков
// последовательностью
////////////////////////////////////////////////////////////////////////////////
int my_rand()
{
	std::lock_guard<std::mutex> lock(rnd_mutex);
	srand(rnd_seed);
	int res = rnd_seed = rand();
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
	auto t1 = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_seconds = t1 - start;
	printf("%10.6f [%s] %s\n", elapsed_seconds.count(), thread_name, message);
}

////////////////////////////////////////////////////////////////////////////////
// функция потока-производителя
////////////////////////////////////////////////////////////////////////////////
void producer()
{
	char message[80];

	write_log("PRODUCER", "Thread started");

	// Цикл производства
	while (true)
	{
		// проверка на остановку производства
		if (stop_production) break;

		// добавление нового элемента в очередь
		{
			std::unique_lock<std::mutex> lock(queue_mutex);
			
			// если очередь переполнена - ждем
			queue_not_full_cond.wait(lock, [] { return get_queue_size() != BUFFER_SIZE - 1; });

			// добавляем новый элемент в очередь
			int item = my_rand() % 100000;
			buffer[queue_end] = item;
			queue_end = (queue_end + 1) % BUFFER_SIZE;

			sprintf(message, "Item (%5d) produced,  queue size = %2d (%2d, %2d)", item, get_queue_size(), queue_start, queue_end);
			write_log("PRODUCER", message);
		}

		// посылаем сигнал для пробуждения потока-потребителя в случае,
		// если он ждет появления новых элементов в очереди
		queue_not_empty_cond.notify_one();

		std::this_thread::sleep_for(std::chrono::milliseconds(my_rand() % PRODUCER_SLEEP_TIME_MS));
	}

	write_log("PRODUCER", "Thread exit");
}

////////////////////////////////////////////////////////////////////////////////
// функция потока-потребителя
////////////////////////////////////////////////////////////////////////////////
void consumer()
{
	char message[80];

	write_log("CONSUMER", "Thread started");

	// Цикл потребления
	while (true)
	{
		std::unique_lock<std::mutex> lock(queue_mutex);

		// проверка на окончание работы
		// выполняется после блокировки мьютекса, чтобы избежать добавления новых
		// элементов в очередь в момент проверки
		if (stop_production && get_queue_size() == 0) break;

		// если очередь пуста - ждем
		queue_not_empty_cond.wait(lock, [] { return get_queue_size() != 0; });

		// удаляем обработанный элемент из очереди
		int item = buffer[queue_start];
		queue_start = (queue_start + 1) % BUFFER_SIZE;

		sprintf (message, "Item (%5d) processed, queue size = %2d (%2d, %2d)", item, get_queue_size(), queue_start, queue_end);
		write_log("CONSUMER", message);

		// посылаем сигнал для пробуждения потока-производителя в случае,
		// если он ждет освобождения места в очереди
		lock.unlock();
		queue_not_full_cond.notify_one();

		std::this_thread::sleep_for(std::chrono::milliseconds(my_rand() % CONSUMER_SLEEP_TIME_MS));
	}

	write_log("CONSUMER", "Thread exit");
}

////////////////////////////////////////////////////////////////////////////////
// Основная программа
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char *argv[])
{
	// запоминаем время запуска программы
	start = std::chrono::system_clock::now();

	printf("Press enter to stop\n");

	std::vector<std::thread> threads;         // идентификаторы потоков
	threads.push_back(std::thread(producer)); // создание потока-производителя
	threads.push_back(std::thread(consumer)); // создание потока-потребителя

	getchar();
	stop_production = true;

	// ожидание завершения потоков
	for (auto& th : threads)
		th.join();

	return 0;
}

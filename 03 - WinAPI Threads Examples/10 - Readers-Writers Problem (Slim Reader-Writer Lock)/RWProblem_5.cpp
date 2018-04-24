////////////////////////////////////////////////////////////////////////////////
// Справедливое решение проблемы читателей-писателей
// (каждый поток должен получить доступ за конечное время)
//
// В качестве разделяемого ресурса используется обычный буфер.
//
// Процесс-читатель проверяет валидность данных при чтении (все элементы буфера
// содержат одинаковый символ), в случае ошибки - выводится сообщение (ошибок
// быть не должно).
//
// Процесс-писатель просто обновляет данные в буфере.
// 
// По результатам работы программы подсчитывается количество произведенных
// операций чтения/записи за время TOTAL_TIME_MS
//
// С данным примером можно провести следующие эксперименты:
//  1. Изменить количественное соотношение между потоками читателями и писателями
//     и посмотреть как изменятся результаты.
////////////////////////////////////////////////////////////////////////////////

#include <windows.h>
#include <stdio.h>
#include <climits>
#include <ctime>

const unsigned int N_READERS        = 4;            // Количество задач-читателей
const unsigned int N_WRITERS        = 4;            // Количество задач-писателей
const DWORD	MUTEX_WAIT_TIME_MS      = 1000;         // Время ожидания мьютекса в мс
const DWORD	TOTAL_TIME_MS           = 10000;        // Общее время работы программы
const DWORD	READ_TIME_MS            = 1;            // Время чтения в мс (эмуляция долгих операций)
const DWORD	WRITE_TIME_MS           = 1;            // Время записи в мс (эмуляция долгих операций)

const unsigned int BUFFER_SIZE      = 50;           // Размер буфера
char buffer[BUFFER_SIZE];                           // Буфер - играет роль общей области памяти

bool abort_all_threads              = false;        // Признак завершения работы всех потоков

SRWLOCK lock;                                       // Блокировка чтения-записи
HANDLE hThreadArray[N_READERS + N_WRITERS];         // Дескрипторы потоков
DWORD dwReadersThreadID[N_READERS];                 // Идентификаторы потоков-читателей
DWORD dwWritersThreadID[N_WRITERS];                 // Идентификаторы потоков-писателей
LARGE_INTEGER PERFORMANCE_COUNTER_FREQUENCY;        // Частота таймера для измерения производительности

unsigned int N_Reads[N_READERS];                    // Количество произведенных операций чтения
unsigned int N_Writes[N_WRITERS];                   // Количество произведенных операций записи

////////////////////////////////////////////////////////////////////////////////
// Функция эмуляции задержки операций ввода-вывода
////////////////////////////////////////////////////////////////////////////////
void DelayMS(DWORD time)
{
	LARGE_INTEGER t0, t1;
	QueryPerformanceCounter(&t0);
	QueryPerformanceCounter(&t1);
	const auto dt = PERFORMANCE_COUNTER_FREQUENCY.QuadPart / 1000 * time;
	while (true)
	{
		if (t1.QuadPart - t0.QuadPart >= dt) return;
		QueryPerformanceCounter(&t1);
	}
}

////////////////////////////////////////////////////////////////////////////////
// Эмуляция операции записи
////////////////////////////////////////////////////////////////////////////////
void WriteOperation(unsigned int* counter = NULL)
{
	// записываем данные в память
	char ch = buffer[0] == 'Z' ? 'A' : buffer[0] + 1;
	for (unsigned int i = 0; i < BUFFER_SIZE; ++i)
		buffer[i] = ch;

	// увеличиваем счетчик операций
	if (counter != NULL)
		++(*counter);

	// задержка для эмуляции длительности операции
	DelayMS(WRITE_TIME_MS);
}

////////////////////////////////////////////////////////////////////////////////
// Эмуляция операции чтения с проверкой целостности буфера
////////////////////////////////////////////////////////////////////////////////
void ReadOperation(unsigned int* counter = NULL)
{
	char local_buffer[BUFFER_SIZE + 1];
	local_buffer[BUFFER_SIZE] = 0;

	// читаем данные из памяти
	bool error = false;
	for (unsigned int i = 0; i < BUFFER_SIZE; ++i)
	{
		local_buffer[i] = buffer[i];
		error |= i > 0 && local_buffer[i] != local_buffer[i - 1];
	}

	// увеличиваем счетчик операций
	if (counter != NULL)
		++(*counter);

	// если была ошибка, выводим сообщение об ошибке
	if (error)
	{
		printf_s("Reader thread %5d detect buffer corruption: %s\n",
			GetCurrentThreadId(), local_buffer);
	}

	// задержка для эмуляции длительности операции
	DelayMS(READ_TIME_MS);
}

////////////////////////////////////////////////////////////////////////////////
// Функция потока-писателя
////////////////////////////////////////////////////////////////////////////////
DWORD WINAPI WriterThreadFunction(LPVOID lpParam)
{
	unsigned int *numWrites = reinterpret_cast<unsigned int*>(lpParam);

	while (!abort_all_threads)
	{
		AcquireSRWLockExclusive(&lock);
		WriteOperation(numWrites);
		ReleaseSRWLockExclusive(&lock);
	}

	return 0;
}

////////////////////////////////////////////////////////////////////////////////
// Функция потока-читателя
////////////////////////////////////////////////////////////////////////////////
DWORD WINAPI ReaderThreadFunction(LPVOID lpParam)
{
	unsigned int *numReads = reinterpret_cast<unsigned int*>(lpParam);

	while (!abort_all_threads)
	{
		AcquireSRWLockShared(&lock);
		ReadOperation(numReads);
		ReleaseSRWLockShared(&lock);
	}

	return 0;
}

////////////////////////////////////////////////////////////////////////////////
// Основная программа
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
	// Получаем частоту таймера для измерения производительности
	QueryPerformanceFrequency(&PERFORMANCE_COUNTER_FREQUENCY);

	// Инициализируем буфер
	for (size_t i = 0; i < BUFFER_SIZE; ++i)
		buffer[i] = 'A';

	// Инициализация блокировки чтения-записи
	InitializeSRWLock(&lock);

	// Создание потоков-читателей
	for (unsigned int i = 0; i < N_READERS; ++i)
	{
		// инициализируем нулем количество операций чтения, произведенных потоком
		N_Reads[i] = 0;

		hThreadArray[i] = CreateThread(
			NULL,                   // использовать настройки безопасности по умолчанию
			0,                      // использовать ращмер стека по умолчанию
			ReaderThreadFunction,   // имя функции потока
			&N_Reads[i],            // аргумент - количество операций чтения, произведенных потоком
			0,                      // создать поток с флагами по умолчанию
			&dwReadersThreadID[i]   // идентификатор потока
		);
		// Если поток не был создан - завершаем работу с ошибкой
		if (hThreadArray[i] == NULL)
			ExitProcess(1);

		// Выводим сообщение об успешном создании потока
		printf_s("Reader thread %5d sucsessfully created\n", dwReadersThreadID[i]);
	}

	// Создание потоков-писатлей
	for (unsigned int i = 0; i < N_WRITERS; ++i)
	{
		// инициализируем нулем количество операций записи, произведенных потоком
		N_Reads[i] = 0;

		hThreadArray[N_READERS + i] = CreateThread(
			NULL,                   // использовать настройки безопасности по умолчанию
			0,                      // использовать ращмер стека по умолчанию
			WriterThreadFunction,   // имя функции потока
			&N_Writes[i],           // аргумент - количество операций записи, произведенных потоком
			0,                      // создать поток с флагами по умолчанию
			&dwWritersThreadID[i]   // идентификатор потока
		);

		// Если поток не был создан - завершаем работу с ошибкой
		if (hThreadArray[N_READERS + i] == NULL)
			ExitProcess(1);

		// Выводим сообщение об успешном создании потока
		printf_s("Writer thread %5d sucsessfully created\n", dwWritersThreadID[i]);
	}

	// Ждем окончания эксперимента и выставляем флаг для завершения всех потоков
	Sleep(TOTAL_TIME_MS);
	abort_all_threads = true;

	// Ожидаем завершения всех потоков
	WaitForMultipleObjects(
		N_READERS + N_WRITERS,  // общее количество потоков
		hThreadArray,           // дескрипторы потоков
		TRUE,                   // ждем завершения всех потоков
		INFINITE);              // ждем в течении бесконечного времени

	// Освобождаем дескрипторы
	for (unsigned int i = 0; i < N_READERS + N_WRITERS; ++i)
		CloseHandle(hThreadArray[i]);

	// Подсчитываем общее количество произведенных операций чтения
	unsigned int N_Reads_Total = 0;
	for (unsigned int i = 0; i < N_READERS; ++i)
	{
		printf_s("Number of reads by thread %5d: %d\n", dwReadersThreadID[i], N_Reads[i]);
		N_Reads_Total += N_Reads[i];
	}
	printf_s("Total number of reads: %d\n", N_Reads_Total);

	// Подсчитываем общее количество произведенных операций чтения
	unsigned int N_Writes_Total = 0;
	for (unsigned int i = 0; i < N_WRITERS; ++i)
	{
		printf_s("Number of writes by thread %5d: %d\n", dwWritersThreadID[i], N_Writes[i]);
		N_Writes_Total += N_Writes[i];
	}
	printf_s("Total number of writes: %d\n", N_Writes_Total);

	return 0;
}
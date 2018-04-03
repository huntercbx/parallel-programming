////////////////////////////////////////////////////////////////////////////////
// Справедливое решение проблемы читателей-писателей
// (каждый поток должен получить доступ за конечное время)
//
// Процесс-читатель проверяет валидность данных при чтении (все элементы буфера
// содержат одинаковый символ), в случае ошибки - выводится сообщение (ошибок
// быть не должно).
//
// Процесс-писатель просто обновляет данные в буфере.
//
// По результатам работы программы подсчитывается количество произведенных
// операций чтения/записи за время TOTAL_TIME_MS
////////////////////////////////////////////////////////////////////////////////

#include <windows.h>
#include <stdio.h>
#include <climits>

const unsigned int N_READERS        = 4;            // Количество задач-читателей
const unsigned int N_WRITERS        = 4;            // Количество задач-писателей
const DWORD	TOTAL_TIME_MS           = 5000;         // Общее время работы программы
const DWORD	READ_TIME_MS            = 10;            // Время чтения в мс (эмуляция долгих операций)
const DWORD	WRITE_TIME_MS           = 10;            // Время записи в мс (эмуляция долгих операций)

const unsigned int BUFFER_SIZE      = 20;           // Размер буфера
char buffer[BUFFER_SIZE] = "AAAAAAAAAAAAAAAAAAA";   // Буфер - играет роль общей области памяти

bool abort_all_threads              = false;        // Признак завершения работы всех потоков
LONG volatile read_count            = 0;            // Количество потоков-читателей имеющих доступ на чтение к ресурсу

HANDLE	hNoWaitingSemaphore;                        // Дескриптор семафора для синхронизации ожидающих потоков
HANDLE	hNoAccessingSemaphore;                      // Дескриптор семафора для разграничения доступа к буферу

HANDLE	hThreadArray[N_READERS + N_WRITERS];        // Дескрипторы потоков
DWORD	dwReadersThreadID[N_READERS];               // Идентификаторы потоков-читателей
DWORD	dwWritersThreadID[N_WRITERS];               // Идентификаторы потоков-писателей

unsigned int N_Reads[N_READERS];                    // Количество произведенных операций чтения
unsigned int N_Writes[N_WRITERS];                   // Количество произведенных операций записи

// Функция потока-писателя
DWORD WINAPI WriterThreadFunction(LPVOID lpParam) 
{
	unsigned int *Num_Writes = reinterpret_cast<unsigned int*>(lpParam);

	while (!abort_all_threads)
	{
		// ждем когда дойдет очередь
		DWORD dwWaitResult = WaitForSingleObject( 
			hNoWaitingSemaphore,        // дескриптор семафора
			INFINITE);                  // ждем в течении бесконечного времени

		// проверка на ошибку
		if (dwWaitResult != WAIT_OBJECT_0)
			return 1;

		// ждем когда дойдет ресурс будет свободен
		dwWaitResult = WaitForSingleObject( 
			hNoAccessingSemaphore,      // дескриптор семафора
			INFINITE);                  // ждем в течении бесконечного времени

		// проверка на ошибку
		if (dwWaitResult != WAIT_OBJECT_0)
			return 1;

		// освобождаем семафор ожидания
		ReleaseSemaphore(hNoWaitingSemaphore, 1, NULL);

		////////////////////////////////////////////////////////////////////////
		/// ЗАПИСЬ ДАННЫХ
		////////////////////////////////////////////////////////////////////////
		{
			// записываем данные в память
			char ch = buffer[0] == 'Z' ? 'A' :  buffer[0] + 1;
			for (unsigned int i = 0; i < BUFFER_SIZE - 1; ++i)
				buffer[i] = ch;

			// увеличиваем счетчик операций записи
			++(*Num_Writes);

			// Эмулируем продолжительность операции записи
			if (WRITE_TIME_MS > 0)
				Sleep(WRITE_TIME_MS);
		}
		////////////////////////////////////////////////////////////////////////

		// освобождаем семафор для доступа к ресурсу
		ReleaseSemaphore(hNoAccessingSemaphore, 1, NULL);
	}

	return 0;
}

// Функция потока-читателя
DWORD WINAPI ReaderThreadFunction(LPVOID lpParam)
{
	unsigned int *Num_Reads = reinterpret_cast<unsigned int*>(lpParam);

	char local_buffer[BUFFER_SIZE];

	while (!abort_all_threads)
	{
		// ждем когда дойдет очередь
		DWORD dwWaitResult = WaitForSingleObject( 
			hNoWaitingSemaphore,        // дескриптор семафора
			INFINITE);                  // ждем в течении бесконечного времени

		// проверка на ошибку
		if (dwWaitResult != WAIT_OBJECT_0)
			return 1;

		// увеличиваем счетчик потоков-читателей имеющих доступ на чтение к ресурсу
		LONG res = InterlockedIncrement(&read_count);

		// если это первый поток читатель, то блокируем доступ к ресурсу
		if (res == 1)
		{
			// ждем когда дойдет ресурс будет свободен
			DWORD dwWaitResult = WaitForSingleObject( 
				hNoAccessingSemaphore,      // дескриптор семафора
				INFINITE);                  // ждем в течении бесконечного времени

			// проверка на ошибку
			if (dwWaitResult != WAIT_OBJECT_0)
			{
				ReleaseSemaphore(hNoWaitingSemaphore, 1, NULL);
				return 1;
			}
		}

		// освобождаем семафор ожидания
		ReleaseSemaphore(hNoWaitingSemaphore, 1, NULL);

		//////////////////////////////////////////////////////////////////////////////
		/// ЧТЕНИЕ ДАННЫХ
		//////////////////////////////////////////////////////////////////////////////
		{
			// читаем данные из памяти
			bool error = false;
			for (unsigned int i = 0; i < BUFFER_SIZE; ++i)
			{
				local_buffer[i] = buffer[i];
				error |= i > 0 && i < BUFFER_SIZE-1 && local_buffer[i] != local_buffer[i-1];
			}

			// если была ошибка, выводим сообщение об ошибке
			if (error)
				printf_s("Reader thread %4d detect buffer corruption: %s\n", GetCurrentThreadId(), local_buffer);
			
			// увеличиваем счетчик операций чтения
			++(*Num_Reads);

			// Эмулируем продолжительность операции чтения
			if (READ_TIME_MS > 0)
				Sleep(READ_TIME_MS);
		}
		//////////////////////////////////////////////////////////////////////////////

		// уменьшаем счетчик потоков-читателей имеющих доступ на чтение к ресурсу
		res = InterlockedDecrement(&read_count);

		// Если это последний поток-читатель, то освобождаем ресурс для доступа
		if (res == 0)
		{
			// освобождаем семафор для доступа к ресурсу
			ReleaseSemaphore(hNoAccessingSemaphore, 1, NULL);
		}
	}
	return 0;
}

int main(int argc, char* argv[])
{
	// Создаем семафор для синхронизации ожидающих потоков
	hNoWaitingSemaphore = CreateSemaphore(
		NULL,                       // использовать настройки безопасности по умолчанию
		1,                          // первоначальное значение семафора
		1,                          // максимальное значение семафора
		NULL                        // создать объект без имени
	);

	// Создать семафор не удалось - выводим сообщение об ошибке и завершаем программу
	if (hNoWaitingSemaphore == NULL)
	{
		printf_s("CreateSemaphore failed (%d)\n", GetLastError());
		ExitProcess(1);
	}

	// Создаем семафор для разграничения доступа к буферу
	hNoAccessingSemaphore = CreateSemaphore(
		NULL,                       // использовать настройки безопасности по умолчанию
		1,                          // первоначальное значение семафора
		1,                          // максимальное значение семафора
		NULL                        // создать объект без имени
	);

	// Создать семафор не удалось - выводим сообщение об ошибке и завершаем программу
	if (hNoAccessingSemaphore == NULL)
	{
		printf_s("CreateSemaphore failed (%d)\n", GetLastError());
		ExitProcess(1);
	}

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
			&dwReadersThreadID[i]); // идентификатор потока

		// Если поток не был создан - завершаем работу с ошибкой
		if (hThreadArray[i] == NULL)
			ExitProcess(1);

		// Выводим сообщение об успешном создании потока
		printf_s("Reader thread %4d sucsessfully created\n", dwReadersThreadID[i]);
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
			&dwWritersThreadID[i]); // идентификатор потока

		// Если поток не был создан - завершаем работу с ошибкой
		if (hThreadArray[N_READERS + i] == NULL) 
			ExitProcess(1);

		// Выводим сообщение об успешном создании потока
		printf_s("Writer thread %4d sucsessfully created\n", dwWritersThreadID[i]);
	}

	// Ждем 5 секунд для получения результатов и выставляем флаг для завершения всех потоков
	Sleep(5000);
	abort_all_threads = true;

	// Ожидаем завершения всех потоков
	WaitForMultipleObjects(
		N_READERS + N_WRITERS,  // общее количество потоков
		hThreadArray,           // дескрипторы потоков
		TRUE,                   // ждем завершения всех потоков
		INFINITE);              // ждем в течении бесконечного времени

	// Освобождаем дескрипторы
	CloseHandle(hNoWaitingSemaphore);
	CloseHandle(hNoAccessingSemaphore);
	for (unsigned int i = 0; i < N_READERS + N_WRITERS; ++i)
		CloseHandle(hThreadArray[i]);

	// Подсчитываем общее количество произведенных операций чтения
	unsigned int N_Reads_Total = 0;
	for (unsigned int i = 0; i < N_READERS; ++i)
	{
		printf_s("Number of reads by thread %4d: %d\n", dwReadersThreadID[i], N_Reads[i]);
		N_Reads_Total += N_Reads[i];
	}
	printf_s("Total number of reads: %d\n", N_Reads_Total);

	// Подсчитываем общее количество произведенных операций чтения
	unsigned int N_Writes_Total = 0;
	for (unsigned int i = 0; i < N_WRITERS; ++i)
	{
		printf_s("Number of writes by thread %4d: %d\n", dwWritersThreadID[i], N_Writes[i]);
		N_Writes_Total += N_Writes[i];
	}
	printf_s("Total number of writes: %d\n", N_Writes_Total);

	return 0;
}

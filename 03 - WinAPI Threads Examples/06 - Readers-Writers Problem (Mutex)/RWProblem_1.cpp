////////////////////////////////////////////////////////////////////////////////
/// Решение проблемы читателей-писателей с использованием мьютекса
///
///	В данном примере используется мьютекс для ограничения доступа к ресурсу.
///	При этом читатели и писатели имеют одинаковые права доступа к ресурсу.
///	В качестве разделяемого ресурса используется обычный буфер.
///
/// Процесс-читатель проверяет валидность данных при чтении (все элементы буфера
/// содержат одинаковый символ), в случае ошибки - выводится сообщение (ошибок
/// быть не должно).
///
/// Процесс-писатель просто обновляет данные в буфере.
/// 
/// По результатам работы программы подсчитывается количество произведенных
/// операций чтения/записи за время TOTAL_TIME_MS
///
/// С данным примером можно провести следующие эксперименты:
///	1. Не освобождать мьютекс (не вызывать ReleaseMutex) в одном из потоков
///	2. Завершить работу одного потока без освобождения мьютекса (добавить return
///		непосредственно перед функцией ReleaseMutex)
////////////////////////////////////////////////////////////////////////////////

#include <windows.h>
#include <stdio.h>
#include <climits>

const unsigned int N_READERS		= 4;			// Количество задач-читателей
const unsigned int N_WRITERS		= 4;			// Количество задач-писателей
const DWORD	MUTEX_WAIT_TIME_MS		= 100;			// Время ожидания мьютекса в мс
const DWORD	TOTAL_TIME_MS			= 5000;			// Общее время работы программы
const DWORD	READ_TIME_MS			= 1;			// Время чтения в мс (эмуляция долгих операций)
const DWORD	WRITE_TIME_MS			= 1;			// Время записи в мс (эмуляция долгих операций)

const unsigned int BUFFER_SIZE		= 20;			// Размер буфера
char buffer[BUFFER_SIZE] = "AAAAAAAAAAAAAAAAAAA";	// Буфер - играет роль общей области памяти

bool abort_all_threads				= false;		// Признак завершения работы всех потоков

HANDLE hMutex;										// Дескриптор мьютекса
HANDLE hThreadArray[N_READERS + N_WRITERS];			// Дескрипторы потоков
DWORD	dwReadersThreadID[N_READERS];				// Идентификаторы потоков-читателей
DWORD	dwWritersThreadID[N_WRITERS];				// Идентификаторы потоков-писателей

unsigned int N_Reads[N_READERS];					// Количество произведенных операций чтения
unsigned int N_Writes[N_WRITERS];					// Количество произведенных операций записи

// Функция потока-писателя
DWORD WINAPI WriterThreadFunction(LPVOID lpParam) 
{
	unsigned int *Num_Writes = reinterpret_cast<unsigned int*>(lpParam);

	while (!abort_all_threads)
	{
		// пытаемся захватить мьютекс
		DWORD dwWaitResult = WaitForSingleObject( 
            hMutex,					// дескриптор мьютекса
			MUTEX_WAIT_TIME_MS);	// ждем в течении указанного времени

        switch (dwWaitResult) 
        {
            // Мьютекс успешно захвачен
            case WAIT_OBJECT_0: 
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

					// освобождаем мьютекс
					ReleaseMutex(hMutex);
				}
				break;

            // Истекло время ожидания
			case WAIT_TIMEOUT:
				printf_s("Writer thread %4d can not lock mutex during %d ms\n", GetCurrentThreadId(), MUTEX_WAIT_TIME_MS);
				break;

			// Один из процессов завершился не освободив мьютекс
			case WAIT_ABANDONED:
				printf_s("Mutex abandoned\n");
				return 1;

			// Системная ощибка
			case WAIT_FAILED:
				return 2;
		}
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
		// пытаемся захватить мьютекс
		DWORD dwWaitResult = WaitForSingleObject( 
            hMutex,					// дескриптор мьютекса
			MUTEX_WAIT_TIME_MS);	// ждем в течении указанного времени
 
        switch (dwWaitResult) 
        {
            // Мьютекс успешно захвачен
            case WAIT_OBJECT_0: 
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

					// освобождаем мьютекс
					ReleaseMutex(hMutex);
				}
				break;

            // Истекло время ожидания
			case WAIT_TIMEOUT:
				printf_s("Reader thread %4d can not lock mutex during %d ms\n", GetCurrentThreadId(), MUTEX_WAIT_TIME_MS);
				break;

			// Один из процессов завершился не освободив мьютекс
			case WAIT_ABANDONED:
				printf_s("Mutex abandoned\n");
				return 1;

			// Системная ощибка
			case WAIT_FAILED:
				return 2;
		}
	}
    return 0;
}

int main(int argc, char* argv[])
{
    // Создаем мьютекс
    hMutex = CreateMutex( 
        NULL,						// использовать настройки безопасности по умолчанию
        FALSE,						// создать без захвата мьютекса текущим потоком
        NULL						// создать объект без имени
        ); 

	// Создать мьютекс не удалось - выводим сообщение об ошибке и завершаем программу
    if (hMutex == NULL) 
    { 
        printf_s("CreateMutex failed (%d)\n", GetLastError());
        ExitProcess(1);
    }

	// Создание потоков-читателей
	for (unsigned int i = 0; i < N_READERS; ++i)
	{
		// инициализируем нулем количество операций чтения, произведенных потоком
		N_Reads[i] = 0;

		hThreadArray[i] = CreateThread(
			NULL,					// использовать настройки безопасности по умолчанию
			0,						// использовать ращмер стека по умолчанию
			ReaderThreadFunction,	// имя функции потока
			&N_Reads[i],			// аргумент - количество операций чтения, произведенных потоком
			0,						// создать поток с флагами по умолчанию
			&dwReadersThreadID[i]);	// идентификатор потока

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
			NULL,					// использовать настройки безопасности по умолчанию
			0,						// использовать ращмер стека по умолчанию
			WriterThreadFunction,	// имя функции потока
			&N_Writes[i],			// аргумент - количество операций записи, произведенных потоком
			0,						// создать поток с флагами по умолчанию
			&dwWritersThreadID[i]);	// идентификатор потока

		// Если поток не был создан - завершаем работу с ошибкой
		if (hThreadArray[N_READERS + i] == NULL) 
			ExitProcess(1);

		// Выводим сообщение об успешном создании потока
		printf_s("Writer thread %4d sucsessfully created\n", dwWritersThreadID[i]);
	}

	// Ждем 5 секунд для получения результатов и выставляем флаг для завершения всех потоков
	Sleep(TOTAL_TIME_MS);
	abort_all_threads = true;

    // Ожидаем завершения всех потоков
    WaitForMultipleObjects(
		N_READERS + N_WRITERS,	// общее количество потоков
		hThreadArray,			// дескрипторы потоков
		TRUE,					// ждем завершения всех потоков
		INFINITE);				// ждем в течении бесконечного времени

    // Освобождаем дескрипторы
	CloseHandle(hMutex);
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

////////////////////////////////////////////////////////////////////////////////
/// Демонстрация проблемы читателей-писателей
///
/// В данном примере блокировки не используются, читатели и писатели имеют
/// одинаковые права доступа к ресурсу. В качестве разделяемого ресурса
/// используется обычный буфер
///
/// Процесс-читатель проверяет валидность данных при чтении (все элементы буфера
/// содержат одинаковый символ), при достижении MAX_ERRORS ошибок чтения из
/// буфера процесс завершается.
///
/// Процесс-писатель обновляет данные в буфере без каких-либо проверок.
////////////////////////////////////////////////////////////////////////////////

#include <windows.h>
#include <stdio.h>
#include <climits>
#include <ctime>

const unsigned int N_READERS        = 4;            // Количество задач-читателей
const unsigned int N_WRITERS        = 1;            // Количество задач-писателей
const DWORD	TOTAL_TIME_MS           = 5000;         // Общее время работы программы
const DWORD	READ_TIME_MS            = 1;            // Время чтения в мс (эмуляция долгих операций)
const DWORD	WRITE_TIME_MS           = 1;            // Время записи в мс (эмуляция долгих операций)

const unsigned int BUFFER_SIZE      = 20;           // Размер буфера
char buffer[BUFFER_SIZE] = "AAAAAAAAAAAAAAAAAAA";   // Буфер - играет роль общей области памяти

bool abort_all_threads              = false;        // Признак завершения работы всех потоков

HANDLE hThreadArray[N_READERS + N_WRITERS];         // Дескрипторы потоков
LARGE_INTEGER PERFORMANCE_COUNTER_FREQUENCY;        // Частота таймера для измерения производительности

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
void WriteOperation()
{
	// записываем данные в память
	char ch = buffer[0] == 'Z' ? 'A' : buffer[0] + 1;
	for (unsigned int i = 0; i < BUFFER_SIZE - 1; ++i)
		buffer[i] = ch;

	// задержка для эмуляции длительности операции
	DelayMS(WRITE_TIME_MS);
}

////////////////////////////////////////////////////////////////////////////////
// Эмуляция операции чтения с проверкой целостности буфера
////////////////////////////////////////////////////////////////////////////////
bool ReadOperation()
{
	char local_buffer[BUFFER_SIZE];

	// читаем данные из памяти
	bool error = false;
	for (unsigned int i = 0; i < BUFFER_SIZE; ++i)
	{
		local_buffer[i] = buffer[i];
		error |= i > 0 && i < BUFFER_SIZE -1 && local_buffer[i] != local_buffer[i - 1];
	}

	// если была ошибка, выводим сообщение об ошибке и возвращаем
	if (error)
	{
		printf_s("Reader thread %4d detect buffer corruption: %s\n",
			GetCurrentThreadId(), local_buffer);

		return false;
	}

	// задержка для эмуляции длительности операции
	DelayMS(READ_TIME_MS);

	return true;
}

////////////////////////////////////////////////////////////////////////////////
// Функция потока-писателя
////////////////////////////////////////////////////////////////////////////////
DWORD WINAPI WriterThreadFunction(LPVOID lpParam) 
{
	while (!abort_all_threads)
		WriteOperation();

	return 0;
}

////////////////////////////////////////////////////////////////////////////////
// Функция потока-читателя
////////////////////////////////////////////////////////////////////////////////
DWORD WINAPI ReaderThreadFunction(LPVOID lpParam)
{
	unsigned int n_errors = 0;

	while (!abort_all_threads)
		ReadOperation();

	return 0;
}

////////////////////////////////////////////////////////////////////////////////
// 
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
	// Получаем частоту таймера для измерения производительности
	QueryPerformanceFrequency(&PERFORMANCE_COUNTER_FREQUENCY);

	DWORD dwThreadID;

	// Создание потоков-читателей
	for (unsigned int i = 0; i < N_READERS; ++i)
	{
		hThreadArray[i] = CreateThread(
			NULL,                   // использовать настройки безопасности по умолчанию
			0,                      // использовать ращмер стека по умолчанию
			ReaderThreadFunction,   // имя функции потока
			NULL,                   // без аргументов
			0,                      // создать поток с флагами по умолчанию
			&dwThreadID);           // идентификатор потока

		// Если поток не был создан - завершаем работу с ошибкой
		if (hThreadArray[i] == NULL) 
			ExitProcess(1);

		// Выводим сообщение об успешном создании потока
		printf_s("Reader thread %4d sucsessfully created\n", dwThreadID);
	}

	// Создание потоков-писатлей
	for (unsigned int i = 0; i < N_WRITERS; ++i)
	{
		hThreadArray[N_READERS + i] = CreateThread(
			NULL,                   // использовать настройки безопасности по умолчанию
			0,                      // использовать ращмер стека по умолчанию
			WriterThreadFunction,   // имя функции потока
			NULL,                   // без аргументов
			0,                      // создать поток с флагами по умолчанию
			&dwThreadID);           // идентификатор потока

		// Если поток не был создан - завершаем работу с ошибкой
		if (hThreadArray[N_READERS + i] == NULL) 
			ExitProcess(1);

		// Выводим сообщение об успешном создании потока
		printf_s("Writer thread %4d sucsessfully created\n", dwThreadID);
	}

	// Ждем указанное время для получения результатов
	Sleep(TOTAL_TIME_MS);
	// По истечении времени выставляем флаг для завершения всех потоков
	abort_all_threads = true;

	// Ожидаем завершения всех потоков
	WaitForMultipleObjects(
		N_READERS + N_WRITERS,      // общее количество потоков
		hThreadArray,               // дескрипторы потоков
		TRUE,                       // ждем завершения всех потоков
		INFINITE);                  // ждем в течении бесконечного времени

	// Освобождаем дескрипторы
	for (unsigned int i = 0; i < N_READERS + N_WRITERS; ++i)
		CloseHandle(hThreadArray[i]);

	return 0;
}

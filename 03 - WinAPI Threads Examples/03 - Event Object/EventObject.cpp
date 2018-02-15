////////////////////////////////////////////////////////////////////////////////
/// Демонстрация использования объекта "событие"
///
/// Помимо основного потока создаются два дополнительных потока.
/// Первый поток ищет минамльные элемент в исходном наборе данных, а второй -
/// максимальный элемент. Генерацией исходных данных занимается основной поток,
/// событие используется для информирования других потоков об окончании этапа
/// генерации данных.
/// 
/// Для демонстрации можно убрать ожидание события в потоках и убедится что
/// получается другой (невалидный) результат.
////////////////////////////////////////////////////////////////////////////////

#include <windows.h>
#include <stdio.h>
#include <climits>

const unsigned int BUFFER_SIZE = 10000000;

float Buffer[BUFFER_SIZE];  // буфер
HANDLE hEvent;              // дескриптор события
HANDLE hThreadArray[2];     // дескрипторы потоков

////////////////////////////////////////////////////////////////////////////////
// Функция потока, отвечающего за поиск минимума
////////////////////////////////////////////////////////////////////////////////
DWORD WINAPI MinThreadFunction(LPVOID lpParam)
{
	printf_s("Thread %d waiting for event...\n", GetCurrentThreadId());

	DWORD dwWaitResult = WaitForSingleObject(
		hEvent,         // дескриптор ожидаемого события
		INFINITE);      // ждем в течении бесконечного времени

	switch (dwWaitResult)
	{
	// Событие произошло
	case WAIT_OBJECT_0:
		printf_s("Thread %d reading data\n", GetCurrentThreadId());
		break;

	// Произошла ошибка
	default:
		printf_s("Wait error (%d)\n", GetLastError());
		return 0;
	}

	unsigned int min_index = 0;
	float min_value = Buffer[0];
	for (unsigned int i = 0; i < BUFFER_SIZE; ++i)
	{
		if (Buffer[i] < min_value)
		{
			min_value = Buffer[i];
			min_index = i;
		}
	}

	printf_s("Thread %d exiting, find min element %g at %d\n",
		GetCurrentThreadId(), min_value, min_index);
	return 0;
}

////////////////////////////////////////////////////////////////////////////////
// Функция потока, отвечающего за поиск максимума
////////////////////////////////////////////////////////////////////////////////
DWORD WINAPI MaxThreadFunction(LPVOID lpParam)
{
	printf_s("Thread %d waiting for event...\n", GetCurrentThreadId());

	DWORD dwWaitResult = WaitForSingleObject(
		hEvent,         // дескриптор ожидаемого события
		INFINITE);      // ждем в течении бесконечного времени

	switch (dwWaitResult)
	{
	// Событие произошло
	case WAIT_OBJECT_0:
		printf_s("Thread %d reading data\n", GetCurrentThreadId());
		break;

	// Произошла ошибка
	default:
		printf_s("Wait error (%d)\n", GetLastError());
		return 0;
	}

	unsigned int max_index = 0;
	float max_value = Buffer[0];
	for (unsigned int i = 0; i < BUFFER_SIZE; ++i)
	{
		if (Buffer[i] > max_value)
		{
			max_value = Buffer[i];
			max_index = i;
		}
	}

	printf_s("Thread %d exiting, find max element %g at %d\n",
		GetCurrentThreadId(), max_value, max_index);
	return 0;
}

////////////////////////////////////////////////////////////////////////////////
// Функция создания массива исходных данных
////////////////////////////////////////////////////////////////////////////////
void CreateData()
{
	printf_s("Main thread creating data...\n");

	for (unsigned int i = 0; i < BUFFER_SIZE; ++i)
		Buffer[i] = rand() / static_cast<float>(RAND_MAX) +
		            rand() / static_cast<float>(RAND_MAX);

	// Сигнализируем об окончании генерации исходных данных
	if (!SetEvent(hEvent))
	{
		printf_s("SetEvent failed (%d)\n", GetLastError());
		return;
	}
}

int main(int argc, char* argv[])
{
	// Создаем событие. Основной поток просигнализирует о событии по завершении генерации данных
	hEvent = CreateEvent(
		NULL,                   // использовать настройки безопасности по умолчанию
		TRUE,                   // событие с ручным сбросом состояния
		FALSE,                  // начальное состояние события
		NULL                    // создать объект без имени
	);

	// Создать событие не удалось - выводим сообщение об ошибке и завершаем программу
	if (hEvent == NULL)
	{
		printf_s("CreateEvent failed (%d)\n", GetLastError());
		ExitProcess(1);
	}

	DWORD dwThreadID;

	// Создание потока для поиска минимума
	hThreadArray[0] = CreateThread(
		NULL,                   // использовать настройки безопасности по умолчанию
		0,                      // использовать ращмер стека по умолчанию
		MinThreadFunction,      // имя функции потока
		NULL,                   // без аргументов
		0,                      // создать поток с флагами по умолчанию
		&dwThreadID);           // идентификатор потока

	// Если поток не был создан - завершаем работу с ошибкой
	if (hThreadArray[0] == NULL)
		ExitProcess(1);

	// Создание потока для поиска максимума
	hThreadArray[1] = CreateThread(
		NULL,                   // использовать настройки безопасности по умолчанию
		0,                      // использовать ращмер стека по умолчанию
		MaxThreadFunction,      // имя функции потока
		NULL,                   // без аргументов
		0,                      // создать поток с флагами по умолчанию
		&dwThreadID);           // идентификатор потока

	// Если поток не был создан - завершаем работу с ошибкой
	if (hThreadArray[1] == NULL)
		ExitProcess(1);

	// Генерируем исходные данные
	CreateData();

	printf_s("Main thread waiting for threads to exit...\n");

	// Ожидаем завершения всех потоков
	WaitForMultipleObjects(
		2,                      // количество потоков
		hThreadArray,           // дескрипторы потоков
		TRUE,                   // ждем завершения всех потоков
		INFINITE);              // ждем в течении бесконечного времени

	// Освобождаем дескрипторы
	CloseHandle(hEvent);
	CloseHandle(hThreadArray[0]);
	CloseHandle(hThreadArray[1]);

	return 0;
}

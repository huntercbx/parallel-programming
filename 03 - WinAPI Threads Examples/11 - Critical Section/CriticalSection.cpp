////////////////////////////////////////////////////////////////////////////////
/// Демонстрация использования критических секций
///
/// Помимо основного потока создаются несколько дополнительных потоков. Задачей
/// дополнительных потоков является подсчет количества простых чисел в указанном
/// диапазоне.
///
/// Необходимо обратить внимание на использование модификатор volatile перед
/// переменной nPrimeNumbers
///
/// Для эксперимента можно убрать вход в критическую секцию и выход из нее и
/// запустить программу несколько раз - в этом случае программа может выдать
/// недостоверные (различающиеся) результаты.
///
/// Также следует обратить внимание на альтернативное решение с использованием
/// атомарной операции.
////////////////////////////////////////////////////////////////////////////////

#include <windows.h>
#include <stdio.h>
#include <climits>
#include <cmath>
#include <ctime>

const unsigned int  N_THREADS       = 3;            // Количество потоков
LONG volatile       nPrimeNumbers   = 0;            // Количество найденных простых чисел
CRITICAL_SECTION    cs;                             // Критическая секция

// Структура передаваемая в поток в качестве параметра
struct SearchParams
{
	unsigned long start;    // начало интервала
	unsigned long end;      // конец интервала
};

// Функция потока, отвечающего за поиск минимума
DWORD WINAPI ThreadFunction(LPVOID lpParam)
{
	// Приведение указателя на данные к нужному типу
	SearchParams* pSearchParams = reinterpret_cast<SearchParams*>(lpParam);
	printf_s("Thread %6d, Interval [%5d, %5d]\n",
		GetCurrentThreadId(),
		pSearchParams->start,
		pSearchParams->end);
	
	for (unsigned long number = pSearchParams->start; number <= pSearchParams->end; ++number)
	{
		bool is_prime = true;
		unsigned long last = (long)floor(sqrt((double)number));
		for (unsigned long j = 2; j <= last && is_prime; ++j)
		{
			if ((number % j) == 0)
				is_prime = false;
		}
		if (is_prime)
		{
			// использование критической секции для доступа к общей переменной
			EnterCriticalSection(&cs);
			++nPrimeNumbers;
			LeaveCriticalSection(&cs);

			// альтернативное решение проблемы с использованием атомарной операции
			//InterlockedIncrement(&nPrimeNumbers);
		}
	}

	return 0;
}

int main(int argc, char* argv[])
{
	// проверка параметров командной строки
	if (argc != 3)
	{
		printf_s("Usage: %s min_number max_number\n", argv[0]);
		return 0;
	}

	unsigned long min_number = atol(argv[1]);
	unsigned long max_number = atol(argv[2]);
	if (min_number > max_number)
	{
		printf_s("Invalid parameter: min_number can not be greater then max_number\n");
		return 0;
	}

	// выводим параметры запуска программы
	printf_s("Searching prime numbers in interval [%lu, %lu]\n", min_number, max_number);

	// Инициализируем критическую секцию
	if (!InitializeCriticalSectionAndSpinCount(
		&cs,                        // указатель на критическую секцию
		1000                        // количество циклов ожидания
		))
	{
		printf_s("InitializeCriticalSectionAndSpinCount failed (%d)\n", GetLastError());
		ExitProcess(1);
	}

	clock_t t1 = clock();           // начало работы программы

	SearchParams    data[N_THREADS];                // данные для работы потоков
	DWORD           dwThreadIdArray[N_THREADS];     // идентификаторы потоков
	HANDLE          hThreadArray[N_THREADS];        // дескрипторы потоков

	// количество чисел для проверки одним потоком
	const unsigned int N = (max_number - min_number)/N_THREADS;

	// Создание требуемого (N_THREADS) числа потоков
	for (unsigned int i = 0; i < N_THREADS; ++i)
	{
		// распределение задач между потоками
		data[i].start = min_number + i*N;
		data[i].end = (i == N_THREADS-1) ? max_number : (data[i].start + N - 1);

		// Создание нового потока
		hThreadArray[i] = CreateThread(
			NULL,                   // использовать настройки безопасности по умолчанию
			0,                      // использовать ращмер стека по умолчанию
			ThreadFunction,         // имя функции потока
			&data[i],               // аргумент для потока
			0,                      // создать поток с флагами по умолчанию
			&dwThreadIdArray[i]);   // идентификатор потока

		// Если поток не был создан - завершаем работу с ошибкой
		if (hThreadArray[i] == NULL)
			ExitProcess(1);
	}

	// Ожидаем завершения всех потоков
	WaitForMultipleObjects(
		N_THREADS,                  // количество потоков
		hThreadArray,               // дескрипторы потоков
		TRUE,                       // ждем завершения всех потоков
		INFINITE);                  // ждем в течении бесконечного времени

	// Освобождаем дескрипторы потоков
	for (int i = 0; i < N_THREADS; ++i)
		CloseHandle(hThreadArray[i]);

	// удаляем критическую функцию
	DeleteCriticalSection(&cs);

	clock_t t2 = clock();           // завершение работы программы

	// вывод результатов
	printf_s("Total number of prime numbers found: %d\n", nPrimeNumbers);

	// вывод затраченного времени
	printf_s("Execution time : %.5f s\n", (float)(t2-t1)/CLOCKS_PER_SEC);

	return 0;
}

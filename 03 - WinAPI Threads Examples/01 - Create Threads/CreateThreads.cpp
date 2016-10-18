////////////////////////////////////////////////////////////////////////////////
/// Демонстрация создания многопочного приложения
///
/// Создается несколько потоков, помимо основного. Для каждого потока выводится
/// его идентификатор и идентификатор процесса, которому поток принадлежит.
/// Идентификаторы потоков должны быть разные, а процесса - одинаковый.
///
/// В данном примере демонстрируется передача данных в поток, для чего
/// используется структура MyStruct.
/// Каждому потоку передается свой экземпляр структуры.
///
/// Следует обратить внимание:
///   - что основной поток продолжает работу после создания остальных потоков;
///   - на использование функции WaitForMultipleObjects для ожидания завершения
///     всех потоков.
////////////////////////////////////////////////////////////////////////////////

#include <windows.h>
#include <stdio.h>

// Количество создаваемых потоков
const int N_THREADS = 8;

// Пример структуры данных, передаваемой в поток
struct MyStruct
{
	int val1;
	int val2;
};

// Функция потока
DWORD WINAPI MyThreadFunction(LPVOID lpParam)
{
	// Приведение указателя на данные к нужному типу
	MyStruct* pMyStruct = reinterpret_cast<MyStruct*>(lpParam);

	// Вывод идентификаторов текущего процесса, потока и параметров
	printf_s(
		"Process ID = %d, Thread ID = %d, Parameters = %d, %d\n",
		GetCurrentProcessId(),
		GetCurrentThreadId(),
		pMyStruct->val1,
		pMyStruct->val2);
	return 0;
}

int main(int argc, char* argv[])
{
	// Вывод идентификаторов текущего процесса и потока
	printf_s("Process ID = %d, Thread ID = %d, This is main thread\n",
		GetCurrentProcessId(),
		GetCurrentThreadId());

	MyStruct    data[N_THREADS];                // данные для работы потоков
	DWORD       dwThreadIdArray[N_THREADS];     // идентификаторы потоков
	HANDLE      hThreadArray[N_THREADS];        // дескрипторы потоков

	// Создание требуемого (N_THREADS) числа потоков
	for (int i = 0; i < N_THREADS; ++i)
	{
		// Заполнение полей структуры, передаваемой в поток
		data[i].val1 = i;
		data[i].val2 = i * 101;

		// Создание нового потока
		hThreadArray[i] = CreateThread(
			NULL,                   // использовать настройки безопасности по умолчанию
			0,                      // использовать размер стека по умолчанию
			MyThreadFunction,       // имя функции потока
			&data[i],               // аргумент для потока
			0,                      // создать поток с флагами по умолчанию
			&dwThreadIdArray[i]);   // идентификатор потока

		// Если поток не был создан - завершаем работу с ошибкой
		if (hThreadArray[i] == NULL)
			ExitProcess(1);
	}

	// Основной поток продолжает работу
	printf_s("Process ID = %d, Thread ID = %d, Main thread is still running!\n",
		GetCurrentProcessId(),
		GetCurrentThreadId());

	// Ожидаем завершения всех потоков
	WaitForMultipleObjects(
		N_THREADS,                  // количество потоков
		hThreadArray,               // дескрипторы потоков
		TRUE,                       // ждем завершения всех потоков
		INFINITE);                  // ждем в течении бесконечного времени

	// Освобождаем дескрипторы потоков
	for (int i = 0; i < N_THREADS; ++i)
		CloseHandle(hThreadArray[i]);

	return 0;
}

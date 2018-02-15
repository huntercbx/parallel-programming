////////////////////////////////////////////////////////////////////////////////
/// Данная программа пытается создать максимально возможное число потоков.
///
/// Следует провести несколько экспериментов с различным размером стека и
/// проанализировать результаты.
////////////////////////////////////////////////////////////////////////////////

#include <windows.h>
#include <stdio.h>

// размер стека у каждого потока в байтах
const int STACK_SIZE = 4 * 1024 * 1024;

////////////////////////////////////////////////////////////////////////////////
// Функция вывода текстового сообщения, соответсвующего коду ошибки
////////////////////////////////////////////////////////////////////////////////
void PrintErrorMessage(DWORD error)
{
	LPWSTR errorMessage = NULL;
	DWORD retCode = FormatMessage(
		FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
		NULL,
		error,
		MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
		(LPWSTR)&errorMessage,
		0,
		NULL);
	if (retCode != 0)
		wprintf_s(L"%s\n", errorMessage);
	else
		error = GetLastError();
	LocalFree(errorMessage);
}

////////////////////////////////////////////////////////////////////////////////
// Функция потока - эмулирует некий вычислительный процесс
//
// Поскольку потоки создаются с флагом CREATE_SUSPENDED, то на самом деле они
//   выполняться они не будут, а будут находиться в режиме ожидания -
//   ждать вызова функции ResumeThread(...)
////////////////////////////////////////////////////////////////////////////////
DWORD WINAPI MyThreadFunction(LPVOID lpParam)
{
	int n = rand();
	while (true)
		n = (n + 997) % 32469;
	return 0;
}

int main(int argc, char* argv[])
{
	size_t  nThreads = 0;                       // количество созданных потоков

	while (true)
	{
		DWORD   idThread;                       // идентификатор потока
		HANDLE  hThread;                        // дескриптор потока

		hThread = CreateThread(
			NULL,                               // использовать настройки безопасности по умолчанию
			STACK_SIZE,                         // размер стека
			MyThreadFunction,                   // имя функции потока
			NULL,                               // функция потока без аргументов
			CREATE_SUSPENDED |                  // не запускать поток на выполнение до вызова функции ResumeThread
			STACK_SIZE_PARAM_IS_A_RESERVATION,  // резервировать память под стек указанного размера
			&idThread);                         // идентификатор потока

		if (hThread == NULL)
		{
			DWORD error = GetLastError();
			printf_s("Last CreateThread(...) failed with error: %d\n", error);
			PrintErrorMessage(error);
			break;
		}

		++nThreads;
		if ((nThreads % 5000) == 0)
			printf_s("%6zu threads created so far...\n", nThreads);
	}
	printf_s("Total number of threads created: %zu\n", nThreads);

	ExitProcess(0);

	return 0;
}

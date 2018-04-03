////////////////////////////////////////////////////////////////////////////////
// Демонстрация проблемы обедающих философов
////////////////////////////////////////////////////////////////////////////////
#include <windows.h>
#include <stdio.h>
#include <ctime>

const DWORD TOTAL_TIME_MS        = 5000;    // Общее время работы программы (мс)
const DWORD FORK_PAUSE_MS        = 20;      // Пауза между взятием левой и правой вилки
const DWORD EATING_TIME_MS       = 100;     // Время приема пищи
const DWORD THINKING_TIME_MS     = 200;     // Минимальное время размышления
const DWORD MUTEX_WAIT_TIME_MS   = 1000;    // Время ожидания вилки до вывода сообщения

HANDLE hEventStart;                         // Сообщение об окончании создания всех потоков
HANDLE hEventStop;                          // Сообщение об окончании работы всех потоков

LARGE_INTEGER frequency;                    // Разрешение таймера
LARGE_INTEGER start;                        // Время запуска программы

// Информация о дескрипоторов мьютексов-вилок, передаваемая в поток
struct ForksInfo
{
	int     index;      // порядковый номер философа
	HANDLE  left;       // дескрипотор левой вилки-мьютекса
	HANDLE  right;      // дескрипотор правой вилки-мьютекса
};

////////////////////////////////////////////////////////////////////////////////
// Функция вывода ссобщения
////////////////////////////////////////////////////////////////////////////////
void write_log(int index, const char* message)
{
	LARGE_INTEGER current;
	if (QueryPerformanceCounter(&current) == FALSE)
		printf_s("QueryPerformanceCounter failed (%d)\n", GetLastError());

	double seconds = double(current.QuadPart - start.QuadPart);
	seconds /= frequency.QuadPart;
	printf_s("%8.6f [%5d] Philosopher %d %s\n", seconds, GetCurrentThreadId(), index, message);
}

////////////////////////////////////////////////////////////////////////////////
// Функция взятия вилки
////////////////////////////////////////////////////////////////////////////////
enum { LEFT = 0, RIGHT = 1};
bool take_fork(ForksInfo* info, int fork)
{
	// дескриптор мьютекса-вилки
	HANDLE hFork = (fork == LEFT) ?
		info->left : info->right;

	while (true)
	{
		// проверка на окончание работы потока
		DWORD dwWaitResult = WaitForSingleObject(
			hEventStop,     // дескриптор ожидаемого события
			0);             // просто проверяем без ожидания
		if (dwWaitResult == WAIT_OBJECT_0)
			return true;

		dwWaitResult = WaitForSingleObject(
			hFork,                  // дескриптор мьютекса вилки
			MUTEX_WAIT_TIME_MS);    // ждем в течении указанного времени

		if (dwWaitResult == WAIT_OBJECT_0)
			break;

		if (dwWaitResult == WAIT_TIMEOUT)
			write_log(info->index, (fork == LEFT) ?
				"can not take left fork" : "can not take right fork");
	}
	
	write_log(info->index, (fork == LEFT) ?
		"has taken left fork" : "has taken right fork");
	return false;
}

////////////////////////////////////////////////////////////////////////////////
// Функция потока-философа
////////////////////////////////////////////////////////////////////////////////
DWORD WINAPI PhilosopherThread(LPVOID lpParam)
{
	// Приведение указателя на данные к нужному типу
	ForksInfo* pForksInfo = reinterpret_cast<ForksInfo*>(lpParam);

	write_log(pForksInfo->index, "thread is created");

	DWORD dwWaitResult = WaitForSingleObject(
		hEventStart,    // дескриптор ожидаемого события
		INFINITE);      // ждем в течении бесконечного времени

	while (true)
	{
		// размышляем в течении заданного времени
		write_log(pForksInfo->index, "is thinking");
		Sleep(rand() % THINKING_TIME_MS);
		write_log(pForksInfo->index, "became hungry");

		// пытаемся взять левую вилку
		if (take_fork(pForksInfo, LEFT)) return 0;

		// пауза между взятиям левой и правой вилки
		Sleep(FORK_PAUSE_MS);

		// пытаемся взять правую вилку
		if (take_fork(pForksInfo, RIGHT)) return 0;

		// едим в течении заданного времени
		write_log(pForksInfo->index, "is eating");
		Sleep(rand() % EATING_TIME_MS);
		write_log(pForksInfo->index, "is not hungry anymore");

		// освобождаем левую вилку
		ReleaseMutex(pForksInfo->left);
		write_log(pForksInfo->index, "has released left fork");

		// пауза между освобождением левой и правой вилки
		Sleep(FORK_PAUSE_MS);

		// освобождаем правую вилку
		ReleaseMutex(pForksInfo->right);
		write_log(pForksInfo->index, "has released right fork");
	}
}

int main(int argc, char* argv[])
{
	if (QueryPerformanceFrequency(&frequency) == FALSE)
		printf_s("QueryPerformanceFrequency failed (%d)\n", GetLastError());
	printf_s("QueryPerformanceFrequency = %f\n", double(frequency.QuadPart));

	if (QueryPerformanceCounter(&start) == FALSE)
		printf_s("QueryPerformanceCounter failed (%d)\n", GetLastError());

	// Создаем событие. Основной поток просигнализирует о завершении создания всех потоков
	hEventStart = CreateEvent(
		NULL,                       // использовать настройки безопасности по умолчанию
		TRUE,                       // событие с ручным сбросом состояния
		FALSE,                      // начальное состояние события
		NULL                        // создать объект без имени
	);

	if (hEventStart == NULL)
	{
		printf_s("CreateEvent failed (%d)\n", GetLastError());
		ExitProcess(1);
	}

	// Создаем событие. Основной поток просигнализирует о завершении работы всем потокам
	hEventStop = CreateEvent(
		NULL,                       // использовать настройки безопасности по умолчанию
		TRUE,                       // событие с ручным сбросом состояния
		FALSE,                      // начальное состояние события
		NULL                        // создать объект без имени
	);

	if (hEventStop == NULL)
	{
		printf_s("CreateEvent failed (%d)\n", GetLastError());
		ExitProcess(1);
	}

	ForksInfo   data[5];                // данные для работы потоков
	HANDLE      hForkMutexesArray[5];   // дескрипторы мьютексов-вилок
	HANDLE      hPhilosophersArray[5];  // дескрипторы потоков-философов

	// создаем мьютексы-вилки
	for (int i = 0; i < 5; ++i)
	{
		// Создаем мьютексы
		hForkMutexesArray[i] = CreateMutex(
			NULL,                   // использовать настройки безопасности по умолчанию
			FALSE,                  // создать мьютекс без захвата текущим потоком
			NULL                    // создать мьютекс без имени
		);

		// Создать мьютекс не удалось - выводим сообщение об ошибке и завершаем программу
		if (hForkMutexesArray[i] == NULL)
		{
			printf_s("CreateMutex failed (%d)\n", GetLastError());
			ExitProcess(1);
		}
	}

	// создаем потоки-философы
	DWORD threadId;
	for (int i = 0; i < 5; ++i)
	{
		// привязка вилок к потокам
		data[i].index = i;
		data[i].left = hForkMutexesArray[i];
		data[i].right = hForkMutexesArray[(i+1) % 5];

		// Создание нового потока
		hPhilosophersArray[i] = CreateThread(
			NULL,                   // использовать настройки безопасности по умолчанию
			0,                      // использовать ращмер стека по умолчанию
			PhilosopherThread,      // имя функции потока
			&data[i],               // аргумент для потока
			0,                      // создать поток с флагами по умолчанию
			&threadId);             // идентификатор потока

		// Если поток не был создан - завершаем работу с ошибкой
		if (hPhilosophersArray[i] == NULL)
		{
			printf_s("CreateThread failed (%d)\n", GetLastError());
			ExitProcess(1);
		}
	}

	Sleep(100);

	// Сигнализируем об окончании генерации исходных данных
	if (!SetEvent(hEventStart))
	{
		printf_s("SetEvent failed (%d)\n", GetLastError());
		ExitProcess(1);
	}

	Sleep(TOTAL_TIME_MS);

	// Сигнализируем об окончании эксперимента
	if (!SetEvent(hEventStop))
	{
		printf_s("SetEvent failed (%d)\n", GetLastError());
		ExitProcess(1);
	}

	// Ожидаем завершения всех потоков
	WaitForMultipleObjects(
		5,                          // количество потоков
		hPhilosophersArray,         // дескрипторы потоков
		TRUE,                       // ждем завершения всех потоков
		INFINITE);                  // ждем в течении бесконечного времени

	// Освобождаем дескриптор события
	CloseHandle(hEventStart);
	CloseHandle(hEventStop);

	// Освобождаем дескрипторы потоков
	for (int i = 0; i < 5; ++i)
		CloseHandle(hPhilosophersArray[i]);

	// Освобождаем дескрипторы мьютексов
	for (int i = 0; i < 5; ++i)
		CloseHandle(hForkMutexesArray[i]);

	return 0;
}

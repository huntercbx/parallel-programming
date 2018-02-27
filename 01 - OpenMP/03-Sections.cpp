////////////////////////////////////////////////////////////////////////////////
/// Программа демонстрирует параллельное выполнение секций.
///
/// При изучении программы следует обратить внимание на следующие моменты:
/// 1. Директивы компилятора (#pragma omp) не требуют подключения заголовочного
/// файла (#include <omp.h>).
///
/// С программой рекомендуется произвести следующие эксперименты:
/// 1. Убрать 'nowait' и сравнить результаты работы
////////////////////////////////////////////////////////////////////////////////

#include <cstdio>

////////////////////////////////////////////////////////////////////////////////
// функция эмулирования вычислительного процесса
////////////////////////////////////////////////////////////////////////////////
void process_data(unsigned int n)
{
	for (unsigned int i=0; i < n; ++i)
	{
		long test = 1;
		for (long l = 1; l < 1000000; ++l)
			test = test % l;
	}
}

int main(int argc, char *argv[])
{

	#pragma omp parallel
	{
		// блок параллельных секций
		#pragma omp sections nowait
		{
			// Первая паралльная секция
			#pragma omp section
			{
				printf("Start of first section\n");
				process_data(500);
				printf("End of first section\n");
			}

			// Вторая паралльная секция
			#pragma omp section
			{
				printf("Start of second section\n");
				process_data(1000);
				printf("End of second section\n");
			}
		}  // конец блока параллельных секций

		#pragma omp single
		printf("End of sections block\n");
	}
	printf("End of parallel block\n");

  return 0;
}

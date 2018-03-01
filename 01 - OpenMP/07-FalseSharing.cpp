////////////////////////////////////////////////////////////////////////////////
// Данный пример иллюстрирует использование параллельных секций OpenMP и
// "ложное" разделение памяти.
//
// Для иллюстрации "ложного" разделения памяти используется массив байт, оба
// потока оперерируют с независимыми элементами массива.
// При этом скорость работы программы зависит от того, насколько "близко"
// расположены эти элементы. При увеличении расстояния между элементами более
// чем на размер строки кэша (32/64 байта), скорость работы программы резко
// возрастет.
//
// Рекомендуется запускать программу в пакетном режиме, например:
// @FOR /L %A IN (-100,1,100) DO @07-FalseSharing.exe %A
////////////////////////////////////////////////////////////////////////////////

#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

std::vector<unsigned char> v(1024);

size_t N = 50000000;

int main(int argc, char *argv[])
{
	// определение смещения
	if (argc != 2)
	{
		printf("Invalid or missing command line arguments\n");
		printf("Usage: %s <offset>\n", argv[0]);
		return -1;
	}

	const int half_len = static_cast<int>(v.size()/2);
	const int offset = half_len + atoi(argv[1]);
	if (offset < 0 || offset >= v.size())
	{
		printf("Offset must be an integer number in [%d, %d]\n", -half_len, half_len - 1);
		return -1;
	}

	// пропускаем нулевое смещение
	if (offset == half_len)
		return 0;

	// время начала вычислений
	double t1 = omp_get_wtime();

	#pragma omp parallel
	{
		#pragma omp sections nowait
		{
			// первая параллельная секция
			#pragma omp section
			{
				for (size_t i=0; i < N; i++)
					++v[half_len];
			}

			// вторая параллельная секция
			#pragma omp section
			{
				for (size_t i=0; i < N; i++)
					++v[offset];
			}
		} // конец блока секций
	} // конец параллельной секции

	// время окончания вычислений
	double t2 = omp_get_wtime();

	// вывод затраченного времени
	printf("%4d, %.5f\n", offset - half_len, t2 - t1);

	return 0;
}

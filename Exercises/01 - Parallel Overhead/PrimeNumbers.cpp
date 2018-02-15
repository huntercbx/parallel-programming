////////////////////////////////////////////////////////////////////////////////
// Данный пример занимается поиском простых чисел в заданном диапазоне.
// Результом работы являются количество найденных чисел в указанном диапазоне.
//
// Программа принимает на вход один аргумент:
//      - правую границу диапозона для поиска простых чисел
////////////////////////////////////////////////////////////////////////////////
#include <cstdio>
#include <algorithm>
#include <omp.h>

////////////////////////////////////////////////////////////////////////////////
// Функция для проверки, является ли заданное число простым
////////////////////////////////////////////////////////////////////////////////
bool IsPrime(unsigned long x)
{
	unsigned long last = (long)floor(sqrt(x));
	for (unsigned long i = 2; i <= last; ++i)
		if ((x % i) == 0) return false;
	return true;
}

////////////////////////////////////////////////////////////////////////////////
// Основная программа
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char *argv[])
{
	// время запуска программы
	double t0 = omp_get_wtime();

	// проверка параметров командной строки
	if (argc != 2)
	{
		printf("Usage: %s max_number\n", argv[0]);
		return 0;
	}

	const unsigned long max_number      = atol(argv[1]);    // максимальное число
	unsigned long       n_prime_numbers = 0;                // количество найденных простых чисел

	// запускаем расчеты
	printf("Searching prime numbers in interval [%lu, %lu]\n", 2, max_number);

	// время начала расчетов (параллельной секции)
	double t1 = omp_get_wtime();

	for (unsigned long i = 2; i < max_number; ++i)
	{
		if (IsPrime(i))
		{
			#pragma omp atomic
			++n_prime_numbers;
		}
	}

	// время окончания расчетов (параллельной секции)
	double t2 = omp_get_wtime();

	// вывод количества найденых простых чисел в диапазоне
	printf("Found %lu prime numbers\n", n_prime_numbers);

	// время окончания работы программы
	double t3 = omp_get_wtime();

	// вывод затраченного времени
	printf("Execution time (program)   : %.5f s\n", t3-t0);	
	printf("Execution time (section)   : %.5f s\n", t2-t1);
	printf("Timer resolution           : %.5f s\n", omp_get_wtick());

	return 0;
}

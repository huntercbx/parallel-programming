////////////////////////////////////////////////////////////////////////////////
// Данная программа находит число Фиббоначи при помощи рекурсивной функции
////////////////////////////////////////////////////////////////////////////////

#include <omp.h>
#include <cstdio>
#include <algorithm>

////////////////////////////////////////////////////////////////////////////////
// Последовательная функция вычисления числа Фиббоначи
////////////////////////////////////////////////////////////////////////////////
int fib_1(int n)
{
	if (n < 2) return n;
	int x, y;
	x = fib_1(n-1);
	y = fib_1(n-2);
	return x + y;
}

////////////////////////////////////////////////////////////////////////////////
// Параллельная функция вычисления числа Фиббоначи
////////////////////////////////////////////////////////////////////////////////
int fib_2(int n)
{
	if (n < 2) return n;
	int x, y;
	#pragma omp task shared(x)
	x = n < 30 ? fib_1(n-1) : fib_2(n-1);
	#pragma omp task shared(y)
	y = n < 30 ? fib_1(n-2) : fib_2(n-2);
	#pragma omp taskwait
	return x + y;
}

int main(int argc, char* argv[])
{
	// определение искомого числа
	const size_t N = (argc == 2) ? atoi(argv[1]) : 40;

	int res_1, res_2;

	double t1 = omp_get_wtime();
	res_1 = fib_1(N);
	double t2 = omp_get_wtime();
	
	double t3 = omp_get_wtime();
	#pragma omp parallel
	{	
		#pragma omp single
		res_2 = fib_2(N);
	}
	double t4 = omp_get_wtime();

	printf("F(%d) = %d = %d\n", N, res_1, res_2);
	
	// вывод затраченного времени
	printf("Execution time (sequential) : %.5f s\n", t2 - t1);
	printf("Execution time (parallel)   : %.5f s\n", t4 - t3);
	printf("Timer resolution            : %.5f s\n", omp_get_wtick());
	if ((t4 - t3) > 0)
		printf("Speedup                     : %.5f s\n", (t2 - t1) / (t4 - t3));	

	return 0;
}

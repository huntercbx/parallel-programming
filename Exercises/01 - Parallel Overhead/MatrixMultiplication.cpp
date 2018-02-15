////////////////////////////////////////////////////////////////////////////////
// Данная программа генерирует две случайные матрицы квадратные матрицы и
// перемножает их.
//
// Программа принимает на вход один необязательный аргумент:
//      - размер квадратных матриц N (значение по умолчанию = 50)
//
// Задача перемножения квадратных матриц размерностью N*N имеет сложность O(N^3)
////////////////////////////////////////////////////////////////////////////////
#include <cstdio>
#include <algorithm>
#include <omp.h>

////////////////////////////////////////////////////////////////////////////////
// Выделение памяти под матрицу размером rows*cols
////////////////////////////////////////////////////////////////////////////////
double** CreateMatrix(size_t rows, size_t cols)
{
	double **M = new double*[rows];
	for (size_t i = 0; i < rows; ++i)
		M[i] = new double[cols];

	return M;
}

////////////////////////////////////////////////////////////////////////////////
// Освобождение памяти из под матрицы M размером rows*cols
////////////////////////////////////////////////////////////////////////////////
void DeleteMatrix(double** M, size_t rows, size_t cols)
{
	for (size_t i = 0; i < rows; ++i)
		delete[] M[i];
	delete[] M;
}

////////////////////////////////////////////////////////////////////////////////
// Заполнение матрицы M размером rows*cols единичными значениями
////////////////////////////////////////////////////////////////////////////////
void FillMatrix(double** M, size_t rows, size_t cols)
{
	for (size_t i = 0; i < rows; ++i)
		for (size_t j = 0; j < cols; ++j)
			M[i][j] = 1.0;
}

////////////////////////////////////////////////////////////////////////////////
// Проверка результатов перемножения единичных матриц
////////////////////////////////////////////////////////////////////////////////
void TestResult(double** M, size_t rows, size_t cols, double result)
{
	for (size_t i = 0; i < rows; ++i)
		for (size_t j = 0; j < cols; ++j)
			if (M[i][j] != result)
				printf("M[%d][%d] = %6.1f\n", i, j, M[i][j]);
}

int main(int argc, char* argv[])
{
	// время запуска программы
	double t0 = omp_get_wtime();

	// определение размера решаемой задачи из параметров командной строки
	const size_t N = (argc == 2) ? atoi(argv[1]) : 50;

	// выделение памяти под матрицы
	double **A = CreateMatrix(N, N);
	double **B = CreateMatrix(N, N);
	double **C = CreateMatrix(N, N);

	// заполнение матриц случайными значениями
	FillMatrix(A, N, N);
	FillMatrix(B, N, N);

	// время начала расчетов (параллельной секции)
	double t1 = omp_get_wtime();

	// умножение матриц
	for (size_t i = 0; i < N; ++i)
	{
		for (size_t j = 0; j < N; ++j)
		{
			double sum = 0;
			for (size_t k = 0; k < N; ++k)
				sum += A[i][k] * B[k][j];
			C[i][j] = sum;
		}
	}

	// время окончания расчетов (параллельной секции)
	double t2 = omp_get_wtime();

	TestResult(C, N, N, N);

	// освобожение памяти
	DeleteMatrix(A, N, N);
	DeleteMatrix(B, N, N);
	DeleteMatrix(C, N, N);

	// время окончания работы программы
	double t3 = omp_get_wtime();

	// вывод затраченного времени
	printf("Execution time (program)   : %.5f s\n", t3-t0);	
	printf("Execution time (section)   : %.5f s\n", t2-t1);
	printf("Timer resolution           : %.5f s\n", omp_get_wtick());

	return 0;
}

#include <mpi.h>
#include <stdio.h>

const size_t GRID_DIM = 3;
const int DO_NOT_REORDER = 0;

void FillMatrix(double *m, size_t rows, size_t cols, double start, double step)
{
	for (size_t i = 0; i < rows; ++i)
	{
		for (size_t j = 0; j < cols; ++j)
		{
			m[i*cols+j] = start;
			start += step;
		}
	}
}

void PrintMatrix(double *m, size_t rows, size_t cols)
{
	for (size_t i = 0; i < rows; ++i)
	{
		for (size_t j = 0; j < cols; ++j)
			printf("%8.2f", m[i*cols+j]);
		printf("\n");
	}
}

void CreateSubmatrixType(MPI_Datatype *type,
	size_t n_rows, size_t n_cols,
	size_t block_rows, size_t block_cols)
{
	// объем памяти занимаемый типом данных MPI_DOUBLE
	int double_lb, double_extent;
	MPI_Type_get_extent(MPI_DOUBLE, &double_lb, &double_extent);

	MPI_Datatype types[2];
	MPI_Type_vector(block_rows, block_cols, n_cols, MPI_DOUBLE, &types[0]);
	types[1] = MPI_UB;

	int blocklengths[2] = {1, 1};
	int displacements[2] = {0, double_extent*block_cols};
	MPI_Type_create_struct(2, blocklengths, displacements, types, type);
	MPI_Type_commit(type);
}

int main( int argc, char *argv[])
{
	int myrank, nprocs;

	// инициализация MPI (создание процессов)
	MPI_Init(&argc, &argv);

	// идентификация текущего процесса
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

	// создаем куб из имеющихся узлов
	int dims[GRID_DIM] = {0, 0, 0};
	int periods[GRID_DIM] = {0, 0, 0};
	MPI_Comm comm3D;
	MPI_Dims_create(nprocs, GRID_DIM, dims);
	MPI_Cart_create(MPI_COMM_WORLD, GRID_DIM, dims, periods, DO_NOT_REORDER, &comm3D);

	// определяем входные параметры задачи
	double *A, *B, *C;
	int matrix_dim[3] = {10, 10, 10};
	bool invalid_parameters = false;
	if (myrank == 0)
	{
		fprintf(stdout, "grid %dx%dx%d created\n", dims[0], dims[1], dims[2]);

		fprintf(stdout, "Input number of rows in matrix A   : ");
		fflush(stdout);
		scanf_s("%d", &matrix_dim[0]);
		fprintf(stdout, "Input number of columns in matrix A: ");
		fflush(stdout);
		scanf_s("%d", &matrix_dim[1]);
		fprintf(stdout, "Input number of columns in matrix B: ");
		fflush(stdout);
		scanf_s("%d", &matrix_dim[2]);
	}

	MPI_Bcast(matrix_dim, 3, MPI_INT, 0, MPI_COMM_WORLD);

	if ((matrix_dim[0] % dims[0]) != 0
		|| (matrix_dim[1] % dims[1]) != 0
		|| (matrix_dim[2] % dims[2]) != 0)
	{
		if (myrank == 0)
			printf("one (or more) matrix dimension is not suitable grid dimension(s)\n");

		// завершение работы
		MPI_Comm_free(&comm3D);
		MPI_Finalize();
		return -1;
	}

	size_t A_rows = matrix_dim[0], AA_rows = matrix_dim[0]/dims[0],
	       A_cols = matrix_dim[1], AA_cols = matrix_dim[1]/dims[1],
	       B_rows = matrix_dim[1], BB_rows = matrix_dim[1]/dims[1],
	       B_cols = matrix_dim[2], BB_cols = matrix_dim[2]/dims[2],
	       C_rows = matrix_dim[0], CC_rows = matrix_dim[0]/dims[0],
	       C_cols = matrix_dim[2], CC_cols = matrix_dim[2]/dims[2];

	// генерируем исходные матрицы в нулевом процессе
	if (myrank == 0)
	{
		A = new double[A_rows*A_cols];
		B = new double[B_rows*B_cols];
		C = new double[C_rows*C_cols];
		FillMatrix(A, A_rows, A_cols, 1, 0.01);
		FillMatrix(B, B_rows, B_cols, 1, 0.02);
	}

	// создаем коммуникаторы для плоскостей XY, XZ, YZ
	MPI_Comm commXY, commXZ, commYZ;
	int XY_dimensions[GRID_DIM] = {1, 1, 0};
	int XZ_dimensions[GRID_DIM] = {1, 0, 1};
	int YZ_dimensions[GRID_DIM] = {0, 1, 1};
	MPI_Cart_sub(comm3D, XY_dimensions, &commXY);
	MPI_Cart_sub(comm3D, XZ_dimensions, &commXZ);
	MPI_Cart_sub(comm3D, YZ_dimensions, &commYZ);

	// создаем коммуникаторы для осей X, Y, Z
	MPI_Comm commX, commY, commZ;
	int X_dimension[GRID_DIM] = {1, 0, 0};
	int Y_dimension[GRID_DIM] = {0, 1, 0};
	int Z_dimension[GRID_DIM] = {0, 0, 1};
	MPI_Cart_sub(comm3D, X_dimension, &commX);
	MPI_Cart_sub(comm3D, Y_dimension, &commY);
	MPI_Cart_sub(comm3D, Z_dimension, &commZ);

	// получаем координаты процесса
	int coords[GRID_DIM];
	int myrank3D;
	MPI_Comm_rank(comm3D, &myrank3D);
	MPI_Cart_coords(comm3D, myrank3D, GRID_DIM, coords);
	printf("Global rank: %2d, 3D rank : %2d, coords = (%d, %d, %d)\n", myrank, myrank3D, coords[0], coords[1], coords[2]);

	// Выделяем памаять в каждом узле
	double *AA, *BB, *CC, *CC_R;
	AA = new double[AA_rows*AA_cols];
	BB = new double[BB_rows*BB_cols];
	CC = new double[CC_rows*CC_cols];
	CC_R = new double[CC_rows*CC_cols];

	MPI_Datatype typeA, typeB, typeC;
	CreateSubmatrixType(&typeA, A_rows, A_cols, AA_rows, AA_cols);
	CreateSubmatrixType(&typeB, B_rows, B_cols, BB_rows, BB_cols);
	CreateSubmatrixType(&typeC, C_rows, C_cols, CC_rows, CC_cols);

	// данные для рассылки матрицы A
	int *sendcountsA = new int[dims[0]*dims[1]];
	int *displsA = new int[dims[0]*dims[1]];
	for (int i = 0; i < dims[0]; ++i)
	{
		for (int j = 0; j < dims[1]; ++j)
		{
			sendcountsA[i*dims[1] + j] = 1;
			displsA[i*dims[1] + j] = i*dims[1]*AA_rows + j;
		}
	}

	// данные для рассылки матрицы B
	int *sendcountsB = new int[dims[1]*dims[2]];
	int *displsB = new int[dims[1]*dims[2]];
	for (int i = 0; i < dims[1]; ++i)
	{
		for (int j = 0; j < dims[2]; ++j)
		{
			sendcountsB[i*dims[2] + j] = 1;
			displsB[i*dims[2] + j] = i*dims[2]*BB_rows + j;
		}
	}

	// данные для рассылки матрицы C
	int *sendcountsC = new int[dims[0]*dims[2]];
	int *displsC = new int[dims[0]*dims[2]];
	for (int i = 0; i < dims[0]; ++i)
	{
		for (int j = 0; j < dims[2]; ++j)
		{
			sendcountsC[i*dims[2] + j] = 1;
			displsC[i*dims[2] + j] = i*dims[2]*CC_rows + j;
		}
	}

	if (coords[2] == 0)
		MPI_Scatterv(A, sendcountsA, displsA, typeA, AA, AA_rows*AA_cols, MPI_DOUBLE, 0, commXY);

	if (coords[0] == 0)
		MPI_Scatterv(B, sendcountsB, displsB, typeB, BB, BB_rows*BB_cols, MPI_DOUBLE, 0, commYZ);

	MPI_Bcast(AA, AA_rows*AA_cols, MPI_DOUBLE, 0, commZ);

	MPI_Bcast(BB, BB_rows*BB_cols, MPI_DOUBLE, 0, commX);

	for (size_t i = 0; i < CC_rows; ++i)
	{
		for (size_t j = 0; j < CC_cols; ++j)
		{
			CC[i*CC_cols+j] = 0.0;
			for (size_t k = 0; k < AA_cols; ++k)
				CC[i*CC_cols+j] += AA[i*AA_rows+k]*BB[k*BB_rows + j];
		}
	}

	MPI_Reduce(CC, CC_R, CC_rows*CC_cols, MPI_DOUBLE, MPI_SUM, 0, commY);

	if (coords[1] == 0)
		MPI_Gatherv(CC_R, CC_rows*CC_cols, MPI_DOUBLE, C, sendcountsC, displsC, typeC, 0, commXZ);

	if (myrank == 0)
		PrintMatrix(C, C_rows, C_cols);

	delete [] AA;
	delete [] BB;
	delete [] CC;
	delete [] CC_R;

	if (myrank == 0)
	{
		delete [] A;
		delete [] B;
		delete [] C;
	}

	// освобождение ресурсов
	MPI_Comm_free(&comm3D);
	MPI_Comm_free(&commXY);
	MPI_Comm_free(&commXZ);
	MPI_Comm_free(&commYZ);
	MPI_Comm_free(&commX);
	MPI_Comm_free(&commY);
	MPI_Comm_free(&commZ);

	// завершение работы
	MPI_Finalize();
	return 0;
}

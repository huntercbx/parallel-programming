////////////////////////////////////////////////////////////////////////////////
// Данная программа решает задачу коммивояжера методом полного перебора 
// (методом "грубой силы", англ. brute force)
////////////////////////////////////////////////////////////////////////////////
#include <cstdio>
#include <vector>
#include <algorithm>
#include <ctime>

////////////////////////////////////////////////////////////////////////////////
// матрица расстояний между городами
////////////////////////////////////////////////////////////////////////////////
const int distance_map[15][15] = 
{{ 0, 29, 82, 46, 68, 52, 72, 42, 51, 55, 29, 74, 23, 72, 46},
 {29,  0, 55, 46, 42, 43, 43, 23, 23, 31, 41, 51, 11, 52, 21},
 {82, 55,  0, 68, 46, 55, 23, 43, 41, 29, 79, 21, 64, 31, 51},
 {46, 46, 68,  0, 82, 15, 72, 31, 62, 42, 21, 51, 51, 43, 64},
 {68, 42, 46, 82,  0, 74, 23, 52, 21, 46, 82, 58, 46, 65, 23},
 {52, 43, 55, 15, 74,  0, 61, 23, 55, 31, 33, 37, 51, 29, 59},
 {72, 43, 23, 72, 23, 61,  0, 42, 23, 31, 77, 37, 51, 46, 33},
 {42, 23, 43, 31, 52, 23, 42,  0, 33, 15, 37, 33, 33, 31, 37},
 {51, 23, 41, 62, 21, 55, 23, 33,  0, 29, 62, 46, 29, 51, 11},
 {55, 31, 29, 42, 46, 31, 31, 15, 29,  0, 51, 21, 41, 23, 37},
 {29, 41, 79, 21, 82, 33, 77, 37, 62, 51,  0, 65, 42, 59, 61},
 {74, 51, 21, 51, 58, 37, 37, 33, 46, 21, 65,  0, 61, 11, 55},
 {23, 11, 64, 51, 46, 51, 51, 33, 29, 41, 42, 61,  0, 62, 23},
 {72, 52, 31, 43, 65, 29, 46, 31, 51, 23, 59, 11, 62,  0, 59},
 {46, 21, 51, 64, 23, 59, 33, 37, 11, 37, 61, 55, 23, 59,  0}};

////////////////////////////////////////////////////////////////////////////////
// функция для вычисления оценки найденого маршрута (целиком или его части)
// solution - маршрут
// len      - количество оцениваемых элементов
////////////////////////////////////////////////////////////////////////////////
int estimate(const std::vector<int>& solution, size_t len)
{
	if (solution.size() < len)
		return -1;

	int sum = 0;
	for (size_t i = 1; i < len; ++i)
		sum += distance_map[solution[i-1]][solution[i]];
	if (solution.size() == len)
		sum += distance_map[solution[len-1]][solution[0]];
	return sum;
}

int main(int argc, char* argv[])
{
	// определение размера решаемой задачи из параметров командной строки
	const size_t N = std::min(15, std::max(5, (argc == 2) ? atoi(argv[1]) : 10));

	// время начала рассчетов
	clock_t t1 = clock();

	// пока что лучшим решением является первое найденное
	std::vector<int> best_solution_found(N);
	for (size_t i = 0; i < N; ++i)
		best_solution_found[i] = i;

	// оценка лучшего маршрута (граница отсечения)
	int best_estimation = estimate(best_solution_found, best_solution_found.size());

	for (size_t i = 1; i < N; ++i)
	{
		// порождаем маршрут, при котором вторым будет посещен города i
		std::vector<int> current_solution(N);
		current_solution[0] = 0;
		current_solution[1] = i;
		for (size_t j = 2; j < N; ++j)
			current_solution[j] = (j <= i) ? j-1 : j;

		bool exit = false;
		while (!exit)
		{
			// производим оценку текущего маршрута
			size_t len = current_solution.size();
			int estimation = estimate(current_solution, len);

			// если текущая оценка лучше, то найдено более оптимальное решение
			if (estimation < best_estimation)
			{
				best_estimation = estimation;
				best_solution_found = current_solution;
			}

			// порождаем следующий (в лексикографическом порядке) маршрут
			exit = !std::next_permutation(current_solution.begin(), current_solution.end());
			exit |= current_solution[0] != 0;
			exit |= current_solution[1] != i;
		}
	}

	// время окончания рассчетов
	clock_t t2 = clock();

	// вывод результата
	printf("Best solution found (%d) : ", best_estimation);
	for (size_t i = 0; i < best_solution_found.size(); ++i)
		printf(i > 0 ? " -> %d": "%d", best_solution_found[i]);
	printf(" -> %d\n", best_solution_found[0]);

	// вывод затраченного времени
	printf("Execution time : %.5f s\n", (float)(t2-t1)/CLOCKS_PER_SEC);

	return 0;
}


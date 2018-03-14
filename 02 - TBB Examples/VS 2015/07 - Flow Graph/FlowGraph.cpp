////////////////////////////////////////////////////////////////////////////////
// Данная программа иллюстрирует построение графа вычислений с использованием
//    библиотеки TBB для вычисления стандартного отклонения выборки данных.
//
// Программа принимает на вход один аргумент:
//      - имя файла с выборкой (каждое значение на отдельной строке)
////////////////////////////////////////////////////////////////////////////////
#include <iostream>
#include <fstream>

#include <tbb/task_scheduler_init.h>
#include <tbb/tick_count.h>
#include <tbb/compat/thread>
#include <tbb/flow_graph.h>

using namespace std;
using namespace tbb;

////////////////////////////////////////////////////////////////////////////////
// Узел источник - чтение данных из файла по одному значению за раз
////////////////////////////////////////////////////////////////////////////////
struct input
{
	input(ifstream& f) : file(f)
	{
	}

	bool operator()(double& x)
	{
		std::string line;
		if (!getline(file, line) || line.empty())
			return false;

		x = atof(line.c_str());
		return true;
	}

private:
	ifstream& file;
};

////////////////////////////////////////////////////////////////////////////////
// Функциональный узел - подсчет количества значений
////////////////////////////////////////////////////////////////////////////////
struct count_rec
{
	size_t operator()(double x)
	{
		++n;
		return n;
	}

private:
	size_t n = 0;
};

////////////////////////////////////////////////////////////////////////////////
// Функциональный узел - подсчет суммы значений
////////////////////////////////////////////////////////////////////////////////
struct sum
{
	double operator()(double x)
	{
		s += x;
		return s;
	}

private:
	double s = 0;
};

////////////////////////////////////////////////////////////////////////////////
// Функциональный узел - вычисление стандартного отклонения
////////////////////////////////////////////////////////////////////////////////
struct std_dev
{
	std_dev(double& res) : result(res)
	{
	}

	double operator() (tuple<double, double, size_t> v)
	{
		auto x_sum = get<0>(v);
		auto x2_sum = get<1>(v);
		auto N = get<2>(v);

		result = sqrt(x2_sum / N - (x_sum / N)*(x_sum / N));
		return result;
	}

private:
	double& result;
};

////////////////////////////////////////////////////////////////////////////////
// Основная программа
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
	// определение размера решаемой задачи из параметров командной строки
	if (argc != 2)
	{
		cout << "Invalid or missing command line arguments" << endl
			<< "Usage: " << argv[0] << " <file_name>";
		return -1;
	}

	ifstream f1;
	f1.open(argv[1], ios_base::in);
	if (!f1.is_open())
	{
		cout << "Can not open file:" << argv[1] << endl;
		return -1;
	}

	// инициализация планировщика задач TBB
	task_scheduler_init init;

	double result;

	// создание узлов графа
	// для функциональных узлов можно использовать как функциональные объекты, так и лямбда-функции 
	flow::graph g;
	flow::source_node<double> source(g, input(f1), false);
	flow::broadcast_node<double> broadcast(g);
	flow::function_node<double, double> squarer(g, flow::unlimited, [](double x) { return x*x; });
	flow::function_node<double, size_t> counter(g, flow::unlimited, count_rec());
	flow::function_node<double, double> sumer_1(g, flow::unlimited, sum());
	flow::function_node<double, double> sumer_2(g, flow::unlimited, sum());
	flow::join_node<tuple<double, double, size_t>, flow::queueing> join(g);
	flow::function_node<tuple<double, double, size_t>, double> std_dev_calc(g, flow::serial, std_dev(result));

	// создание ребер графа
	make_edge(source, broadcast);
	make_edge(broadcast, sumer_1);
	make_edge(broadcast, squarer);
	make_edge(broadcast, counter);
	make_edge(squarer, sumer_2);
	make_edge(sumer_1, get<0>(join.input_ports()));
	make_edge(sumer_2, get<1>(join.input_ports()));
	make_edge(counter, get<2>(join.input_ports()));
	make_edge(join, std_dev_calc);

	// активиация источника и ожидание окончания обработки
	source.activate();
	g.wait_for_all();

	// вывод результатов
	cout << "Standart deviation = " << result << endl;

	f1.close();
	return 0;
}

////////////////////////////////////////////////////////////////////////////////
// Демонстрация использования параллельного алгоритма parallel_do для подсчета
// ссылок на HTML страницы в локальной директории
//
// Программа принимает на вход имя исходного HTML файла, с которого начинается
// поиск, в результате будут обработаны все файлы, которые доступны со стартовой
// страницы
//
// В качестве тестовых исходных данных можно воспользоваться документацей на TBB,
// указав в качестве исходного файл <TBB_Root>\doc\html\index.html
//
// В отладочной конфигурации можно просмотреть каким потоком были обработаны
// конкретные файлы.
////////////////////////////////////////////////////////////////////////////////
#include <iostream>
#include <fstream>
#include <sstream>
#include <list>

#include <tbb/task_scheduler_init.h>
#include <tbb/parallel_do.h>
#include <tbb/concurrent_hash_map.h>
#include <tbb/tick_count.h>
#include <tbb/compat/thread>

using namespace std;
using namespace tbb;

typedef concurrent_hash_map<string, unsigned int> PageReferenceMap;
PageReferenceMap references;

class Crawler {
public:

	Crawler(const std::string& path) : base_path(path)
	{}

	// Данное определение типа необходимо для parallel_do
	typedef string argument_type;

	void operator()(string file_name, tbb::parallel_do_feeder<string>& feeder) const
	{
		#ifdef _DEBUG
			// вывод идентификатора потока и диапазона
			stringstream ss;
			ss << "Thread " << this_thread::get_id() << " - " << file_name << endl;
			cout << ss.str();
		#endif

		string content;
		read_page(file_name, content);
		process_page(content, feeder);
	}

private:
	////////////////////////////////////////////////////////////////////////////////
	// Чтение файла в строку целиком
	////////////////////////////////////////////////////////////////////////////////
	void read_page(const std::string& file_name, std::string& content) const
	{
		ifstream file;
		file.open(base_path + file_name, ios_base::in);
		if (file.is_open())
		{
			file.seekg(0, ios::end);
			content.reserve(static_cast<unsigned int>(file.tellg()));
			file.seekg(0, ios::beg);

			content.assign(
				std::istreambuf_iterator<char>(file),
				std::istreambuf_iterator<char>());
		}
		else
			content.clear();
		file.close();
	}

	////////////////////////////////////////////////////////////////////////////////
	//  Поиск ссылок на странице и добавление их в обработку
	////////////////////////////////////////////////////////////////////////////////
	void process_page(const std::string& content, tbb::parallel_do_feeder<string>& feeder) const
	{
		size_t from = 0, to, pos;
		while (true)
		{
			from = content.find("<a href=\"", from);
			if (from == string::npos) break;
			to = content.find("\"", from + 9);
			if (to == string::npos) break;
			string link = content.substr(from + 9, to-from - 9);

			if (link[0] == '.' && link[1] == '/')
				link = link.substr(2, link.size() - 2);

			pos = link.find("#");
			if (pos != string::npos)
				link = link.substr(0, pos);

			from = to;

			if (link.empty()) continue;

			{
				PageReferenceMap::accessor a;
				if (references.insert(a, link))
				{
					a->second = 1;
					feeder.add(link);
				}
				else
					a->second++;
			}
		}
	}

private:
	std::string base_path;
};

////////////////////////////////////////////////////////////////////////////////
// Основная программа
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
	if (argc != 2)
	{
		cout << "Invalid or missing command line arguments" << endl
		     << "Usage: " << argv[0] << " <file_name>";
		return -1;
	}

	// инициализация планировщика задач TBB
	task_scheduler_init init;

	string start(argv[1]), base_path;
	size_t pos = start.find_last_of('\\');
	if (pos != string::npos)
	{
		base_path = start.substr(0, pos + 1);
		start = start.substr(pos + 1, start.size() - pos - 1);
	}

	list<string> pages;
	pages.push_back(start);

	{
		PageReferenceMap::accessor a;
		references.insert(a, pages.front());
		a->second = 1;
	}

	parallel_do(pages.begin(), pages.end(), Crawler(base_path));

	// вывод результатов работы
	for (auto i = references.begin(), ie = references.end(); i != ie; ++i)
		cout << i->first << " = " << i->second << endl;

	return 0;
}


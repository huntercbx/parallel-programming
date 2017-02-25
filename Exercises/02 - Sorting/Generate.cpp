////////////////////////////////////////////////////////////////////////////////
// Программа генерирует файл со случайными строками для тестирования сортировки.
// Программа принимает на вход два необязательных аргумента:
//      - количество строк в выходном файле (по умолчанию 1 000 000),
//      - зерно генератора случайных чисел (по умолчанию будет сгенирировано)
////////////////////////////////////////////////////////////////////////////////
#include <iostream>
#include <fstream>
#include <string>
#include <ctime>

const size_t PROGRESS_UPDATE = 250000;  // период обновления прогресса
const size_t LENGTH = 20;               // длина генерируемых строк
const char*  FILENAME = "input.txt";    // имя выходного файла

using namespace std;

int main(int argc, char *argv[])
{
	size_t N = 1000000;
	unsigned int seed = static_cast<unsigned int>(time(nullptr));

	if (argc >= 2)
		N = atoi(argv[1]);

	if (argc >= 3)
		seed = atoi(argv[2]);

    cout << "SEED = " << seed << endl;
    srand(seed);

   	ofstream file;
	file.open(FILENAME);
    cout << "Saving \"" << FILENAME << "\" ";

    file << N << endl;
    for (size_t i = 0; i < N; ++i)
    {
   		string str;
        for (size_t j = 0; j < LENGTH; ++j)
			str += 'A' + rand() % 26;
        file << str << endl;

        // обновляем прогресс
        if (i % PROGRESS_UPDATE == 0)
            cout << ".";
    }
    file.close();
}
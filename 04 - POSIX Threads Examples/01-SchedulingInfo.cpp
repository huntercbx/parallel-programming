////////////////////////////////////////////////////////////////////////////////
// Данный пример выводит информацию о поддерживаемых операционной системой
// дисциплинах диспетчеризации.
//
// Для каждой поддерживаемой дисциплины диспетчеризации получается минимальный
// и максимальный приоритеты.
//
// Для основного потока программы выводится информация о дисциплине
// диспетчеризации установленной по умолчанию.
////////////////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <pthread.h>
#include <errno.h>

// вывод информации об указанной политике диспетчеризации
void print_policy_info(int policy)
{
	const char policy_fifo[]	= "SCHED_FIFO ";
	const char policy_rr[]		= "SCHED_RR   ";
	const char policy_other[]	= "SCHED_OTHER";
	const char policy_batch[]	= "SCHED_BATCH";
	const char policy_idle[]	= "SCHED_IDLE ";

	const char *policy_name;
	switch (policy)
	{
	case SCHED_FIFO:
		policy_name = policy_fifo; break;
	case SCHED_RR:
		policy_name = policy_rr; break;
	case SCHED_OTHER:
		policy_name = policy_other; break;
#ifndef PTHREAD_WIN32
	case SCHED_BATCH:
		policy_name = policy_batch; break;
	case SCHED_IDLE:
		policy_name = policy_idle; break;
#endif
	default:
		return;
	}

	errno = 0;
	int min_priority = sched_get_priority_min(policy);
	int max_priority = sched_get_priority_max(policy);
	if (errno == EINVAL)
		printf("policy %s is not supported\n", policy_name);
	else
		printf("policy %s : min priority = %5d, max priority = %5d\n",
			policy_name, min_priority, max_priority);
}

int main(int argc, char *argv[])
{
	// выводим информацию о дисциплинах диспетчеризации
	print_policy_info(SCHED_FIFO);
	print_policy_info(SCHED_RR);
	print_policy_info(SCHED_OTHER);
#ifndef PTHREAD_WIN32
	print_policy_info(SCHED_BATCH);
	print_policy_info(SCHED_IDLE);
#endif

	// получаем идентификатор текущего (главного) потока
	pthread_t main_thread = pthread_self();

	// получаем параметры диспетчеризации для потока
	int				policy;
	sched_param		param;
	int res = pthread_getschedparam(main_thread, &policy, &param);
	if (res == 0)
	{
		printf("\nCurrent thread ");
		switch (policy)
		{
		case SCHED_FIFO:
			printf("policy = SCHED_FIFO, "); break;
		case SCHED_RR:
			printf("policy = SCHED_RR, "); break;
		case SCHED_OTHER:
			printf("policy = SCHED_OTHER, "); break;
		default:
			printf("policy = ???, ", policy); break;
		}
		printf("priority = %d\n", param.sched_priority);
	}
	else
		printf("pthread_getschedparam failed (%d)\n", res);

	return 0;
}
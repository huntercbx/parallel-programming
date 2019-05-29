class MyThread implements Runnable
{
    public Thread	t;
    private boolean suspendFlag;
    private float task_progress;

    MyThread()
    {
        t = new Thread(this);
        t.start();
        suspendFlag = false;
        task_progress = 0;
    }

    public void run()
    {
        System.out.println("Thread " + t.getName() + " started");

        // основной цикл работы потока
        for (int counter = 0; counter < 2000 ; ++counter)
        {
            // проверка на приостановку работы
            synchronized (this)
            {
                try {
                    while (suspendFlag)
                        wait();
                } catch (InterruptedException e) {
                    System.out.println("Thread " + t.getName() + " interrupted, progress=" + task_progress);
                    return;
                }
            }

            // проверка на прерывание работы потока
            if (t.isInterrupted())
            {
                System.out.println("Thread " + t.getName() + " interrupted, progress=" + task_progress);
                return;
            }

            // эмулирование вычислений
            int test = 1;
            for (int j = 1; j < 1000000; ++j)
                test = test % j;

            // обновление прогресса выполнения
            task_progress = (float)counter/2000;
        }

        System.out.println("Thread " + t.getName()  + " endeded");
    }

    public void suspend()
    {
        System.out.println("Thread " + t.getName() + " suspended, progress=" + task_progress);
        suspendFlag = true;
    }

    public synchronized void resume()
    {
        System.out.println("Thread " + t.getName() + " resumed, progress=" + task_progress);
        suspendFlag = false;
        notify();
    }

}

public class SuspendResume
{

    public static void main(String[] args)
    {
        MyThread t1 = new MyThread();
        MyThread t2 = new MyThread();
        MyThread t3 = new MyThread();

        try {
            Thread.sleep(2000);
            t1.suspend();
            t2.suspend();

            Thread.sleep(2000);
            t1.resume();
            t2.resume();
            t3.t.interrupt();

            t1.t.join();
            t2.t.join();
            t3.t.join();

        } catch (InterruptedException e) {}
    }

}

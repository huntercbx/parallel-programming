class MyThread implements Runnable {

    private Thread th;

    MyThread()
    {
        th = new Thread(this);
        th.start();
    }

    public void run()
    {
        System.out.println("Thread started: name = " + th.getName() + ", id = " + th.getId());
        for (int i=0; i < 100; ++i)
        {
            int test = 1;
            for (int j = 1; j < 1000000; ++j)
                test = test % j;
        }
        System.out.println("Thread ended: name = " + th.getName());
    }
}

public class CreateThreads {

    /**
     * @param args
     */
    public static void main(String[] args) {

        System.out.println("Priopity: min = " + Thread.MIN_PRIORITY
                + ", max = " + Thread.MAX_PRIORITY
                + ", norm = " + Thread.NORM_PRIORITY);

        for (int i=0; i < 8; ++i)
            new MyThread();

        System.out.println("There are " + Thread.activeCount() + " active threads in group \""
                + Thread.currentThread().getThreadGroup().getName() + "\"");

        try {
            Thread.sleep(2000);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }
}
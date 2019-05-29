import java.util.concurrent.atomic.AtomicInteger;

class MyThread extends Thread {
    private int start;
    private int end;
    private AtomicInteger n_found;

    MyThread(int s, int e, AtomicInteger n)
    {
        super();
        start = s;
        end = e;
        n_found = n;
    }

    public void run()
    {
        System.out.println("Start search prime numbers in interval ["+ start + "," + end + "]");

        for(int i = start; i != end; ++i)
        {
            boolean isPrime = true;
            for(int j = 2; isPrime && j < i/2; ++j)
                isPrime = (i % j) != 0;

            if (isPrime)
                n_found.incrementAndGet();
        }
    }
}

public class PrimeNumbers {

    /**
     * @param args
     */
    public static void main(String[] args) {
        int N_THREADS = 4;
        int MAX_NUMBER = 1000000;
        AtomicInteger n_prime_numbers = new AtomicInteger(0);

        long t1 = System.currentTimeMillis();

        MyThread treads[] = new MyThread[N_THREADS];

        int N = MAX_NUMBER/N_THREADS;
        for (int i = 0; i < N_THREADS; ++i)
        {
            int st = (i == 0) ? 2 : i*N;
            int end = (i == N_THREADS-1) ? MAX_NUMBER : (st + N - 1);
            treads[i] = new MyThread(st, end, n_prime_numbers);
            treads[i].start();
        }

        try {
            for (int i = 0; i < N_THREADS; ++i)
                treads[i].join();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }

        long t2 = System.currentTimeMillis();

        System.out.println("Found " + n_prime_numbers + " prime numbers in interval [2," + MAX_NUMBER + "]");
        System.out.println("Time elapsed: " + (t2-t1) + " ms");
    }
}

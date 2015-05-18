With all CUDA code besides allocating host memory disabled, running
./classify shuffled_lsa_labelled.txt clocks in at 15.051 s, as measured with the
time command. With all printing disabled and a batch size of 2048, it runs in
16.178 s, so IO and parsing alone take up 15.051 / 16.178 = 93.0% of the
program's execution time. With printing enabled, it runs in 16.466 s, so in that
case 15.051 / 16.466 = 91.4% of the execution time is used by IO and parsing
alone.


For the following, I disabled the kernel at the end of the main loop that runs
on the remaining reviews, but it makes a pretty negligble difference.

With a batch size of 65536, my kernel has an average latency of 2.9791 ms, for a
throughput of 21999 reviews / ms.

With a batch size of 16384, my kernel has an average latency of 1.1642 ms, for a
throughput of 14073 reviews / ms.

With a batch size of 2048, my kernel has an average latency of 599.23 us, for a
throughput of 3417.7 reviews / ms.

With a batch size of 1024, my kernel has an average latency of 587.00 us, for a
throughput of 1744.4 reviews / ms.

With a batch size of 32, my kernel has an average latency of 246.50 us, for a
throughput of 129.82 reviews / ms.

With a batch size of 1, my kernel has an average latency of 61.733 us, for a
throughput of 16.199 reviews / ms.
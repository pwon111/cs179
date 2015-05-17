All benchmarking was done on an otherwise idle GeForce GTX 780 Ti, with the
printing callback disabled in order to maximize concurrency and performance.
This set took me roughly ten hours, a lot of which was just trying different
ways to address the issue of having to do a bunch of atomic operations with
global memory, including an approach which assigned each review in a batch to a
cluster, and then recomputed their average position at the end (which clearly
did not give the desired results).

1. The latency of classifying a single batch is equal to the sum of the
latencies of each part of the process of doing so, i.e. using cudaMemcpyAsync
twice to copy the input data to the GPU and the output cluster assignments back,
as well as the time it takes to run the sloppyClusterKernel itself. According to
nvprof, with k = 50 and batch_size = 2048, the host to device transfers took on
average 69.910 us, while the much smaller device to host transfers took 
2.9090 us, for a combined latency of 72.819 us. The sloppyClusterKernel then
took on average 1.6134 ms = 1613.4 us, for a total latency of
1613.4 + 72.819 = 1686.219 us = 1.6862 ms per batch. This scales linearly with
batch_size, since the number of elements that the two
cudaMemcpyAsync calls each have to copy is O(n), as the input is n * REVIEW_DIM
elements long, and the output is n elements long, while the number of atomic
operations the kernel has to do to global memory is also O(n) as REVIEW_DIM per
review. These values don't include the overhead for the CUDA API calls at
runtime, but these are all on the order of 10-20 us so they have little effect
on the overall latency.

2. The throughput of a program is equal to the lowest throughput of any of its
individual stages, which in this case is clearly that of executing the
sloppyClusterKernel, as it takes significantly more time than the
cudaMemcpyAsync operations. According to nvprof, it takes an average of
1.6134 ms to execute the kernel, which process batch_size = 2048 per kernel for
a pipeline throughput of 2048 / (1.6134 * 10^-3) = 1,269,369 reviews / s.
However, because the program spends so much time doing other things, it still
takes around 15.406 s (timed using the time command) to process all 1,569,265
reviews, for an overall throughput of 1,569,265 / 15.406 = 101861 reviews / s.
These rates are more or less constant in batch_size, since as explained above,
the kernel itself, which is the highest-latency stage of the pipeline, has to do
O(batch_size) atomic operations on global memory (for fixed k) but also clearly
processes O(batch_size) reviews. This becomes less true for smaller batch_sizes,
since there is always some overhead associated with each thread calculating the
distance of its input to each cluster, but for larger batch_sizes I have found
that latency tends to scale linearly with batch_size. This is also generally
true for the program as a whole since the most time-intensive operations are all
O(n) with respect to batch_size, but also allow the program to process
O(batch_size) reviews per batch. This is supported by the fact that the
program's running time does not deviate much from 15 s over a wide range of
batch sizes. The loader.py script, on the other hand, takes several minutes to
process the contents of shuffled_reviews.json, and piping its output into the
cluster program is thus clearly inferior to cat-ing the shuffled_lsa.txt file
in, since the streams would constantly be starved for data to transfer to the
GPU for the kernel to execute.

3. Cluster does (or at least very nearly does) saturate the PCI-E interface
between device and host. Using the bandwidthtest program provided by Nvidia,
I was able to benchmark my pinned transfer bandwidth to 6.041925 GB/s for host
to device, and 6.557944 GB/s for device to host. Multiplying the former by the
amount of time nvprof says is spent in host to device transfers, I find that I
should be able to transfer 6.041925 * 10^9 B/s * 53.927 * 10^-3 s = 325,822,889
bytes during the program's runtime. Since cudaMemcpyAsync is called 766 times,
it copies 767 * batch_size * REVIEW_DIM * sizeof(float) bytes, which in this
case is 314,163,200 bytes, just short of the theoretical limit. However, the
cluster program should be able to transfer
6.557944 * 10^9 B/s * 2.2439 * 10^-3 s = 14,715,371 bytes from the device back
to host based on nvprof output, and it actually does
767 * batch_size * sizeof(int), which is 6,283,264 bytes, less then half of the
theoretical limit. Despite this, it seems as though the kernel is the bottleneck
for essentially any value of k, as after testing it on values from 2 through
200, the memory transfer operations take almost identical amounts of time for
any value of k, with the kernel itself still taking up the vast majority of the
program's execution time.

4. I do not think that this algorithm's performance would be improved if
multiple GPUs were used. The batch-based kernel relies on the fact that the
updates it makes to the cluster location and member review counts are atomic,
and thus threads can see the changes that each other is making. Because of this,
if reviews are processed in a manner such that the changes they cause are not
immediately visible to all other threads, then the cluster center locations
that are each being updated diverge as they become based on different sets of
data. Thus, if two GPUs were used they would have to be able to see the changes
being made on the other device, which would require mmemory to be copied back
and forth constantly, which would greatly increase program latency and decrease
throughput as the kernel would have to be blocked from executing every time.
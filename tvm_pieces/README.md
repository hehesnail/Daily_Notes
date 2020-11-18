# Boring TVM Code

## *2020.11.17*
* use *parallel*, *vectorize* to create schedule which accelerate the execution time of operator.
* Use *split* for splitting the for loop, reorder can change the axis orders, quite like Halide.
* Identify the the bottleneck is memory bandwidth or computation. vector_add and broadcast_add are all memory-bound element-wise calculation.
* A good schedule needs to consider multiple performance-related factors together.

## *2020.11.18*
* for matrix multiplication, *reorder* the sum axis to avoid column based memory access
* choose the right tiling size to improve the cache effcience, *tile* primitive tile blocks, *fuse* can fuse two axes into one to be further parallelized.
* The non-continuous write issue is severer than the non-continuous read, so we can first write the results to a local buffer for each sub-matrix computation, and then write them back to the final matrix C.
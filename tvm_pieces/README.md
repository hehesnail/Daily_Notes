# Boring TVM Code

## *2020.11.17*
* use parallel, vectorize to create schedule which accelerate the execution time of operator.
* Use split for splitting the for loop, reorder can change the axis orders, quite like Halide.
* Identify the the bottleneck is memory bandwidth or computation. vector_add and broadcast_add are all memory-bound element-wise calculation.
* A good schedule needs to consider multiple performance-related factors together.
# Boring TVM Code

## *2020.11.17*
* use parallel, vectorize to create schedule which accelerate the execution time of operator
* use split for splitting the for loop, quite like Halide
* identify the the bottleneck is memory bandwidth or computation
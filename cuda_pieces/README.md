### CUDA C/C++ Basics
* **_\_global\_\_**, runs on the device, called from host code
* **cudaMalloc**, **cudaFree**, **cudaMemcpy** ---> malloc, free, memcpy
* grid -> block(blockIdx) -> threads(threadIdx)
* Why threaad? -> unlike blocks, threads can communicate & synchronize
* Within a block, threads share data via shared memory (fast on-chip memory), **_\_shared\_\_**
* **_\_syncthreads** ---> synchronize all threads within a block, used to prevent RAW / WAR / WAW hazards, all threads must reach the barrier.
* Kernal launches are asynchronous
  * cudaMemcpy -> block cpu until copy is complete
  * cudaMemcpyAsync -> asynchronous, does not block cpu
  * cudaDeviceSynchronize -> block cpu until all preceding cuda calls have completed
* Report errors: cudaGetLastError, cudaGetErrorString
* Device Management: cudaGetDeviceCount, cudaSetDevice, cudaGetDevice, cudaGetDeviceProperties.
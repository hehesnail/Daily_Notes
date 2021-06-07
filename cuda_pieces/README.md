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

### CUDA C Programming Guide
* **cuda device memory**
  * Besides **cudaMalloc** to allocate linear memory, **cudaMallocPitch** & **cudaMalloc3D** for allocations of 2D & 3D arrays(allocation is padded to meet alignment needs).
  * **Device Memory L2 Access Management**: start with cuda 11.0, compute capability 8.0.  the capability to influence persistence of data in the L2 cache, potentially providing higher bandwidth and lower latency accesses to global memory.
  * **shared memory**: Shared memory is expected to be much faster than global memory as mentioned in Thread Hierarchy and detailed in Shared Memory. It can be used as scratchpad memory (or software managed cache) to minimize global memory accesses from a CUDA block, the gemm example.
    * shared_memory block共享内存加速效果还是非常明显的，尤其是在数据有大量重复读取的情形下，通过在 thread block 中声明一块shared_memory 类似 cache, 减少从global memory中读取内存的次数;
    * 从矩阵乘法的两个例子来看，在我的渣笔记本上实现 2048x2048数据规模，block size为 16x16, sharded_memory大概为不分块的 3 倍多.
  * **page locked memory**: 
    * cudaHostAlloc, cudaFreeHost ---> page-locked mem
    * cudaHostRegister ---> page locks a range of memory allocated by malloc.
    * Benifits: 1). asynchronous concurrent execution; 2). mapped memory, elinimate need to copy to/from device memory; 3). write combining memory.
    * **portable memory**: block of page-locked memory can be used in conjunction with any device in the system.  cudaHostAllocPortable / cudaHostRegisterPortable.
    * **write combining memory**: cudaHostAllocWriteCombined, Write-combining memory frees up the host's L1 and L2 cache resources, making more cache available to the rest of the application. 
    * **mapped memory**: cudaHostAllocMapped, cudaHostRegisterMapped, cudaHostGetDevicePointer; 1). no need to allocate a block in device memory and copy data between this block and the block in host memory; 2).no need to use streams to overlap data transfers with kernel execution;
  * **asynchronous concurrent execution**:
    * **streams**: cudaStream_t, a stream is a sequence of commands that execute in order.
    * cudaMemcpyAsync
    * cudaDeviceSynchronize, cudaStreamSynchronize, cudaStreamWaitEvent, cudaStreamQuery
    * for overlapping behavior, see the programming guide.
    * cudaLaunchHostFunc to lanch the host function callback.
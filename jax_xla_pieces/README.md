# 简介

## JAX

JAX = Numpy on CPU/GPU/TPU + 自动微分 + XLA JIT Optimization，JAX为前端，使用XLA作为其后端编译器。

JAX不仅是Numpy on GPU，更多的是对于代码的转换，目的是获取更高效的数值计算代码。主要核心点为3个：1). jit，用于加速函数执行；2). grad，用于反向求导；3). vmap 即vectorizing map，用于自动向量化/批量化。

## XLA

XLA：Accelerated Linear Algebra Compiler，XLA目标：

1. 提高执行速度。编译子计算图以减少短暂运算的执行时间，从而消除 TensorFlow 运行时的开销；融合流水线运算以降低内存开销；并针对已知张量形状执行专门优化以支持更积极的常量传播。
2. 提高内存使用率。分析和安排内存使用，原则上需要消除许多中间存储缓冲区。
3. 降低对自定义运算的依赖。通过提高自动融合的低级运算的性能，使之达到手动融合的自定义运算的性能水平，从而消除对多种自定义运算的需求。
4. 减少移动资源占用量。通过提前编译子计算图并发出可以直接链接到其他应用的对象/头文件对，消除 TensorFlow 运行时。这样，移动推断的资源占用量可降低几个数量级。
5. 提高可移植性，方便为新硬件编写新后端。

XLA 的输入语言称为“HLO IR”或仅为“HLO”(High Level Optimizer)，XLA 接受在 HLO 中定义的计算图并将其编译为适用于各种架构的机器指令。XLA 采用模块化设计，可以轻松融入其他后端以针对某些新颖的硬件架构。TensorFlow 源代码树中包含适用于 x64 和 ARM64 架构的 CPU 后端，以及 NVIDIA GPU 后端。

<div align="center">
<img src="https://github.com/hehesnail/Boring_code/blob/main/imgs/xla.png" width="50%" height="50%" /> 
</div>

如上图所示，XLA 提供了多种与目标无关的优化和分析过程（例如 CSE）、与目标无关的运算融合，以及用于为计算分配运行时内存的缓冲区分析。完成与目标无关的步骤之后，XLA 会将 HLO 计算发送到后端。后端可以执行进一步的 HLO 级优化，而此时将考虑目标特定的信息和需求。最后针对特定目标生成代码。XLA 所含的 CPU 和 GPU 后端使用 LLVM 进行低级 IR、优化和代码生成。这些后端发出有效表示 XLA HLO 计算所需的 LLVM IR，然后调用 LLVM 以从此 LLVM IR 中发出原生代码。从提供的新增硬件的链接来看，当前主要基于LLVM来扩展新的硬件，非LLVM方式则工作量最大。

Useful Links:

XLA Operations： [xla ops](https://www.tensorflow.org/xla/operation_semantics)，各个Ops语义描述，输入输出定义；

XLABuilder定义：[xla builder](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/xla/client/xla_builder.h)，主要通过XLABuilder添加对应HLO指令，xla_op以及xla_computation定义；

XLA 新增硬件方法：[xla new hardware](https://tensorflow.google.cn/xla/developing_new_backend?hl=zh-cn)，目前支持cpu/gpu均通过LLVM完成；

XLA python端接口：[xla python interface](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/xla/python/xla_client.py)，这里 xla python interface通过python module xla_extension引入，xla_extension是通过pybind11调用cxx端实现；

JAX XLA：[lax](https://github.com/google/jax/blob/master/jax/lax.py)，
[xla_bridge](https://github.com/google/jax/blob/master/jax/lib/xla_bridge.py)，
[xla_exec](https://github.com/google/jax/blob/master/jax/interpreters/xla.py)



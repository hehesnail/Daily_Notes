# OSDI
## OSDI 2020

| Title                         |    Field           |   Outcome |
|------                         | ----               | ---       |
| A Unified Architecture for Accelerating Distributed DNN Training in Heterogeneous GPU/CPU Clusters |  distributed training, schedule | BytePS, combination of all reduce & parameter server. |
|Heterogeneity-Aware Cluster Scheduling Policies for Deep Learning Workloads | schedule | Gavel, multi-jobs schedule policy which consider platform heterogeneity.  |
| PipeSwitch: Fast Pipelined Context Switching for Deep Learning Applications | schedule, GPU sharing | PipeSwitch, enbale multiple apps time-share one GPU by pipeline parallelism over PCIE.
|AntMan: Dynamic Scaling on GPU Clusters for Deep Learning | schedule, GPU sharing | AntMan, multi-jobs schedule on GPU cluster |
|Ansor: Generating High-Performance Tensor Programs for Deep Learning | auto-kernel-gen, auto schedule | Ansor, a two stage auto schedule methods in TVM. |
|Rammer: Enabling Holistic Deep Learning Compiler Optimizations with rTasks | ai compiler, auto-kernel-gen | NNFusion, compilatio based schedule for inter-intra op parallel |
|A Tensor Compiler for Unified Machine Learning Prediction Serving | machine learning compiler | Hummingbird, convert ML models to set of tensor ops|
|KungFu: Making Training in Distributed Machine Learning Adaptive | distributed training, schedule | KungFu, adpative policys which monitor training process while adaptively change cofig parameters. |

## OSDI 2021

| Title                         |    Field           |   Outcome |
|------                         | ----               | ---       |
|Pollux: Co-adaptive Cluster Scheduling for Goodput-Optimized Deep Learning | schedule | Pollux, multi-jobs scheduler which monitor training status and re-allocate resources bsed on goodput.  |
| PET: Optimizing Tensor Programs with Partially Equivalent Transformations and Automated Corrections | graph opt | PET, partially equivalent transformations, and then correct the results to restore full equivalence. |
|Dorylus: Affordable, Scalable, and Accurate GNN Training with Distributed CPU Servers and Serverless Threads | GNN opt, serverless | Dorylus, computation seperation for pipeling the graph and tensor tasks. |
|GNNAdvisor: An Adaptive and Efficient Runtime System for GNN Acceleration on GPUs | GNN opt | GNNAdvisor, an adaptive and efficient runtime system to accelerate GNN workloads on GPU platforms.|
|Marius: Learning Massive Graph Embeddings on a Single Machine | GNN opt | Marius, partition caching & buffer-aware data orderings to minimize disk access to maximize utilization. |
|P3: Distributed Deep Graph Learning at Scale | GNN opt| P3, aims on large world graph in distributed settings. eliminate communication & partition overheads, pipelined push-pull parallelism|

## OSDI 2022

| Title                         |    Field           |   Outcome |
|------                         | ----               | ---       |
|SparTA: Deep-Learning Model Sparsity via Tensor-with-Sparsity-Attribute | sparse tensor | Tensor with Sparsity Attribute(TeSA), augment default tensor, allow TeSA flow globally to create highly efficient, specialized operators.|
|ROLLER: Fast and Efficient Tensor Compilation for Deep Learning | ai compiler, auto-kernel-gen | rTile, abstraction encapsulates tensor shapes that align with the key features of the underlying accelerator for quick kernel generation.|
| Walle: An End-to-End, General-Purpose, and Large-Scale Production System for Device-Cloud Collaborative Machine Learning | schedule |device-cloud tasks distributing platform to enhance the alibaba MNN.|
|Unity: Accelerating DNN Training Through Joint Optimization of Algebraic Transformations and Parallelization | graph opt, auto-schedule | Unified parallel computation graph for both algebraic transforms and parallelization.|
|Microsecond-scale Preemption for Concurrent GPU-accelerated DNN Inferences | schedule, gpu kernels | REEF, the first GPU-accelerated DNN inference serving system, enables microsecond-scale kernel preemption. |
|Alpa: Automating Inter- and Intra-Operator Parallelism for Distributed Deep Learning | schedule, auto-kernel-gen, auto-schedule| unify data, operator, and pipeline parallelism and generate parallel strategy automaticly.|
|Looking Beyond GPUs for DNN Scheduling on Multi-Tenant Clusters | schedule | Synergy, resource sensitive scheduler, mult-jobs schedule with workload-aware cpu/memory allocation.|

# SOSP

## SOSP 2019

| Title                         |    Field           |   Outcome |
|------                         | ----               | ---       |
|PipeDream: generalized pipeline parallelism for DNN training | distributed training, pipeline parallelism| PipeDream, a system that adds inter-batch pipelining to intra-batch parallelism |
|A generic communication scheduler for distributed DNN training acceleration | schedule, distributed training | ByteScheduler, partitioning and rearranging the tensor transmissions can result in good performance|
|TASO: optimizing deep learning computation with automatic generation of graph substitutions | graph opt| TASO, automatic graph rewrite generation based on rewrite rules and correntness verification.|


## SOSP 2021

| Title                         |    Field           |   Outcome |
|------                         | ----               | ---       |
| Generating Complex, Realistic Cloud Workloads using Recurrent Neural Networks | DL for system | RNN based cloud workloads modeling|
| Gradient Compression Supercharged High-Performance Data Parallel DNN Training | distributed training, gradient compression | CaSync, incorporate gradient compression in current data parallel tranining framework. |
|HEALER: Relation Learning Guided Kernel Fuzzing | Fuzzing test | Learn relations between system calls and use learned relations to guide input generation and mutation.|

# ASPLOS

## ASPLOS 2020

| Title                         |    Field           |   Outcome |
|------                         | ----               | ---       |
| Interstellar: Using Halide's Scheduling Language to Analyze DNN Accelerators| Mappings, Evalutation benchmark | modify halide compiler to generate hardware, create a system that can fairly compare these prior accelerators.|
| FlexTensor: An Automatic Schedule Exploration and Optimization Framework for Tensor Computation on Heterogeneous System | auto-schedule | Prior ansor schedule exploration and optimization methods.|
| AutoTM: Automatic Tensor Movement in Heterogeneous Memory Systems using Integer Linear Programming | training, NVM | Utilize NVM to imporve the deep learning training.|
|Vortex: Extreme-Performance Memory Abstractions for Data-Intensive Streaming Applications | streaming process| E.G apps: file I/O wrapper, bounded producer-consumer pipeline, vanishing array, key-partitioning engine, and novel in-place radix sort| 
|Capuchin: Tensor-based GPU Memory Management for Deep Learning | training, memory management| Reduces the memory footprint via tensor eviction/prefetching and recomputation. Decisions made by dynamic access patterns tracked at runtime.  |

## ASPLOS 2021

| Title                         |    Field           |   Outcome |
|------                         | ----               | ---       |
|Warehouse-scale Video Acceleration: Co-design and Deployment in the Wild| co-design | the design of a new hardware accelerator building block – a video coding unit (VCU) and the warehouse-scale system architecture |
|Exploiting Gustavson’s Algorithm to Accelerate Sparse Matrix Multiplication | accelerator, sparsity | GAMMA, the Gustavson-Algorithm Matrix-Multiplication Accelerator.|
|A Compiler Infrastructure for Accelerator Generators | hardware generators | Calyx, a new intermediate language (IL) and open-source infrastructure for building compilers that generate hardware accelerators.|
|Vectorization for Digital Signal Processors via Equality Saturation | auto vectorization | dsp program auto vectorization. |
|Mind Mappings: Enabling Efficient Algorithm-Accelerator Mapping Space Search| mapping, search method | A surrogate model based mapping search method.|
|Analytical Characterization and Design Space Exploration for Optimization of CNNs| cost model, search method | presents the first comprehensive analytical modeling for data movement volume for multi-level tiled CNN execution on multi-level memory hierarchy.|
|VEGEN: A Vectorizer Generator for SIMD and Beyond| auto vectorization| introduce Lane Level Parallelism, which captures the type of parallelism implemented by both SIMD and nonSIMD vector instructions, code-generation framework that jointly performs vectorization and vector instruction selection.|

## ASPLOS 2022

| Title                         |    Field           |   Outcome |
|------                         | ----               | ---       |
|TaskStream: accelerating task-parallel workloads by recovering program structure | task parallel | propose a task execution model for accelerators called TaskStream, for handling irregural tasks on CGRA or dataflow architectures.|
|A full-stack search technique for domain optimized deep learning accelerators | hardware generators | defines a broad optimization environment covering key design decisions within the hardware-software stack.|
|RecShard: statistical feature-based memory optimization for industry-scale neural recommendation | DL recommend system |  RecShard, a fine-grained embedding table (EMB) partitioning and placement technique.|
|AStitch: enabling a new multi-dimensional optimization space for memory-intensive ML training and inference on modern SIMT architectures | auto-kernel-gen | AStitch opens a new multi-dimensional optimization space for memory-intensive ML computation, fusion strategy.|
|VELTAIR: towards high-performance multi-tenant deep learning services via adaptive compilation and scheduling | schedule | an adaptive scheduling scheme to ensure resource usage efficiency and reduceconflict rate, adaptive compilation strategy which dynamically pick a program with proper exclusive and shared resource usage | 
|Vector instruction selection for digital signal processors using program synthesis | auto vectorization |  a new algorithm that first abstracts the target platform instructions into high-level uber-instructions, then program synthesis is used to lift input code and lower to machine code.|
|Breaking the computation and communication abstraction barrier in distributed machine learning workloads | distributed training/inference, ai compiler | CoCoNet: 1). DSL to describle computation and communications, 2). optimization passes, 3). compiler for generating comm & comp optimized gpu kernels. |

# PLDI

## PLDI 2022

| Title                         |    Field           |   Outcome |
|------                         | ----               | ---       |
|FreeTensor: A Free-Form DSL with Holistic Optimizations for Irregular Tensor Programs| ai compiler, dsl | Since operator-based programming shows significant limitations for irregular patterns, thus FreeTensor supports redundancy-avoid programming by introducing fine-grained control flow.|
|Exocompilation for Productive Programming of Hardware Accelerators|auto-schedule, x compiler |Exo, based on the principle of exocompilation: externalizing target-specific code generation support and optimization policies to user-level code. Exo allows custom hardware instructions, specialized memories, and accelerator configuration state to be defined in user libraries. |
|DISTAL: The Distributed Tensor Algebra Compiler| distributed compiler |DISTAL, a compiler for dense tensor algebra that targets modern distributed and heterogeneous systems.|
|An Asymptotic Cost Model for Autoscheduling Sparse Tensor Programs|cost model, sparse tensor, auto-schedule |Present the first automatic asymptotic scheduler for sparse tensor programs.|
|All You Need Is Superword-Level Parallelism: Systematic Control-Flow Vectorization with SLP|auto vectorization|SuperVectorization, a new vectorization framework that generalizes SLP vectorization to uncover parallelism that spans different basic blocks and loop nests.|

## PLDI 2021

| Title                         |    Field           |   Outcome |
|------                         | ----               | ---       |
|AKG: Automatic Kernel Generation for Neural Processing Units using Polyhedral Transformations |auto-kernel-gen, ai compiler | AKG leverages polyhedral schedulers to perform a much wider class of transformations, and extends the semantics of the polyhedral representation to combine complex tiling techniques and hierarchical fusion strategies. |
|DNNFusion: Accelerating Deep Neural Networks Execution with Advanced Operator Fusion| graph opt |  To address this challenge, this paper proposes a novel and extensive loop fusion framework called DNNFusion.|
|DeepCuts: a deep learning optimization framework for versatile GPU workloads | auto-kernel-gen |  DeepCuts analyzes the DL workload, groups multiple DL operations into a single GPU kernel, and generates optimized GPU kernels considering kernel implementation parameters and GPU architecture parameters.|

## PLDI 2020 & 2019

| Title                         |    Field           |   Outcome |
|------                         | ----               | ---       |
|LLHD: A Multi-level Intermediate Representation for Hardware Description Languages |mlir, hardware ir| propose the LLHD multilevel IR. LLHD is designed as simple, unambiguous reference description of a digital circuit, yet fully captures existing HDLs.|
|Co-optimizing Memory-Level Parallelism and Cache-Level Parallelism| compiler memory opt| propose compiler support that optimizes both the latencies of last-level cache (LLC) hits and the latencies of LLC misses.|
|Compiling KB-Sized Machine Learning Models to Tiny IoT Devices| ai compiler, iot|SeeDot, a domain-specific language to express ML inference algorithms and a compiler that compiles SeeDot programs to fixed-point code that can efficiently run on constrained IoT devices.|

# CGO
## CGO 2022

| Title                         |    Field           |   Outcome |
|------                         | ----               | ---       |
|SPNC: An Open-Source MLIR-Based Compiler for Fast Sum-Product Network Inference on CPUs and GPUs|mlir, ai compiler | SPNC, the first tool flow for generating fast native code for SPN inference on both CPUs and GPUs, including the use of vectorized/SIMD execution|
|A Compiler Framework for Optimizing Dynamic Parallelism on GPUs|gpu compiler| a compiler framework for optimizing the use of dynamic parallelism in applications with nested parallelism. Three key optimizations: thresholding, coarsening, and aggregation|
|Automatic Horizontal Fusion for GPU Kernels | kernel fusion, auto-kernel-gen | horizontal fusion technique aims to increase the threadlevel parallelism to hide instruction latencies|
|Comprehensive Accelerator-Dataflow Co-design Optimization for Convolutional Neural Networks| co-design, dataflow, search method| develop the first optimization approach that uses analytical modeling and the solution of constrained nonlinear optimization problems for comprehensive algorithm-architecture co-design optimization |
|CompilerGym: Robust, Performant Compiler Optimization Environments for AI Research| research plat, DL for compiler | CompilerGym , a set of environments for real world compiler optimization tasks, and a toolkit for exposing new optimization tasks to compiler researchers|

## CGO 2021 & 2020

| Title                         |    Field           |   Outcome |
|------                         | ----               | ---       |
|Progressive Raising in Multi-level IR | mlir, ir raising| Progressive Raising,raises from lower to higher-level abstractions to leverage domain-specific  transformations for low-level representations|
|Compiling Graph Applications for GPUs with GraphIt| graph compiler | G2, an extension to the GraphIt compiler framework, that achieves high performance on both CPUs and GPUs using the same algorithm specification|
|UNIT: Unifying Tensorized Instruction Compilation| auto-tensorization |UNIT, to unify the compilation for tensorized instructions.|
|MLIR: A Compiler Infrastructure for the End of Moore’s Law|mlir | MLIR aims to address software fragmentation, improve compilation for heterogeneous hardware, significantly reduce the cost of building domain specific compilers, and aid in connecting existing compilers together.|
|StencilFlow: Mapping Large Stencil Programs to Distributed Spatial Computing Systems| stencil computation, distributed compiler, heter compilation| mapping directed acyclic graphs of heterogeneous stencil computations to spatial computing systems, assuming large input programs without an iterative component. |
|Optimizing Ordered Graph Algorithms with GraphIt | graph compiler | DSL to simplify writing high-performance parallel ordered graph algorithms|
|Automatic Generation of High-Performance Quantized Machine Learning Kernels| auto-kernel-gen, quantization| new automated approach to implementing quantized inference for machine learning models.|
|ATMem: adaptive data placement in graph applications on heterogeneous memories | graph comp, NVM, memory management | ATMem—a runtime framework for adaptive granularity data placement optimization in graph applications. |

# ATC

| Title                         |    Field           |   Outcome |
|------                         | ----               | ---       |
|Whale: Efficient Giant Model Training over Heterogeneous GPUs| distributed training | Whale, a general and efficient distributed training framework for giant models, introduces two new high-level primitives to express all existing parallel strategies as well as their hybrids.|
|PetS: A Unified Framework for Parameter-Efficient Transformers Serving|DL servering | PetS, the first unified framework for multi-task PETs serving, different PET tasks are expressed by a unified representation in the same framework, which enables flexible PET task management.|
|Fine-tuning giant neural networks on commodity hardware with automatic pipeline model parallelism| pipeline parallelism | FTPipe, a system that explores a new dimension of pipeline model parallelism, making multi-GPU execution of fine-tuning tasks for giant neural networks readily accessible on commodity hardware|

# CC

| Title                         |    Field           |   Outcome |
|------                         | ----               | ---       |
|IR Design for Heterogeneity: Challenges and Opportunities | mlir | talk slides|
|Integrating a Functional Pattern-Based IR into MLIR| mlir | RISE IR, the first time a practical integration of a functional pattern-based IR with other IRs and it enables the construction of sophisticated code generators for domain specific languages.|
|One-Shot Tuner for Deep Learning Compilers | Auto tuning |One-Shot Tuner, take a neural predictor inspired approach to reduce the auto-tuning overhead and show that a performance predictor model trained prior to compilation can produce optimized tensor operation codes without repeated search and hardware  measurements. |
|MLIR-based code generation for GPU tensor cores |mlir, gpu kernel-gen | build a transformation and lowering pipeline to automatically generate near-peak performance code for matrix-matrix multiplication |
|Graph Transformations for Register-Pressure-Aware Instruction Scheduling| instruction schedule | propose graph transformations for the RP minimization objective |
|Caviar: An E-Graph Based TRS for Automatic Code Optimization| term rewrite, e-graph | Caviar, an e-graphbased TRS for proving expressions within compilers. |
|Automating reinforcement learning architecture design for code optimization | DL for compiler | SuperSonic, a new open-source framework to allow compiler developers to integrate RL into compilers easily. |
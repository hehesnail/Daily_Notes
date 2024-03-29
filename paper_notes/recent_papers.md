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

## ASPLOS 2023

| Title                         |    Field           |   Outcome |
|------                         | ----               | ---       |
|TensorIR: An Abstraction for Automatic Tensorized Program Optimization | tvm, ir design, auto-tensorization|TensorIR generalizes the loop nest representation used in existing machine learning compilers to bring tensor computation as the first-class citizen. TensorIR compilation automatically uses the tensor computation primitives for given hardware backends. |
|FLAT: An Optimized Dataflow for Mitigating Attention Bottlenecks|attention opt, mapping| FLAT, a dataflow processes costly attention operations through a unique fusion mechanism, transforming the memory footprint quadratic growth to merely a linear one|
|Homunculus: Auto-Generating Eficient Data-Plane ML Pipelines for Datacenter Networks|ML service, automated ML pipeline| Homunculus takes as input, the training data and accompanying network constraints, and automatically generates and installs a suitable model onto the underlying switching hardware|
|Mobius: Fine Tuning Large-Scale Models on Commodity GPU Servers|distributed training|A novel pipeline parallelism scheme enabling heterogeneous memory for large-scale model training, while bringing fewer communications than existing systems|
|Betty: Enabling Large-Scale GNN Training with Batch-Level Graph Partitioning|GNN training|Betty introduces two novel techniques, redundancy-embedded graph (REG) partitioning and memory-aware partitioning, to effectively mitigate the redundancy and load imbalances issues across the partitions.|
|SparseTIR: Composable Abstractions for Sparse Compilation in Deep Learning|ir desing, sparse|a sparse tensor compilation abstraction that offers composable formats and composable transformations for deep learning workloads.|
|Hidet: Task-Mapping Programming Paradigm for Deep Learning Tensor Programs|ai compiler, schedule|propose to embed the scheduling process into tensor programs and use dedicated mappings, called task mappings, to define the computation assignment and ordering directly in the tensor programs.|
|TiLT: A Time-Centric Approach for Stream Query Optimization and Parallelization|stream process|TiLT, a novel intermediate representation (IR) that offers a highly expressive temporal query language amenable to effective query optimization and parallelization strategies.|
|STI: Turbocharge NLP Inference at the Edge via Elastic Pipelining|edge inference|propose Speedy Transformer Inference (STI). Built on the key idea of maximizing IO/compute resource utilization on the most important parts of a model|
|TelaMalloc: Efficient On-Chip Memory Allocation for Production Machine Learning Accelerators|memory management, ml compiler| demonstrate a new method for solving the memory allocation problem on machine learning accelerators. Our approach combines heuristics with a solver-based approach to explore a complex search space more efficiently|
|WACO: Learning Workload-Aware Co-optimization of the Format and Schedule of a Sparse Tensor Program|sparse, cost model| present WACO, a novel method of co-optimizing the format and the schedule of a given sparsity pattern in a sparse tensor program|
|Overlap Communication with Dependent Computation via Decomposition in Large Deep Learning Models|large model serving|a novel technique to effectively reduce its data communication overheads by overlapping communication with computation|

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

## CGO 2023
| Title                         |    Field           |   Outcome |
|------                         | ----               | ---       |
|Accelerating Deep Neural Networks on Mobile Multicore NPUs|SoCs, schedule|  propose parallelization and pipelining models for neural networks on mobile multicore NPU systems. We introduce key optimization criteria — data redundancy, data dependency, data reusability, data exchange and computation redundancy, and relevant optimizations such as stratum construction, halo exchange, and halo-first policy.|
|Bridging Control-Centric and Data-Centric Optimization|ir design, mlir|Introducing data-centric optimizations to a standard multi-level compilation pipeline through an MLIR dialect|
|Code Generation for In-Place Stencils|mlir|propose the first domain-specific code generator for iterative in-place stencils|
|Flexer: Out-of-Order Scheduling for Multi-NPUs|inst schedule, ai compiler| Flexer, an out-of-order (OoO) scheduler that maximizes instruction-level parallelism and data reuse on such multi-NPU systems. |
|Pin or Fuse? Exploiting Scratchpad Memory to Reduce Off-Chip Data Transfer in DNN Accelerators|schedule, ai compiler| propose a compiler technique to generate code that utilizes both pinning and fusion to minimize execution latency of a DNN model.|
|To Pack or Not to Pack: A Generalized Packing Analysis and Transformation|Gemm opt, mlir|proposes GPAT, a generalized packing analysis and code transformation that applies packing, when beneficial, to a generic input loop nest.|

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
|ATMem: adaptive data placement in graph applications on heterogeneous memories | graph computing, NVM, memory management | ATMem—a runtime framework for adaptive granularity data placement optimization in graph applications. |

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

# EuroSys
## EuroSys 2020 & 2021
| Title                         |    Field           |   Outcome |
|------                         | ----               | ---       |
|Balancing Efficiency and Fairness in Heterogeneous GPU Clusters for Deep Learning| schedule | Gandivafair, a distributed, fair share scheduler that balances conflicting goals of efficiency and fairness in GPU clusters for deep learning training. |
|Subway: Minimizing Data Transfer during Out-of-GPU-Memory Graph Processing|graph computing | Subway, the first solution that only loads active edges of the graph to the GPU memory, design a fast subgraph generation algorithm with a simple yet efficient subgraph representation and a GPU-accelerated implementation. |
|Peregrine: A Pattern-Aware Graph Mining System | graph computing | PEREGRINE, a pattern-aware graph mining system that directly explores the subgraphs of interest while avoiding exploration of unnecessary subgraphs, and simultaneously bypassing expensive computations throughout the mining process. |
|Improving Resource Utilization by Timely Fine-Grained Scheduling| schedule | Ursa, enables the scheduler to capture accurate resource demands dynamically from the execution runtime and to provide timely, fine-grained resource allocation based on monotasks. |
|Accelerating Winograd Convolutions using Symbolic Computation and Meta-programming| gpu kernel-gen | optimize Winograd convolutions based on symbolic computation, a system to automate the generation of efficient and portable Winograd convolution code for various GPUs.|
|DGCL: An Efficient Communication Library for Distributed GNN Training| GNN opt | propose the distributed graph communication library (DGCL) for efficient GNN training on multiple GPUs.|
|Seastar: Vertex-Centric Programming for Graph Neural Networks |GNN opt| Seastar system, a vertex-centric programming model for GNN training on GPU and provides idiomatic python constructs to enable easy development of novel homogeneous and heterogeneous GNN models. |
|Profiling Dataflow Systems on Multiple Abstraction Levels|dataflow graph, profiling | profile compiling dataflow systems at higher abstraction levels. Our approach tracks the code generation process and aggregates profiling data to any abstraction level.|
|Accelerating Graph Sampling for Graph Machine Learning using GPUs|GNN opt| NextDoor, employs a new approach to graph sampling that we call transit-parallelism, which allows load balancing and caching of edges|
|Tahoe: Tree Structure-Aware High Performance Inference Engine for Decision Tree Ensemble on GPU |ML opt|Tahoe, rearranges tree nodes to enable efficient and coalesced memory accesses, also rearranges trees, such that trees with similar structures are grouped together in memory and assigned to threads in a balanced way.|
|Tesseract: Distributed, General Graph Pattern Mining on Evolving Graphs| graph computing |  Tesseract scales out by decomposing a stream of graph updates into perupdate mining tasks and dynamically assigning these tasks to a set of distributed workers.|

## EuroSys 2022
| Title                         |    Field           |   Outcome |
|------                         | ----               | ---       |
|Unicorn: Reasoning about Configurable System Performance through the Lens of Causality |ML for Sys | Unicorn–a methodology that enables reasoning about configurable system performance with causal inference and counterfactual reasoning|
|LiteReconfig: Cost and Content Aware Reconfiguration of Video Object Detection Systems for Mobile GPUs| schedule | LiteReconfig features a cost-benefit analyzer to decide which features to use, and which execution branch to run, at inference time. |
|Fleche: An Efficient GPU Embedding Cache for Personalized Recommendations|memory management | Fleche uses one cache backend for all embedding tables to improve the total cache utilization, and merges small kernel calls into one unitary call to reduce the overhead of kernel maintenance. |
|GNNLab: a factored system for sample-based GNN training over GPUs | GNN opt, training | GNNLab, a sample-based GNN training system in a single machine multi-GPU setup. |
|Out-of-order backprop: an effective scheduling technique for deep learning | schedule, training | Out-of-order (ooo) back-prop, exploiting the dependencies of gradient computations, enables to reorder their executions to make the most of the GPU resources. |
|D3: A Dynamic Deadline-Driven Approach for Building Autonomous Vehicles|Autonomous vehicles system | D3 (Dynamic Deadline-Driven), a novel execution model that centralizes the deadline management, and allows applications to adjust their computation by modeling missed deadlines as exceptions, design and implement ERDOS, an opensource realization of D3 for AV pipelines. |
|Varuna: Scalable, Low-cost Training of Massive Deep Learning Models| schedule, DL training | Varuna a new system that enables training massive deep learning models on commodity networking, makes thrifty use of networking resources and automatically configures the user’s training job to efficiently use any given set of resources.|
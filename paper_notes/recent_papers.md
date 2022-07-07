# OSDI
## OSDI 2020

| Title                         |    Field           |   Outcome |
|------                         | ----               | ---       |
| A Unified Architecture for Accelerating Distributed DNN Training in Heterogeneous GPU/CPU Clusters |  distributed training, schedule | BytePS, combination of all reduce & parameter server. |
|Heterogeneity-Aware Cluster Scheduling Policies for Deep Learning Workloads | schedule | Gavel, multi-jobs schedule policy which consider platform heterogeneity.  |
| PipeSwitch: Fast Pipelined Context Switching for Deep Learning Applications | schedule, GPU sharing | PipeSwitch, enbale multiple apps time-share one GPU by pipeline parallelism over PCIE.
|AntMan: Dynamic Scaling on GPU Clusters for Deep Learning | schedule, GPU sharing | AntMan, multi-jobs schedule on GPU cluster |
|Ansor: Generating High-Performance Tensor Programs for Deep Learning | kernel generation, auto schedule | Ansor, a two stage auto schedule methods in TVM. |
|Rammer: Enabling Holistic Deep Learning Compiler Optimizations with rTasks | ai compiler, op parallelism | NNFusion, compilatio based schedule for inter-intra op parallel |
|A Tensor Compiler for Unified Machine Learning Prediction Serving | machine learning compiler | Hummingbird, convert ML models to set of tensor ops|
|KungFu: Making Training in Distributed Machine Learning Adaptive | distributed training, adaptive parameters cofig | KungFu, adpative policys which monitor training process while adaptively change cofig parameters. |

## OSDI 2021

| Title                         |    Field           |   Outcome |
|------                         | ----               | ---       |
|Pollux: Co-adaptive Cluster Scheduling for Goodput-Optimized Deep Learning | schedule | Pollux, multi-jobs scheduler which monitor training status and re-allocate resources bsed on goodput.  |
|Oort: Efficient Federated Learning via Guided Participant Selection | federated learning |Oort, improve the performance of federated training and testing  |
| PET: Optimizing Tensor Programs with Partially Equivalent Transformations and Automated Corrections | graph rewrite | PET, partially equivalent transformations, and then correct the results to restore full equivalence. |
|Dorylus: Affordable, Scalable, and Accurate GNN Training with Distributed CPU Servers and Serverless Threads | GNN optimization, serverless | Dorylus, computation seperation for pipeling the graph and tensor tasks. |
|GNNAdvisor: An Adaptive and Efficient Runtime System for GNN Acceleration on GPUs | GNN optimization | GNNAdvisor, an adaptive and efficient runtime system to accelerate GNN workloads on GPU platforms.|
|Marius: Learning Massive Graph Embeddings on a Single Machine | GNN optimization | Marius, partition caching & buffer-aware data orderings to minimize disk access to maximize utilization. |
|P3: Distributed Deep Graph Learning at Scale | GNN optimization| P3, aims on large world graph in distributed settings. eliminate communication & partition overheads, pipelined push-pull parallelism|

## OSDI 2022

| Title                         |    Field           |   Outcome |
|------                         | ----               | ---       |
|SparTA: Deep-Learning Model Sparsity via Tensor-with-Sparsity-Attribute | sparse tensors | Tensor with Sparsity Attribute(TeSA), augment default tensor, allow TeSA flow globally to create highly efficient, specialized operators.|
|ROLLER: Fast and Efficient Tensor Compilation for Deep Learning | ai compiler, kernel generation | rTile, abstraction encapsulates tensor shapes that align with the key features of the underlying accelerator for quick kernel generation.|
| Walle: An End-to-End, General-Purpose, and Large-Scale Production System for Device-Cloud Collaborative Machine Learning | schedule |device-cloud tasks distributing platform to enhance the alibaba MNN.|
|Unity: Accelerating DNN Training Through Joint Optimization of Algebraic Transformations and Parallelization | graph rewrite, parallel strategy | Unified parallel computation graph for both algebraic transforms and parallelization.|
|Microsecond-scale Preemption for Concurrent GPU-accelerated DNN Inferences | schedule, gpu kernels | REEF, the first GPU-accelerated DNN inference serving system, enables microsecond-scale kernel preemption. |
|Alpa: Automating Inter- and Intra-Operator Parallelism for Distributed Deep Learning | schedule, auto-schedule, DL parallelism| unify data, operator, and pipeline parallelism and generate parallel strategy automaticly.|
|Looking Beyond GPUs for DNN Scheduling on Multi-Tenant Clusters | schedule | Synergy, resource sensitive scheduler, mult-jobs schedule with workload-aware cpu/memory allocation.|

# SOSP

## SOSP 2019

| Title                         |    Field           |   Outcome |
|------                         | ----               | ---       |
|PipeDream: generalized pipeline parallelism for DNN training | distributed training, pipeline perallelism| PipeDream, a system that adds inter-batch pipelining to intra-batch parallelism |
|A generic communication scheduler for distributed DNN training acceleration | schedule, distributed training | ByteScheduler, partitioning and rearranging the tensor transmissions can result in good performance|
|TASO: optimizing deep learning computation with automatic generation of graph substitutions | graph rewrite | TASO, automatic graph rewrite generation based on rewrite rules and correntness verification.|


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
|Exploiting Gustavson’s Algorithm to Accelerate Sparse Matrix Multiplication | accelerator | GAMMA, the Gustavson-Algorithm Matrix-Multiplication Accelerator.|
|A Compiler Infrastructure for Accelerator Generators | hardware generators | Calyx, a new intermediate language (IL) and open-source infrastructure for building compilers that generate hardware accelerators.|
|Vectorization for Digital Signal Processors via Equality Saturation | auto vectorization | dsp program auto vectorization. |
|Mind Mappings: Enabling Efficient Algorithm-Accelerator Mapping Space Search| mapping | A surrogate model based mapping search method.|
|Analytical Characterization and Design Space Exploration for Optimization of CNNs| cost model, search method | presents the first comprehensive analytical modeling for data movement volume for multi-level tiled CNN execution on multi-level memory hierarchy.|
|VEGEN: A Vectorizer Generator for SIMD and Beyond| auto vectorization| introduce Lane Level Parallelism, which captures the type of parallelism implemented by both SIMD and nonSIMD vector instructions, code-generation framework that jointly performs vectorization and vector instruction selection.|

## ASPLOS 2022

| Title                         |    Field           |   Outcome |
|------                         | ----               | ---       |
|TaskStream: accelerating task-parallel workloads by recovering program structure | task parallel | propose a task execution model for accelerators called TaskStream, for handling irregural tasks on CGRA or dataflow architectures.|
|A full-stack search technique for domain optimized deep learning accelerators | hardware generators | defines a broad optimization environment covering key design decisions within the hardware-software stack.|
|RecShard: statistical feature-based memory optimization for industry-scale neural recommendation | DL recommend system |  RecShard, a fine-grained embedding table (EMB) partitioning and placement technique.|
|AStitch: enabling a new multi-dimensional optimization space for memory-intensive ML training and inference on modern SIMT architectures | kernel generation | AStitch opens a new multi-dimensional optimization space for memory-intensive ML computation, fusion strategy.|
|VELTAIR: towards high-performance multi-tenant deep learning services via adaptive compilation and scheduling | schedule | an adaptive scheduling scheme to ensure resource usage efficiency and reduceconflict rate, adaptive compilation strategy which dynamically pick a program with proper exclusive and shared resource usage | 
|Vector instruction selection for digital signal processors using program synthesis | auto vectorization |  a new algorithm that first abstracts the target platform instructions into high-level uber-instructions, then program synthesis is used to lift input code and lower to machine code.|
|Breaking the computation and communication abstraction barrier in distributed machine learning workloads | distributed training/inference, compiler based optimization | CoCoNet: 1). DSL to describle computation and communications, 2). optimization passes, 3). compiler for generating comm & comp optimized gpu kernels. |
## HASCO: Towards Agile HArdware and Software CO-design for Tensor Computation (ISCA 2021)

Basic idea: develop a holistic solution that is a combination of hardware acceleration and software mapping.

Aspect: tensor computation(intrinsic) HW/SW Co-design.

### Contributions:
1. propose HASCO to co-design hardware accelerators and software mapping in concert. HASCO offers a holistic solution to tensor computations.
2. propose efficient algorithms to explore the hardware-software interface (tensorize).
3. develop heuristic, Q-learning, and Bayesian optimization algorithms to explore the design spaces efficiently.

### Framework:

<div align="center">
<img src="https://github.com/hehesnail/Daily_Notes/blob/main/imgs/co_design_imgs/hasco_fig1.png" width="70%" height="70%" /> 
</div>

### HW/SW Partition
1. Tensorize Choices

<div align="center">
<img src="https://github.com/hehesnail/Daily_Notes/blob/main/imgs/co_design_imgs/hasco_fig2.png" width="70%" height="70%" /> 
</div>

2. Partition Space Generation: lowers both tensor computations and intrinsics into TSTs and performs a two-step approach: index matching and structure matching

<div align="center">
<img src="https://github.com/hehesnail/Daily_Notes/blob/main/imgs/co_design_imgs/hasco_fig3.png" width="70%" height="70%" /> 
</div>

### Hardware Generation
1. use a sequence of the parametric hardware primitives to form the skeleton of a spatial accelerator, and the primitive factors (accelerator parameters) compose the design space.
2. design space is composed of the following parameters: [scratchpad size, # scratchpad banks, local memory size, burst length of DMAC, maximal transfer size of DMAC, dataflow, PE array shape].
3. develop a Chisel generator in HASCO, which translates the four common intrinsics (GEMV, GEMM, convolution, and dot product) and the hardware primitives into spatial accelerators.

<div align="center">
<img src="https://github.com/hehesnail/Daily_Notes/blob/main/imgs/co_design_imgs/hasco_fig4.png" width="70%" height="70%" /> 
</div>

<div align="center">
<img src="https://github.com/hehesnail/Daily_Notes/blob/main/imgs/co_design_imgs/hasco_fig5.png" width="70%" height="70%" /> 
</div>

### Expriments
1. Benchmark: MTTKRP, TTM, CONV2D, GEMM.
2. Hardware: Gemmini to generate GEMM accelerators. We use the Rocket Chip generator and our Chisel generator to build accelerators with the other intrinsics.
3. compare HASCO with AutoTVM and an accelerator library.
4. Metrics: Maestro and prototype accelerators as Rocket Chip SoCs on a Xilinx VU9P FPGA board.

## Understanding Reuse, Performance, and Hardware Cost of DNN Dataflows: A Data-Centric Approach Using MAESTRO (MICRO 2019)

Basic idea: the DNN dataflows for spatial acceleartor is predictiable if carefully analyze the data reuse patterns.

Aspect: Analytic cost model for spatial accelerator

### Spatial Accelerator & Dataflow

<div align="center">
<img src="https://github.com/hehesnail/Daily_Notes/blob/main/imgs/co_design_imgs/spatial_acc.png" width="70%" height="70%" /> 
</div>

**Spatial DNN accelerators** employ hundreds of processing elements (PEs) to exploit inherent parallelism in DNN applications. PEs typically include scratchpad memories (L1) and ALUs that perform multiply-accumulate operations (MACs). To reduce energy and time-consuming DRAM accesses, most DNN accelerators also include a shared scratchpad buffer (L2) large enough to stage data to feed all the PEs. Shared L2 buffer and PEs are interconnected with a network-on-chip(NoC).

**Dataflow**: the data partitioning and scheduling strategies used by DNN accelerators to leverage reuse and perform staging. More concisely, (1). how we schedule the DNN computations (e.g., choice of loop transformations) and (2) how we map computations across PEs.

**Performance and energy efficiency**: (1) target DNN model and its layers types/dimensions, (2) dataflow, and (3) available hardware resources and their connectivity.

### Contributions:
1. introduce a **data-centric** notation to represent various accelerator dataflows with data mappings and reuses being first-class entities.
2. show how data-centric directives can be used to reason about the reuse.
3. propose analytical cost model named MAESTRO with DNN model, dataflow description and hardware configuration in, output stimates of end-to-end execution time, energy (including all compute, buffer, and interconnect activities), NoC costs, and so on

### Data Reuse Taxonomy
* Multicasting:
    1. spatial multicasting: patially replicates the data point via wires, and delivers the data point to multiple spatial destinations
    2. temporal multicasting: temporally replicates the data point via a smaller local buffer, and delivers the data point to multiple temporal destinations (i.e., different time instances) at the same PE.
* Redution:
    1. spatial reduction: accumulates partial outputs from multiple spatial sources and spatially accumulates them via multiple compute units.
    2. temporal reduction: temporal reduction accumulates partial outputs from multiple
temporal sources

<div align="center">
<img src="https://github.com/hehesnail/Daily_Notes/blob/main/imgs/co_design_imgs/maestro_fig1.png" width="70%" height="70%" /> 
</div>

### Data-Centric Representation

Target of dataflow: 
1. the schedule of DNN computations (e.g., choice of loop transformations) across time for exploiting a wide range of reuse. 
2. the mapping of the DNN computations across PEs for parallelism.

**Spatial Map(size, offset) α**: specifies a distribution of dimension α (e.g., R, X) of a data structure across PEs.

**Temporal Map(size, offset) α**: specifies a distribution of dimension α of a data structure across time steps in a PE.

**Cluster(N)**: logically groups multiple PEs or nested sub-clusters. all the mapping directives above a CLUSTER directive see logical clusters while those below the CLUSTER directive see inside of each logical cluster. Enable multi-dimensional spatial distributions.

### Hardware Implementation of Reuse

<div align="center">
<img src="https://github.com/hehesnail/Daily_Notes/blob/main/imgs/co_design_imgs/maestro_fig3.png" width="70%" height="70%" /> 
</div>

### MAESTRO Framework

<div align="center">
<img src="https://github.com/hehesnail/Daily_Notes/blob/main/imgs/co_design_imgs/maestro_fig4.png" width="70%" height="70%" /> 
</div>

### Experiments - Dataflow tradeoffs

<div align="center">
<img src="https://github.com/hehesnail/Daily_Notes/blob/main/imgs/co_design_imgs/maestro_fig5.png" width="70%" height="70%" /> 
</div>

<div align="center">
<img src="https://github.com/hehesnail/Daily_Notes/blob/main/imgs/co_design_imgs/maestro_fig6.png" width="70%" height="70%" /> 
</div>

<div align="center">
<img src="https://github.com/hehesnail/Daily_Notes/blob/main/imgs/co_design_imgs/maestro_fig7.png" width="70%" height="70%" /> 
</div>

### Experiments - Hardware Design-Parameters and Implementation Analysis

<div align="center">
<img src="https://github.com/hehesnail/Daily_Notes/blob/main/imgs/co_design_imgs/maestro_fig8.png" width="70%" height="70%" /> 
</div>

### Furture direction
Plan to leverage MAESTRO to implement a **dataflow auto-tuner** to find an optimal dataflow on the specified DNN model and hardware configuration. With the optimal dataflow, we plan to extend our infrastructure to **automatically generate RTL**, facilitating end-to-end DNN acceleration flow.

## Union: A Unified HW-SW Co-Design Ecosystem in MLIR for Evaluating Tensor Operations on Spatial Accelerators (PACT 2021)

Basic idea: current analytic cost model for spatital accelerator lies in single domain which is not practical in other domain, thus based on mlir, one can reuse the already aviable models via unified IR.

Aspect: Analytic model for spatial accelerator reusing.

### Contributions
* provide a plug-and-play unified ecosystem to quickly evaluate tensor operations in various domains such as ML and HPC on spatial accelerators leveraging the MLIR infrastructure.
* introduce new unified abstractions to describe tensor operations and their mappings on spatial accelerators to integrate different mappers and cost models.
* introduce operation-level/loop-level analysis to identify operations to be evaluated with the target spatial accelerator using a cost model.

### Framework

<div align="center">
<img src="https://github.com/hehesnail/Daily_Notes/blob/main/imgs/co_design_imgs/union_fig1.png" width="70%" height="70%" /> 
</div>

### Union Abstractions

Weakness of compute-centric reprensentation

<div align="center">
<img src="https://github.com/hehesnail/Daily_Notes/blob/main/imgs/co_design_imgs/union_fig2.png" width="70%" height="70%" /> 
</div>

<div align="center">
<img src="https://github.com/hehesnail/Daily_Notes/blob/main/imgs/co_design_imgs/union_fig3.png" width="70%" height="70%" /> 
</div>

<div align="center">
<img src="https://github.com/hehesnail/Daily_Notes/blob/main/imgs/co_design_imgs/union_fig4.png" width="70%" height="70%" /> 
</div>

## METRO: A Software-Hardware Co-Design of Interconnections for Spatial DNN Accelerators (arxiv 2021)

Basic idea: decoupling the traffic scheduling policies from hardware fabrics and moving them to the software level.

Aspect: NoC HW/SW co-design.

### Contributions
* identify the inefficiency of traditional NoCs and clarify their fundamental drawbacks.
* propose, design, and implement METRO , a software-hardware co-designed interconnection to achieve higher data transmission efficiency.
* up to 73.6% overall processing time reduction.

### Tiled Spatial DNN Accelerators

<div align="center">
<img src="https://github.com/hehesnail/Daily_Notes/blob/main/imgs/co_design_imgs/metro_fig1.png" width="70%" height="70%" /> 
</div>

As the computation capacity of a spatial accelerator scales up, processing a single DNN layer using all tiles is becoming less efficient.

Thus, needs to process multiple layers from a DNN model on high throughput spatial accelerator simultaneously. It means that a spatial accelerator is partitioned into multiple disjoint regions, which may operate under different dataflows. Thus, tiles in these regions may also need different interconnections.

### Traffic types & Contention

<div align="center">
<img src="https://github.com/hehesnail/Daily_Notes/blob/main/imgs/co_design_imgs/metro_fig2.png" width="70%" height="70%" /> 
</div>

<div align="center">
<img src="https://github.com/hehesnail/Daily_Notes/blob/main/imgs/co_design_imgs/metro_fig3.png" width="70%" height="70%" /> 
</div>

### METRO Overview

* Context infos: 1) the source and the destination of each traffic pattern and 2) the timing when each traffic pattern is issued.

<div align="center">
<img src="https://github.com/hehesnail/Daily_Notes/blob/main/imgs/co_design_imgs/metro_fig4.png" width="70%" height="70%" /> 
</div>

* Software traffic scheduling framework
  * Input: the workload descriptions and dataflow specification. 
  * Routing problem: which paths do the traffic flows take from source to the destination.
  * Flow control problem: when are the flows injected into the network.
  * Output: resulted scheduling policies are dumped as configurations of hardware fabrics.

<div align="center">
<img src="https://github.com/hehesnail/Daily_Notes/blob/main/imgs/co_design_imgs/metro_fig5.png" width="70%" height="70%" /> 
</div>

* Dual-Phase Routing
  * “Hub” selection: the tile which has the minimum Manhattan distance from the source (for Multicast) or destination (for Reduce).
  * Phase-1 Routing: Evolutionary Algorithm (EA) to search a sequence of intermediate nodes, and employ the X-Y routing to determine the path between two intermediate nodes.
  * Phase-2 Routing: BFS to build the spanning tree rooted from the “hub” for the lowest propagation depth, perform tree-based multicast distribute the flow to all destinations.

* Hardware Design

<div align="center">
<img src="https://github.com/hehesnail/Daily_Notes/blob/main/imgs/co_design_imgs/metro_fig6.png" width="70%" height="70%" /> 
</div>

### Evaluation Setup

<div align="center">
<img src="https://github.com/hehesnail/Daily_Notes/blob/main/imgs/co_design_imgs/metro_fig7.png" width="70%" height="70%" /> 
</div>

* Estimate the computation latency of processing tiles: Timeloop
* Cost model of tiles in terms of area and energy: Accelergy
* Estimate the energy and area of primary logical components: gem5-Alladin
* Evaluate the SRAM buffers: CACTI
* Hardware parameters: NVDLA Cores, scale up to 512 GOPs.
* Estimate the performance of traditional on-chip networks: Booksim2

## AMOS: Enabling Automatic Mapping for Tensor Computations On Spatial Accelerators with Hardware Abstraction (ISCA 2022)

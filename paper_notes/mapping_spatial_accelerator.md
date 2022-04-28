## Understanding Reuse, Performance, and Hardware Cost of DNN Dataflows: A Data-Centric Approach Using MAESTRO (MICRO 2020)

Aspect: Analytic cost model for spatial accelerator

### Spatial Accelerator & Dataflow

<div align="center">
<img src="https://github.com/hehesnail/Daily_Notes/blob/main/imgs/../../../../../../../../imgs/co_design_imgs/spatial_acc.png" width="100%" height="100%" /> 
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
<img src="https://github.com/hehesnail/Daily_Notes/blob/main/imgs/../../../../../../../../imgs/co_design_imgs/maestro_fig1.png" width="100%" height="100%" /> 
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
<img src="https://github.com/hehesnail/Daily_Notes/blob/main/imgs/../../../../../../../../imgs/co_design_imgs/maestro_fig3.png" width="100%" height="100%" /> 
</div>

### MAESTRO Framework

<div align="center">
<img src="https://github.com/hehesnail/Daily_Notes/blob/main/imgs/../../../../../../../../imgs/co_design_imgs/maestro_fig4.png" width="100%" height="100%" /> 
</div>

## LISA: Graph Neural Network based Portable Mapping on Spatial Accelerators (HPCA 2022)

## Union: A Unified HW-SW Co-Design Ecosystem in MLIR for Evaluating Tensor Operations on Spatial Accelerators (PACT 2021)

## METRO: A Software-Hardware Co-Design of Interconnections for Spatial DNN Accelerators (arxiv 2021)

## Taming the Zoo: The Unified GraphIt Compiler Framework for Novel Architectures (ISCA 2021)
### Contributions:
* Propose a compiler framework with a novel and carefully designed intermediate representation, GraphIR; hardware-independent passes; and hardware-specific GraphVMs to generate fast code on diverse architectures.
* A novel extensible scheduling language that allows programmers to explore the optimization spaces of different hardware platforms.
* Implementations of four GraphVMs that can generate efficient code for CPUs, GPUs, Swarm, and the HammerBlade Manycore.
<div align="center">
<img src="https://github.com/hehesnail/Boring_code/blob/main/imgs/ugc_arch.png" width="70%" height="70%" /> 
</div>

### Interesting hardware architectures
This paper includes two novel architectures, Swarm and HammerBlade Manycore, details ref to the paper.
<div align="center">
<img src="https://github.com/hehesnail/Boring_code/blob/main/imgs/ugc_swarm.png" width="70%" height="70%" /> 
</div>

<div align="center">
<img src="https://github.com/hehesnail/Boring_code/blob/main/imgs/ugc_hammerblade.png" width="70%" height="70%" /> 
</div>

<div align="center">
<img src="https://github.com/hehesnail/Boring_code/blob/main/imgs/ugc_hardware_summary.png" width="70%" height="70%" /> 
</div>

### Graph IR & VM
* Graph IR:
<div align="center">
<img src="https://github.com/hehesnail/Boring_code/blob/main/imgs/ugc_ir.png" width="70%" height="70%" /> 
</div>

* Graph VM:
  * **CPU:** edge-based and vertex-based traversals, different representations for priority queue data structures, cache and NUMA optimizations, vertex data array of struct and struct of array transformations, among others.
  * **GPU:** Load-balancing runtime library, kernel fusion, edgeBlocking and fused vs. unfused frontier creation.
  * **Swarm:** vertex sets to tasks, shared to private state, fine-grained splitting and spatial hints
  * **HammerBlade:** blocked access optimization, alignment-based partitioning etc.

### Schedule
UGC creates an abstract interface with virtual functions for all of the information that the hardware-independent compiler needs, implement new scheduling object classes for each GraphVM by inheriting from this abstract interface.
<div align="center">
<img src="https://github.com/hehesnail/Boring_code/blob/main/imgs/ugc_schedule.png" width="70%" height="70%" /> 
</div>

### Experiments
* Hardwares:
  * CPU: Intel Xeon E5-2695 v3 12-core CPUs, for a total of 24 cores.
  * GPU: NVIDIA Tesla V100 GPU with 32GB of GDDR5 main memory.
  * Swarm: open source Swarm architectural simulator.
  * HammerBlade: use detailed, cycle-accurate RTL simulation to model the RISC-V cores, network on chip, and LLC. DRAMSim3 for HBM2 memory, simulator environment using SystemVerilog DPI.
* Algorithm: PageRank, BFS, SSSP with delta-stepping, connected components  (CC) and betweenness centrality (BC).
* Graphs: 10 graphs, Orkut (OK), Twitter (TW), LiveJournal (LJ), SinaWeibo (SW), Hollywood (HW), Pokec (PK), and Indochina (IC) have power-law degree distributions, while RoadUSA (RU), RoadNetCA (RN), and RoadCentral (RC) have bounded degree distribution.

### Conclusions
* Hard to find out the refernce value due to the lack of background on graph computing. 
* This team(including MIT Commit) works deeply in this area from both software and hardware perspective. The Swarm arhictecture is also proposed by some of the authors.  This paper extended the GraphIt, the graph computing compiler, for various hardwares. This may be influenced by the preliminary advances made in deep learning compiler community.
* For several novel architectures, one can use the simulation tools provided to conduct experiements if there is a need to demonstrate the performance or efficiency of their work on brand new architecture.
* Whether the optmization methods in this paper is useful for **GNN** should be determined one by one based on the hardware. The method proposed in GNNAdvisor may be a better place to start up.
* Although the GraphIt is open source, the codes for this paper is not updated currently.
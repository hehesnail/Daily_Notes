## Mind Mappings: Enabling Efficient Algorithm-Accelerator Mapping Space Search (ASPLOS 2021)
### Contributions
1. first to enable target domain-independent mapping space search for programmable accelerator
2. first to formulate mapping space search as a first-order optimization problem, enabling an efficient gradient-based search;
3. experiments validate the efficiency of the proposed method, also open-sourced the codebase.

### Methods 
<div align="center">
<img src="https://github.com/hehesnail/Boring_code/blob/main/imgs/mind_mappings.png" width="70%" height="70%" /> 
</div>

### Benchmark & metric
1. Tasks: Conv layer & MTTKRP with different shape/kernel size settings
2. Baseline methods: Genetic algorithm, Simulated annealing, RL
3. Comparision metric: EDP, energy-delay product.
4. Experiment method: ios-iteration search quality / ios-time search quality
5. key validate: generality, quality, optimiality, time-per-step.

### What can we borrow?
0. source code: https://github.com/kartik-hegde/mindMappings
1. formulate our schedule problem as mapping space search problem, i.e., map the deep learning model to the hetergenous system. 
2. simanneal, DEAP package for search. 


## A learned performance model for TPU (MLSys 2021)
1. Four aspects of a well performance model:
   * general to handle non-trivial tensor programs
   * generalize across different domain applications.
   * not rely on well-crafted features
   * retargetable to different optimization tasks
2. Proposed approach:
   * data flow graphs -> GNN for capture local features -> lstm/transformer for global infos of a graph -> encode op properties in vector feature -> retargetable to various tasks.
3. Proposed model:
   * Inputs: node features, kernel features, adj matrix from XLA representations.
   * Model: GNN for node embedding, utilize the GraphSAGE as gnn model.
   * Model: LSTM & Transformer to extract global structure among all node embeddings.
   * Objectives: MSE for op-fusion task, pairwise rank loss for tiling size selection task.
<div align="center">
<img src="https://github.com/hehesnail/Boring_code/blob/main/imgs/tpu_cost_model.png" width="70%" height="70%" /> 
</div>

4. Datasets:
   * Based on 104 XLA program, generate datasets for tile size selection & fusion task datasets with different configs, 25 million samples and 208 million samples respectively in final.
   * Insight: may need a large dataset for GNN model to converge.


## A Deep Learning Based Cost Model For Automatic Code Optimization (MLSys 2021)
**TODO**
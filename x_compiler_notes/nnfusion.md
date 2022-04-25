## NNFusion(Rammer osdi20) Notes

This markdown contain several personal thinking about NNFusion framework.

The main goal is to answer the following questions: 
* The overall workflow and architecture of the nnfusion.
* How rOperator/rTask/rProgram which claimed in paper for intra-inter schedule implemented including data structure and algorithm?
* How the clamed vEU abstraction formulated?
* The policy for scheduling rOperators to vEUs.
* Finally, what can we borrow for our design ?

### KQ1: Overall workflow of nnfusion

Omit

### KQ2: rOperator/rTask/rProgram implemented mechanism

First, we need to locate the related classes in the source code, from the github osdi20-artifact branch, we obtain these infos.

| Name in Paper   | Name in Code |  Source File |
| -----------     | -----------  |  ----------- |                
| Rammer          | NNFusion     |  src/nnfusion/engine/pass/graph/blockfusion/            |
| RammerBase      | NNFusion (no BlockFusion Opt)  | -
| rOperator       | BlockCudaEmitter | src/nnfusion/core/kernels/cuda_gpu/cuda_emitter.hpp |
| rProgram        | BlockExecutorProgram | src/nnfusion/engine/pass/graph/blockfusion/common.hpp |
| rTask           | BlockExecutorInstruction | src/nnfusion/engine/pass/graph/blockfusion/common.hpp |
| vDevice         | BlockParallelDevice | src/nnfusion/engine/pass/graph/blockfusion/block_parallel_device.hpp |

The Engine add **BlockFusion** pass to the whole passes. Engine -> BlockFusion -> BlockFuseOptimizer.Optimize().

**Optimize**: ---> WaveFront Schedule Policy in the paper, Optimize <--->  Schedule(G, D)

In the optimize process, the corresponding map is shown below:

ExtractFusionGroups + SplitGroups ---> WaveFront, i.e., get the wavefront of DFG.

MergeGroups + virtual device schedule_kernel ---> ScheduleWave and exec time comparision

```c++
class BlockFuseOptimizer
{
    // key data members
    bool m_db_ready;
    std::shared_ptr<Graph> m_graph;
    std::vector<std::shared_ptr<TaggedNode>> m_nodes;
    std::shared_ptr<cache::KernelCacheManager> m_kernel_db;
};
```
* ExtractFusionGroups:
    ```c++
    // defs of fusion group
    struct FusionGroup
    {
        // key data members
        size_t id;
        bool merge;
        std::vector<size_t> nodes;
        std::vector<size_t> sub_group;
        std::vector<float> duration;
        std::vector<std::shared_ptr<KernelEmitter>> block_kernels;
    };
    // defs of tagnode
    struct TaggedNode
    {
        // key data members
        std::shared_ptr<GNode> node;
        size_t group_id;
        size_t ready_inputs;
        bool visited;
        std::pair<int, int> BEs;
    };
    ```
    * add unvisited & ready inputs node id to queue, iter subgid add to sub_group 
    * while queue not empty, create FusionGroup with g_id -> cur_group
      * iterate sub_group, pop node id from queue, verify node and add to cur_group
      * for newly added node outputs, check the dst node inputs ready or not, add to queue and iter the subgid.
      * add subgid to next_sub_group, and add cur_group to FusionGroups vector.
    * **Key Function**: classify all nodes in Graph to different FusionGroups in a bfs way, during the visiting process, inputs of node should be ready.
    * ret: ```std::vector<std::shared_ptr<FusionGroup>>```

* SplitGroup
    * **Key Function**: if there exists a group size > MAX_GROUP_SIZE(128), split the group to multiple groups via subgroup data member. The main purpose may be to avoid group contain too many nodes. 

* MergeGroups(optional)
    * **Key Function**: for the groups which can be merged, profile the merged group and two groups to get the time, if merged group is fast, merge the group.

* FuseGroupOnGraph  
    * TODO, figure the how the virtual parallel device schedule results affect the further passes.
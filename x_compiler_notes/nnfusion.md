## NNFusion(Rammer) Notes

This markdown contain several personal thinking about NNFusion framework.

The main goal is to answer the following questions: 
* The overall workflow and architecture of the nnfusion.
* How rOperator/rTask/rProgram which claimed in paper for intra-inter schedule implemented including data structure and algorithm?
* How the clamed vEU abstraction implemented?
* The policy for scheduling rOperators to vEUs.
* Finally, what can we borrow for our design ?

### KQ1: Overall workflow of nnfusion

<div align="center">
<img src="https://github.com/hehesnail/Daily_Notes/blob/main/imgs/nnfusion.png" width="100%" height="100%" /> 
</div>

### KQ2&4: rOperator/rTask/rProgram implemented mechanism

First, we need to locate the related classes in the source code, from the github osdi20-artifact branch, we obtain these infos.

| Name in Paper   | Name in Code |  Source File |
| -----------     | -----------  |  ----------- |                
| Rammer          | NNFusion     |  src/nnfusion/engine/pass/graph/blockfusion/            |
| RammerBase      | NNFusion (no BlockFusion Opt)  | -
| rOperator       | BlockCudaEmitter | src/nnfusion/core/kernels/cuda_gpu/cuda_emitter.hpp |
| rProgram        | BlockExecutorProgram | src/nnfusion/engine/pass/graph/blockfusion/common.hpp |
| rTask           | BlockExecutorInstruction | src/nnfusion/engine/pass/graph/blockfusion/common.hpp |
| vDevice         | BlockParallelDevice | src/nnfusion/engine/pass/graph/blockfusion/block_parallel_device.hpp |

Key data structure defs:

```c++
// rTask, the rTask mainly maintain a be_id, like task_id in paper
class BlockExecutorInstruction
{
public:
    BlockExecutorInstruction(int _be_id)
        : be_id(_be_id)
    {
    }
    virtual ~BlockExecutorInstruction() = default;

public:
    int be_id; // bind to which block executor
};

// rOperator, not an actual operator, auto-gen kernel code via operator template codes
class BlockCudaEmitter : public CudaEmitter
{
public:
    BlockCudaEmitter(shared_ptr<KernelContext> ctx)
        : CudaEmitter(ctx)
        , num_local_thread_sync(0)
        , shared_memory_size(0)
        , is_emitting_block_kernel(false)
    {
    }
    // xxxxx
private:
    size_t num_local_thread_sync;
    size_t shared_memory_size;
    bool is_emitting_block_kernel;
    FunctionUnit_p m_block_function_unit;
};


using BEInstruction_p = std::shared_ptr<BlockExecutorInstruction>;
using BlockKernel_p = std::shared_ptr<BlockCudaEmitter>;

// rProgram, contain both rOperator and rTasks, rOperator contains many rTasks
class BlockExecutorProgram
{
public:
    size_t num_bes;
    std::vector<std::vector<BEInstruction_p>> block_executor_instructions; // be_id with related block instruction
    std::vector<BlockKernel_p> block_kernels; // (key, value): 
                                                // (kernel_id, BlockCudaEmitter)
};

using BEProgram_p = std::shared_ptr<BlockExecutorProgram>;
```

The Engine add **BlockFusion** pass to the whole passes. Engine -> BlockFusion -> BlockFuseOptimizer.Optimize().

**Optimize**: 

---> WaveFront Schedule Policy in the paper, Optimize <--->  Schedule(G, D)

In the optimize process, the corresponding map is shown below:

ExtractFusionGroups + SplitGroups ---> WaveFront, i.e., get the wavefront of DFG.

MergeGroups ---> still an exprimental option.

for wavefront operators, FuseGroupOnGraph ---> Inner schedule process.

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
    // defs of fusion group, a fusion group corresponds to a wavefront 
    struct FusionGroup
    {
        // key data members
        size_t id;
        bool merge;
        std::vector<size_t> nodes;
        std::vector<size_t> sub_group;
        std::vector<float> duration;    // commonly the duration is 10 in src, 
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

**Abstracition of BlockParallelDevice(vDevice)**
```c++
class BlockParallelDevice
{
public:
    using Pointer = shared_ptr<BlockParallelDevice>;
    BlockParallelDevice(size_t _num_bes,
                        BlockKernelSchedulePolicy _policy = BlockKernelSchedulePolicy::DEFAULT)
    {
        create_device(_num_bes, _policy);
    }
    BlockParallelDevice(size_t _num_bes, std::shared_ptr<BlockKernelScheduler> _scheduler)
    {
        create_device(_num_bes, _scheduler);
    }
    // xxxxxx
private:
    size_t num_bes;
    std::map<std::string, int> kernel_name_id_map;
    std::vector<std::vector<BEInstruction_p>> block_executors;
    std::vector<int> block_executor_steps;
    std::vector<BlockKernel_p> block_kernels;         // (key, value): (kernel_id, BlockCudaEmitter)
    std::vector<KernelMetric_p> block_kernel_metrics; // (key, value): (kernel_id, kernel_metric)
    std::shared_ptr<BlockKernelScheduler> scheduler;
};
```

* FuseGroupOnGraph  
    * Start from an wavefront group G, create the BlockParallelDevice to schedule these operators & rtasks.
        ```c++
        // codegen for the block fusion node, 1024 stands for the max number of block per kernel
        auto virtual_device_p =
            std::make_shared<BlockParallelDevice>(DEFAULT_BE, BlockKernelSchedulePolicy::RANGE);
        ```
    * Key data structure, the BlockKernelScheduler & RangeBlockKernelScheduler.
        ```c++
        class BlockKernelScheduler{
        public:
            using Pointer = shared_ptr<BlockKernelScheduler>;
            BlockKernelScheduler(int _num_bes){
                num_bes = _num_bes;
                schedule_kernel_log.clear();
            }
            // xxxxxx
        protected:
            int num_bes;
            std::map<int, BlockKernelScheduleRecord> schedule_kernel_log;
        };
        ```
        ```c++
        class RangeBlockKernelScheduler : public BlockKernelScheduler {
        public:
            RangeBlockKernelScheduler(int _num_bes)
                : BlockKernelScheduler(_num_bes){
                be_lane.resize(num_bes);
            }
            // xxxxx
        private:
            std::vector<std::vector<int>> be_lane; // -1 indicates sync, kernel_id indicates kernel running
        };
        ```
        The be_lane here is used to find ready bes & time_start for each scheduling kernel. 
        
        The ready bes are be_ids to match the kernel required bes.
        
        Time_start indicates the smallest time this kernel should wait to start, time_start - be_lane\[be\_id\].size() times -1 will be added in be_lane for certain be_id. 

        Also, when assign one block to a be, add the kernel_id in be_lane[be_id] duration times. The sync/wait point add -1s. Thus, the maximum number of each be_lane[i] is the total estimated time for this group.
    * If the estimated time < normal execution time, replace the original gnode kernel emitter to this BlockFusionCUDACodeGen, and replace group nodes with dummy node.

### KQ3: How the clamed vEU/vDevice abstraction implemented?

The vEU/vDevice abstraction is platform related. For GPU platform, To bypass the
built-in scheduler, RAMMER leverages a persistent thread-block (PTB) to implement a vEU in a vDevice. 

PTB is a thread-block containing a group of continuously running threads, where RAMMER is able to “pin” the PTB to the desired SM. RAMMER dispatches each vEU (implemented by a PTB) to a desired SM through the GPU scheduler.

Check "A study of persistent threads style gpu programming for gpgpu work-
loads."

Generally speaking, there must be a way to mapping vEU to either software-implemented or hardware supported processing unit which can be exposed to user side. In this way, we can assign smaller splited kernels(rTasks) to vEU for parallel execution.

### KQ5: what can we borrow for our design? 
* Rammer核心解决的问题为当device可提供较高并行能力时，而受限于单个kernel存在无法fully utilize硬件能力的情形时，以kernel为粒度的调度必将造成硬件资源的浪费，同时kernel dispatch存在动态开销。
* 为解决此问题，将kernel break为多个可并行的rTasks，这也是由cuda kernel中多个block可并行直观演化过来。但是gpu并未提供可直接调度block至不同SM上的接口，因此需要抽象出一层virtual device即virtual execution unit用做rTasks statitc schedule。同时，也需要提供一直机制完成vEU和实际硬件的映射。在GPU下，vEU通过PTB实现可分配算子的单个block，并调度至SM上运行。同时rTaskId可执行对应block的运算。
* 其本质可以看作是提供一种机制，即kernel转换为多个子kernel的方法，多个子kernel向虚拟并行硬件单元匹配，同时因virtual parallel device粒度为a group of operators, 因此可实现intra-inter operator schedule。
* Questions: 
  * kernel -> parallel tiling kernels (rTasks) -> rewrite to multiple ops indexed with id; 
  * 引入virtual device 及 virtual EU抽象。vEU和硬件如何适配，CPU场景下收益不大;
  * sync源语如何设计，如何处理依赖？
  * 异构资源下，引入多个vDevices？
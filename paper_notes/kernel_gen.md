## AKG: Automatic Kernel Generation for Neural Processing Units using Polyhedral Transformations (PLDI 2021)
### Overall Architecuture
<div align="center">
<img src="https://github.com/hehesnail/Boring_code/blob/main/imgs/akg_arch.png" width="70%" height="70%" /> 
</div>
AKG 整体框架如上图所示，从中可以看到其为TE -> Poly IR -> TIR的变换流程，本小结主要偏重于AKG针对内存结构支持所做的工作，其他点参照论文。

### 如何支持复杂的内存结构?
Pass Analysis + Attribute hints + Codegen Support.

For akg, 在poly schedule tree变换阶段, 此时schedule tree创建时直接从TIR构建, 对应Flatten Pass前，通过对IR读写进行分析, 标记每个Stmt的类型(cube conv, cube gemm, vector); 后完成fusion & tiling 变换后 + 已标记信息, 通过MemoryManager Pass完成auto storage management; GenHalide生成TIR供TVM Codegen生成代码, 以NPUIslEmitter为例, 其在generate halide ir时会对每个读写操作得到对应AttrStmt进行标记, 后续Pass中对此AttrStmt不进行操作, 应该是直接在Codegen层面完成解析并支持.

### 多面体模型的必要性
非必要，Ref to paper，使用多面体模型的目的在于:
* 更多的循环变换方式, suitable for npu;
* 可在poly ir上实现自动化内存管理；

从个人观感来看，其主要针对循环(conv, gemm)等操作变换至特定的结构(fusion/tiling)等并使其对应至NPU上CUBE/Vector等的输入输入内存。这些变换TVM schedule primitive + TVM标记也可实现，如文中baseline为AutoTVM(manual schedule template + params search)，而AutoTVM的方式弊端在仍需提供schedule template，AKG的变换则是自动化的。此工作应同Ansor并行，孰优孰劣尚不清楚，不过从Ansor的代码来看，其对于NPU内存结构的支持度不够，需进行额外支持。

## 机会点
**总体评估: 工程量大，相关工作少，亮点较少.**

AKG属于对TVM的深度修改，涉及Poly IR以及TIR两个层级IR的优化和变换，也需对NPU结构有较深入的理解。其他貌似较多论文主要针对GPU Kernel生成，也意味着较难做出必要Fancy的工作。从论文的角度看，仅支持一个全新的NPU平台，工程量大但创新度是不够的，需要更多的是在方法层面上的创新，AKG中对于支持Ascend 910的着墨并不多，全文均集中在Poly变换以及claim 此IR和IR变换对于NPU结构有多好多好之类。卖点是其poly可高效利用NPU结构 + 自动化(tiling & memory) + First NPU Platorm。

**Co-optimization**

平台层面支持全新平台作为论文可能创新度不足，因此考虑更加异构的环境，包括host/device的自动调度，这点也更类似于图切分和Kernel生成一起做，AKG上主要针对单算子搞的，且Ascend有一定可编程能力支持部分逻辑算子，若在更加受限的硬件上，考虑到Graph-level的分配和Kernel自动生成可能为一个方向。

## AKG Source Code Notes
### AKG vs TVM (version 0.6) Changes
#### header files
* namespace tvm -> air
* runtime/device_api.h ---> add device kDLCce
* akg_expr_operator.h -> add new file, add asin, acos, asinh, acosh
* codegen.h ---> overload Build function with global_external_call_names(Array<NodeRef\>)
* expr.h ---> Expr constructors from double/int64_t/uint64_t value
* expr_operator.h ---> add %, isinf, isfinite, infinity, supported in tvm 0.7
* ir.h  ---> add builtin intrinsics (reinterpret_cast_op & ldg)
        ---> Swizzle loop type for ForType
    ---> AttrStmt.attr_key:
      * pragma_emit_insn(vector operation scope),
      * reduce_update(reduce_update),
      * promote_vectorization(modify ast to support vectorization),
      * bind_thread_x(adapt to one-dim mapping),
      * wmma_scope(for tensorcore interface),
      * atomic_tot(mark tensor-of-tensor)
        ---> add tir builtin intrinsics, akg_fragment_elem(for tensor core fragment operator fusion), tvm_cce_string_print
* ir_pass.h ---> add new func StmtUseVar
            ---> InjectDoubleBuffer, add use_transfer_buffer flag
            ---> add ThreadSyncStmt, LowerThreadAllreduceStmt, InferFragmentStmt
* lowered_func.h ---> add args_real, workspace(for refering to extra global mem needed)
* tensor.h ---> slice support /, %
* schedule.h ---> Stage add buffer_align & buffer_tile schedule primitive
             ---> StageNode add realize_aligns, realize_bound
* operation.h ---> BaseComputeNode add ComputeRealizeBounds method
              ---> HybridOpNode add (Tensor->Buffer) input_buffers_, output_buffers_, (Tensor->Region) input_regions_, output_regions_
              ---> compute, fcompute func support 5 vars
#### python side
* runtime_ctypes.py ---> add type code for cce
* autotvm ---> sa_model_optimizer.py/xgboost_cost_mode.py/xgboost_tuner.py support AKG tuning model.
* hybrid
* rpc/client.py ---> add def cce for RPCSession
* intrin.py ---> add isinf, isfinite ops
* irbuilder.py ---> add for_range_n, extern_call, load, store
* ndarray.py ---> add cce support for runtime ndarray test
* schedule.py ---> buffer_align, buffer_tile, emit_insn
* target.py ---> add cce target

#### cxx side
* api/ ---> add related ops/pass registeration
* arith/ ---> some related modifications
* runtime/ ---> pack_args.h, add PackFuncVoidAddrCCE, PackFuncVoidAddr_
           ---> module.cc, add new cond for cce target
           cce/ ---> the cce runtime, cce_common.h, cce_device_api.cc, cce_module.cc, cce_module.h
                ---> provide utility functions for alloc/free memory, kernel launch, file/binary save/load.
* codegen/ ---> build_module.cc, add cce target
           ---> codegen.cc, Build function contains **codegen.build_cce, can't find src, binary in prebuild folder.**
           ---> codegen_c.h/codegen_c.cc,
           ---> akg_codegen_cuda.cc, akg_cuda_intrin_rule, codegen_cuda.cc
* op/ ---> compute_op.cc, add ComputeRealizeBounds, annote reduce_axis with reduce_update  for poly pass
      ---> hybrid_op.cc, input/output buffers
* schedule/ ---> bound.cc AlignRootBound & schedule_lang.cc (buffer_algin, buffer_tile)
* pass/ ---> support for TensorCore: unroll_loop, inject_double_buffer, storage_rewrite, tensor_core.cc
        ---> support for CCE: simple_passes.cc(SubstituteCCE), loop_partition(CCE)

### python/ops
#### lang/cce
* most of ops are defined in tvm.compute
* common.py ---> call_pure_intrin: fargmax, fargmin, mad, dropout, iou, nms, topk_sort, proposal_sort, fnot, four2five_nchw, load_im2col_c1_buf
                                   sin, cos, sinh, cosh, divide_var, vmadd, vmla
* common.py utilize the Ascend provided functions.
#### ops
* array, use hybrid for logical ops
* math, add.py ---> note for this op, in/out lies in **local.UB**
* img2col, call load_im2col_c1_buf in lang.cce, l1 -> ub; maxpool op as eg
* conv/matmul --->  mmad intrinsic

### AKG Lower & Build
py side: build_module -> lower -> build(codegen)

#### DSA Infos
def in dsa_utils.h / dsa_utils.cc, specify the memory heierachy, MemType, DataFlowAttrs

#### Lower(only lower_cuda.cc, cce not provided)
* CudaLowerBegin
  * schedule::AutoInline & AutoFuse, normalize -> InferBound -> ScheduleOps, TensorAccessRewrite
  * ReplaceSeparator(change all '.' to '_'), RewriteMultiValueFunc(rename func name, here is op name for tvm 0.6?)
  * RenameRealize, for realize_scope with "local.UB", change related op name with "local.UB" too.
  * ElementwiseFlatten
  * RewriteTensorIndex
* CudaLowerStageTuning
  * GenTuningSpace
    * **GenIsl**: generate isl schedule from Halide
      * MakeScheduleTree -> ScopMakeScheduleTree -> create the  isl::schedule
      * **CreateDataFlowInfo**
         ---> call **DMADataFlow** to analyze Stmt, STMT_OP_TPYE, TENSOR_DATAFLOW_TYPE
              SetTensorNameFlows &  SetTensorMemFlows
              ---> CreateStmtDataFlow, stmt three types-> cube: conv, cube: gemm, vector
              ---> UpdateFlowInfo
    * **Transform**: transform isl schedule tree
      1. SchedulePassMgr
      2. DsaMgrStrategy or GPUMgrStrategy
        * DsaMgrStrategy Registered Passes
           * InitSchedule, ConstrainSchedule, GroupStatements (User Config)
           * ComputeSchedule, ReorderInvariantSetSchedule, SinkC0 (User Config)
           * SinkLastAxis (User Config), KeepOuterBandOrder (User Config)
           * UnGroupStatements, SplitOuterBand (User Config)
           * ComputeInnerBandDependency, ComputeTransferCopyin (Pattern Match)
           * **TileOuterBand**
           * ReorderInvariantSetSchedule, ResetCoincidenceOfReduce
           * SetAllCoincidence (User Config)
           * **Reschedule**
           * ReorderInnerBand, ChangeMarkNodePosition, LabelRealizeOutPosition
           * InsertNodeForAllocC(Pattern Match)
           * **MemoryManager**: for auto storage management
           * TransferStmt(Pattern Match)
           * ReorderMarkNodes, MarkFuseOp, MarkOuterMost(User Config)
    * **GenHalide**: generate halide ir from isl schedule -> isl::ast_build -> isl::ast_node
                     -> NPUIslEmitter or GPUIslEmitter -> Stmt
        
        * **NPUIslEmitter**: all memory related tags defs in dsl_utils.h/cc converted to tvm tir::AttrStmt
    * GenerateTuningSpace or GenerateTilingSpace
    * If NPU, DsaHalideOptimizer
* CudaLowerPoly
  * AutoPoly
Remaining TVM Lower Passes
* CudaLowerBeforeFlattern
* CudaLowerFlattern
* CudaLowerBeforeRewrite
* CudaLowerRewrite
* CudaLowerBeforeLowerFunc
* CudaLowerDone
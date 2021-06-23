# TVM Notes

### *2020.11.17*
* use *parallel*, *vectorize* to create schedule which accelerate the execution time of operator.
* Use *split* for splitting the for loop, reorder can change the axis orders, quite like Halide.
* Identify the the bottleneck is memory bandwidth or computation. vector_add and broadcast_add are all memory-bound element-wise calculation.
* A good schedule needs to consider multiple performance-related factors together.

### *2020.11.18*
* for matrix multiplication, *reorder* the sum axis to avoid column based memory access
* choose the right tiling size to improve the cache effcience, *tile* primitive tile blocks, *fuse* can fuse two axes into one to be further parallelized.
* The non-continuous write issue is severer than the non-continuous read, so we can first write the results to a local buffer for each sub-matrix computation, and then write them back to the final matrix C.

### *2020.11.19*
* 可按照优化矩阵乘法的思路，对卷积操作中，宽和高的维度上进行*tile*操作，来提高cache利用率，但随着channel size的增大，性能会下降
* 仅对宽高维度做*tile*提升有限，采用相同思想对于channel维度也进行*tile*优化，需改变data layout

### *2020.11.24*
* depthwise conv和普通conv的优化思路相同，区别在于depthwise conv仅需tile单个维度的channel，因其输入输出channel相等同
* pooling这种 memory-bound 的算子，optimize的方式有限，可使用te.schedule.AutoInlineInjective(sch) 避免重复计算，类似于Halide的中inline的计算方式
  
### *2020.11.25*
* batch normalization 和 pooling的优化方法差不多，也是用AutoInlineInjective使计算全部inline，外层两axis做parallel，而内层使用vectorize优化
* *TOPI* (TVM Opeartor Invertory) provides numpy-style generic operations and schedules with higher abstractions than TVM. *TOPI* also provides higher level scheduling recipes depending on a given context to generate optimized kernel codes. *TOPI* also provides common neural nets operations such as _softmax_ with optimized schedule. 
* *TOPI* provide generic functions, different backends have different implementations to optimize the performance, thus should utilize the right backend implementation and schedule.

### *2020.11.26*
* Schedule primitives: 
    * ***split***: split the specified axis into two axises.
    * ***tile***: help to execute the computation tile by tile over two axises.
    * ***fuse***: fuse two consecutive axises of one computation.
    * ***reorder***: reorder the axises in the specified order.
    * ***bind***: bind a specified axis with a thread axis, often used in gpu programming.
    * ***compute_at***: A schedule may consist of multiple operators, tvm by default will compute at root which may result in redundant computation. It can move the computation of one operator to the axis of computation of another computation.
    * ***compute_inline***: mark one stage as inline, the body of computation will be expanded and inserted at the address where the tensor is required.
    * ***compute_root***: move the computation of one stage to the root.
    * **The schedule primitives are quite like Halide**.
* TVM support basic arithmetic operations, these funcsions are target system dependent and may have different names of different target platforms. *Direct way*: call target specific function via extern function call. *TVM instrinsic*: unified intrinsic call, like te.exp(). User can customize the instrinsic rules during runtime, like tvm.target.register_intrin_rule("cuda", "exp", my_cuda_math_rule, override=True). Add own instrinsic, two funcs need to be noticed: 
    * tvm.ir.register_op_attr; 
    * tvm.target.register_intrin_rule. 

* *scan* operator to describle the symbolic loop. *s_state*: the placeholder describles the transition state of the scan. *s_init*: how to initialize the first k timesteps. *s_update*: how to update the value at timestep t. The *scan* takes in state placeholder, initial value and update description. Multi-stage scan cell and multiple states.

### *2020.11.28*
* ***tvm.te.hybrid***: Hybrid programming apis of tvm python package, maps a subset python to HalideIR. So far, it is a text format dedicated to HalideIR phase0. *Funcs:* build, decorate, script, source_to_op. *tvm.te.hybrid* to indicate a function is a hybrid function.
    * Pass tvm data structures: Tensor, Var, Expr .*Imm, tvm.container.Array
    * Tuning: loop annotations (unroll, parallel, vectorize, bind), loop manipulation (split, fuse, reorder)
    * Loops: use range (aka serial,unroll,parallel and vectorize), const_range
    * Variables: All the mutable variables will be lowered to an array with size 1. It regards the first store of a variable as its declaration. Currently, the type of variable should be either float32 or int32.
    * Attributes: only shape and dtype attribute are supported, only constant-indexed access is supported.
    * Conditional statement and expression: if cond1 and cond2 and cond3: ... else: ...; No True or False keyword supported.
    * Math intrinsics: log, exp, sigmoid, tanh, power, popcount.
    * Array allocation: allocation(shape, type, share/local) to declare an array buffer. Under construction.
    * Thread bind and Assert statement.
* External Tensor function: use *te.extern* to add an extern array function call. In the extern call, we declare the shape of output tensors. In the second argument we provide the list of inputs.

### *2020.11.29*
* Auto-tuner: 
    * 1). Define the search space: @autotvm.template, get config object: cfg = autotvm.get_config(), cfg.define_split, cfg.define_reorder etc. then apply the configEntity.
    * 2). Search through the space: RandomTuner, GridSearchTuner, GATuner, XGBTuner. *First*, create a tunning task. *Second*, define how to measure (autotvm.measure_option) the generated code and pick a tuner, build and run two steps. *Finally*, apply history best from teh cache file and check its correctness by autotvm.apply_history_best.
* Auto-scheduler:
    * Template-based autotvm relies on manual templates to define the search space.auto-scheduler does not require any templates. The auto-scheduler can automatically generate a large search space and find a good schedule in the space.
    * Define the computation and decorate it with @auto_scheduler.register_workload, the auto-scheduler can get the whole computational graph. Create the search task by *tvm.auto_scheduler.create_task* and set parameters for auto-scheduler by *auto_scheduler.TuningOptions*. Run the serach via *auto_scheduler.auto_schedule*. After schedule, can laod the bset schedule from record file by *auto_scheduler.load_best*.
### *2020.11.30*
* ***tvm runtime system***: requirements: deployment, debug, link, prototype, expose, experiment.
    * PackedFunc: type-erased function, when call a PackedFunc, it packs the input arguments to TVMArgs on stack, and gets the result back via TVMRetValue. One can register PackedFunc in C++ and calls from python. Limitation of TVMArgs and TVMRetValue: int, float, string, PackedFunc itself, Module for compiled modules, DLTensor* for tensor object exchange, TVM Object to represent any object in IR. PakcedFunc is a universal glue in TVM.
    * Module: the compiled object. User can get the compiled function from Module as PackedFunc.
    * Object: All the language object in the compiler stack is a subclass of Object. Each object contains a string type_key that uniquely identifies the type of object. *ObjectRef* can be viewd as shared_ptr to Object container. Each Object subclass will override this to visit its members.
    * Check packed_func.h for C++ API and c_runtime_api.cc for C API and how to provide callback.

### *2020.11.31*
* ***tir***: 看了一手tir的python api定义，理清了几点，1). te (tensor expression)中相当多operator的定义和实现是通过import tir中相应的函数，因此可直接通过tir.xxx调用，二者等价；2). 通过简单的溯源，tir 中 class 和 对用 func 的最终都会映射到同一相应 C++ class创建 node；3). tir python中相应的函数使用_ffi_api可以对应到 C++对应的class，比如 tir op.py文件中 def abs(x) 通过调用 _ffi_api.abs(x)得到对应的 expression(PrimExpr). 还需理清整个数据流程.

### *2020.12.23-2021.1.4*
* ***TVM source code reading***:
    * The tvm runtime system: PackedFunc, Registry, Manager in C++ side and the corresponding class in python _ffi. Also, notice the macros which are prefixed with TVM_REGISTER_XXX
    * Schedule and stage: how to create_schedule, and then obtain each stage, for each stage the corrpresonding schedule pass like tile, split, fuse and so on.
    * Lower process: how the schedule is lowerd, mainly can refer to the build_module.py and the corresponding cxx passes. Various optimization happens in this stage. Figure out what happened in each pass.
    * IRVistor and IRMutator, use these to access the AST and modify AST. Many passes are inherited from the Vistor and the Mutator to finish the task.
    * Codegen: when build a module, how to obtain the LLVM IR and then the executatble binary file. Refer to the codegen.cc, codegen_llvm.cc, llvm_module.cc etc. Basiclly, inherited from the Vistor and Mutator, create the LLVM context, module, then utilize the LLVM IRBuilder to generate the function call and the BasicBlock and Instructions in function. After generate the LLVM IR, can utilize the LLVM PassManager to enable the LLVM optimizations. For the executable file, call the SaveToFile method to transform the LLVM IR.

### *2021.1.10*
* ***Reread the docs for tvm***
    * runtime::Object is one of the primary data structures in TVM runtime besides the runtime::PackedFunc. It is a reference-counted base class with a type index to support runtime type checking and downcasting. 
    * The components in tvm/ir are shared by tvm/relay and tvm/tir, notable ones include IRModule, Type, PassContext and Pass Op.
    * tvm/tir : TIR contains the definition of the low-level program representations. We use tir::PrimFunc to represent functions that can be transformed by TIR passes. Besides the IR data structures, the tir module also defines a set of builtin intrinsics and their attributes via the common Op registry, as well as transformation passes in tir/transform.
    * tvm/te: te name te stands for “tensor expression”. This is a domain-specific language module that allows us to construct tir::PrimFunc variants quickly by writing tensor expressions. Importantly, a tensor expression itself is not a self-contained function that can be stored into IRModule. Instead, it is a fragment of IR that we can stitch together to build an IRModule. te/schedule provides a collection of scheduling primitives to control the function being generated. In the future, we might bring some of these scheduling components to the a tir::PrimFunc itself.
    * Analyze the source code of IR, which offers the unified interface for the Relay and TIR.

### *2021.1.11*
* ***analyze the TIR structure, especially TIR/IR/expr.h, buffer.h, var.h***
    * detailed info written in notebook, tobe summarized.

### *2021.1.18*
* ***The TIR structure***
    * 重新扫了下object, objectptr, objectref, 后续又看了下 tir/expr下的一些东西，对于Var产生兴趣，每个Var由其地址所唯一区分，之前被误导了淦，在Allocate, Let, For, LetStmt中是每个Var仅被bind一次，之前看成只在这几个中bind。。。
    * 结合llvm codegen看，这几个Node会维护一个 VarNode* -> llvm::Value* 的var_map表，对于Let, For, LetStmt的Var 基本是通过Make_value得到，Allocate则会调用到 llvm irbuilder 的 alloca 并convert成指针类型，其实这个过程就将抽象的Var 转为了具体的 数 或者 内存指针。
    * 在compute时候，会根据compute传入的参数进行解析，shape被解析为 vector<IterVar> axis 同时从中会获取到 类型为 vector<Var> 的args参数，而计算函数FCompute为函数模板对象，根据args调用，则会得到 类型为PrimExpr body，即内部的计算描述。  
    * 目前来看，tir这块理解最欠缺的是内存管理相关的IR，之后着重分析。

### *2021.1.19*
* ***TIR Buffer***
    * 看了下TIR中Buffer的管理，因利用IRBuilder将Tensor创建buffer_ptr时出错。其实因为Tensor会在lower的过程中最后转换为Buffer，因此这种行为相当于将lower过程中搞得提前到了Tensor赋值的过程中来，故报错，若非要解决这个问题只能根据Tensor的dtype, shape重新构建Buffer才行。
    * TIR Buffer在 C++端(src/tir/ir/buffer.cc, include/tir/buffer.h) 和 Python端(python/tvm/tir/buffer.py)行为基本一致，因Python Buffer class 直接调用C++ register的类构造函数 以及 成员函数，其中decl_buffer也是直接调用C++端的函数。BufferNode中基本三个member为 Var data -> 指向数据的指针，DataType dtype -> 数据类型 以及 Array<PrimExpr> shape -> buffer的shape大小。那关于Buffer的内存分配，看了下codegen_llvm.cc中，llvm的IRBuilder只有在Allocate Stmt的时候，才会调用 CreateAlloca 即分配内存，故Buffer的内存管理主要还是由 lower的过程当中，适时地插入Allocate Stmt时才完成。那么关于 Store / Load 基本是先获取到 对应Buffer，后利用 llvm IRBuilder 创建指针访问。
    * python端 ir_builder 使更容易的获取buffer元素 ---> BufferVar 类，类构建irbuilder, buffer_var(Var), content_type；通过__setitem__ (emit Store) 和 __getitem__ (emit Load)来创建IR，同时注意 buffer_ptr函数接收一个Buffer对象，将其 data 和 dtype提取并创建BufferVar对象，就可以A[i]这种方式获取Buffer中的元素，免去了很多麻烦。pointer函数根据类型先创建 Var，后创建 BufferVar绑定到刚创建的Var上。allocate函数会直接emit allocate stmt. 
  
### *2021.2.19*
* ***Storage Flatten Pass***
    * Ref [tvm tensor management notes](https://github.com/hehesnail/Boring_code/blob/main/tvm_pieces/tvm_storage.md)
### *2021.3.7*
* ***Some notes***
    * Writing a Customized Pass: https://tvm.apache.org/docs/tutorials/dev/low_level_custom_pass.html#sphx-glr-tutorials-dev-low-level-custom-pass-py 
    * 以前还真没注意怎么在 python 端写 tvm lower pass的，基本还是和 cxx side 差不多， python 端更加不灵活吧stmt_functor 中 post_order_visit 获取想 modify 的 IR， stmt_functor.ir_transform进行变换， stmt_functor.substitue替换，将 Pass可通过在 tvm.transform.PassContext 中添加
    * 从 compile models来看，首先是 relay.frontend.from_xxx， 后创建 graph executor， relay.build_module.create_executor那其实 第一步主要就是解析 不同前端框架 保存模型参数的一个过程，然后匹配成 relay_func 的格式；其中不同前端中的OP(conv, pool)之类的会被的等价转换为 relay 中 op，relay中 build_module过程中调用cxx端的 RelayBuildModule 中 build，后调用  BuildRelay， 其中会创建 graph_codegen 通过  _GraphRuntimeCodegen 创建  GraphRuntimeCodegenModule 调用其中 Codegen方法，在 visit 其中 CallNode时候，_CompileEngineLower，其中会根据 OpStrategy完成对不同OP实现的选择
### *2021.3.13*
* ***Tensor IR First Impression***
    * 粗略地看了下RFC(https://discuss.tvm.apache.org/t/rfc-tensorir-a-schedulable-ir-for-tvm/7872)上的讨论, TensorIR是TIR可优化的增强版, 主要提出 Block statement structure来warp IR, Block会作为最小的scheduling and tensorization的单元，其中包含 Iter Vars, Block读写region, allocated buffer 以及 body stmt. 从这点来看，是在已有的TIR的基础上新增了更加粗粒度的访问块，在Block unit中附加上schedule所需的信息，从而实现直接对于TIR的schedule；
    * 几点好处: 1). schedule直接作用在 TIR, 而不需要通过在schedule tree调度后拼接 stmt形成 TIR，TF/PyTorch/ONNX -> Relay -> TIR -> schedule -> TIR -> scheudle -> TIR -> C++/CUDA; 2). 更加灵活的描述，相对于TE; 3). 更好地支持 memory hierarchy 以及 execution hierarchy; 4). 对于 tensorization 支持更加灵活; 5). 可以在每次schedule对于IR进行验证
    * 目前来看, 关于Block 的 PR已经提了;
### *2021.3.15 & 16 & 17*
* ***Storage Rewrite Pass***
    * Ref [tvm tensor management notes](https://github.com/hehesnail/Boring_code/blob/main/tvm_pieces/tvm_storage.md)
### *2021.3.23 -> 28*
* ***Schedule Lang***  
    *  Schedule & Stage creation ---> Done
    *  ScheduleOps: main process to obtain body ---> Done
    *  ReadGraph/FeedGraph/AttachPath ---> Done
    *  Split/tile/reorder/fuse ---> Done
    *  Compute_at/compute_inline/compute_root ---> Done
    *  Unroll/parallel/vectorize/tensorize/pragma ---> lowered in tir passes. parallel->llvm codegen, unroll->unroll_loop pass, vectorize->vectorize_loop pass, tensorize->compute_op make_tensorize, pragma->compute_op MakeLoopNest, to AttrStmt ---> Done (tir passes not included)
    *  InjectInline ---> Done
    *  InjectAttach ---> Done
    *  Cache_read ---> Done
    *  Cache_write ---> (TODO)
    *  Message passing helpers ---> MakeBoundCheck, PassUpBoundCheck, PassDownBitMaskOr (Done), remaining range processing
    *  RebaseNonZeroMinLoop ---> Done
    *  InferBound ---> (TODO)
    *  LegalizeInvalidAttach ---> Done
    *  ComputeOp Body 
        * Main Process: **DetectComputeType** ---> **MakeComputeStmt**
        * **DetectComputeType**: 根据对应 stage 中 iter_var_attrs的属性，区分出 kTensorize(MakeTensorize), kCrossThreadReduction(MakeCrossThreadReduction), kNormal(MakeComputeStmt).
        * **MakeComputeStmt**: 构建ComputeStmt的主要过程: 
          * 1). Create ComputeLoopNest 
          * 2). MakeIfNest  
          * 3). 有 reduce_axis, MakeReduction，拼接 CommonNest 和 ReduceNest，并完成最终body拼接  
          * 4). 无 reduce_axis, Merge provide and main_nest 
        * **ComputeLoopNest**: 
          * 1). Call MakeLoopNest创建主循环，此函数的主要作用是根据leaf_iter_vars 构建Loop中的 For Nest, 其中会根据 iter_var 的 IterVarAttr的不同，采取不同的行为，并添加 AttrStmt；在每个 IterVar 创建完成之后会利用 loop_scope的 AttrStmt进行标记，会在compute_at中用到；
          * 2). Call MakeBoundCheck 检查 main_nest当中是否需要加入额外的 If判断，其中会调用PassUpBoundCheck来对Split, Fuse, Rebase等 RelationNode进行处理；
          * 3). 若存在 reduce_axis, 需要在正确的地方插入对应 Reduce的初始化语句(init_nest)，具体调用 PassDownBitMaskOr 来判断进行过 Split，Fuse，Rebase等操作的IterVar是否为reduce_axis；之后在生成 init_nest的时候，会找到reduce_axis出现的位置，以此开始，并且会跳过所有的非 reduce_axis的iter var，而 init_nest 和 init_predicated的生成调用的也同样是 MakeLoopNest 和 MakeBoundCheck，不过此次会生成新的 iter_var, 以".init"为后缀；
          * 4). 此时 num_common_loop 在是否有reduce_axis的情况下不同，即 begin_loop & leaf_iter_vars.size();
        * **MakeIfNest**: 根据 predicates 生成 IfThenElse Node；
        * **If reduce_axis**: 
          * 1). 通过 MakeReduction 获取到内部的 body (actual computation)，init, provide，并将 init body 拼接在 init_nest的 body中；
          * 2). 根据 num_common_loop将 main_nest分为 common, reduce，并在reduce body拼接为provide，后和 init相拼后合入 common中； common->init->reduce->provide;
        * **If no reduce axis**: 这个相对简单，MakeProvide (ProducerStore)获取 provide body, 和 main_nest一拼就完事；
        * 最后将 ComputeStmt中的 IterVar 替换为 PrimExpr.

### *2021.3.29*
* ***Unroll_loop & Vectorize_loop***
    * unroll_loop: Done 
    * vectorize_loop: Done 

### *2021.4.3 & 4.4*
* ***InjectDoubleBuffer***
    * 首先搞清楚，啥时候才能 double_buffer，看 Pass里面，double_buffer_scope必须在一个 for 循环中；这时候看 scheduleops 以及 schedule_postproc_to_primfunc的时候，看不出什么端倪，因为正常 MakePipline的时候；即使 set_stage 为 double_buffer_scope，此 AttrStmt必不在循环中，之后看例子，是因为 double_buffer这玩意，必须配合到 compute_at一起用，attach 到某个循环轴上，也会在 loop中。另一种情形就是使用 IRBuilder 在 for 中 allocate 并 set 对应 buffer_var 为 double_buffer_scope, emit成 extern_op
    * Finish, not summarized.

### *2021.4.19*
* ***Auto-schedule Get Started***
    * Typical Flow: 
         * Describle computation rule(**register_workload**)
         * Create theSearchTask with comp_func, args, targ (**SearchTask**), also can obtain compute_dag for the created_graph
         * Set up tune params via(**TuneOptions**), and use **LocalRPCMeasureContext** as gpu runner, **LocalRunner** for x86 runner.
         * Start auto-tuning just by **tune**, and **apply_best** ret the searched schedule and args to creating tvm module.
         * Build the tvm module and evaluate the searched result.
         * Addtionally, one can also resume the past search, by creating cost model(i.e., **XGBModel**) and call **update_from_file**. create the search_policy via **SketchPolicy** and sepcify the init_search_callbacks.The remaining things are just the same as above. 
         * For auto-schedule the network(often be converted to relay module), call **extract_tasks** to parse the search-ops based tracing mode. Use **TaskScheduler** to run multiple search tasks. The remining thing is just to apply best log and build the relay module.
         * Brief impression, the core things are: search_task.py, search_policy,py, cost_model/cost_model.py & xgb_model.py, measure.py.
         * Next things: follow the workflow, read original paper and analyze the source codes from python to cxx side.

### *2021.4 & 5*
* Auto Schedule Summary Notes: [auto_schedule notes](https://github.com/hehesnail/Boring_code/blob/main/tvm_pieces/auto_schedule_notes.md)
* TVM Tensor Management Notes: [tvm tensor management notes](https://github.com/hehesnail/Boring_code/blob/main/tvm_pieces/tvm_storage.md)

### *2021.6*
* Relay paper notes -> TODO
* Workflow
* **function.py & build_module.py & build_module.cc**
  * 一般来说，先创建 Relay **Function**(relay.ir.Function)，通过传入 params(tvm.relay.Var) 以及 body(tvm.relay.Expr)创建; 后将该 func 添加进入**IRMoulde**中; 可通过 BindParamsByName 绑定 Function中对应 Var 为外部参数（如权重），针对其中 VarNode 直接替换为对应 Const即可;
    * Relay FunctionNode ---> BaseFuncNode ---> RealyExprNode; 其中RelayExprNode的描述，BaseNode of all-non primitive expressions, 而依照Paper中说法，DL type system一个特殊的点在于要将 function call等操作看作 primitive 同等级的，RelayExprNode支持 tensor types, functions, ADT作为第一等公民; 
    * PrimExprNode 中 DataType 同 RelayExprNode 中 tvm::Type的区别; PrimExpr的数据类型匹配是通过runtime时候进行check，这对于 POD Value是足够的; 对于 Tensor Type等类型，需要借助其文中提到的type system进行 type inference; 因此，BaseFunc 继承 RelayExpr， PrimFunc 以及 RelayFunc继承自BaseFunc;
  * call build, helper func builds relay func to graph exectutor; 
    * 主要参数为 IRModule，params(str->NDArray)，以及target信息; 
    * 对于直接传入Func而言， 先绑定 params 后 create IRModule;
    * 后续创建 BuildModule 对象， call build method ---> ret graph_json/runtime module/ graph params; 后根据 executor， 返回对应 AOTExecutorFactoryModule or GraphExecutorFactoryModule.
  * python side BuildModule __init\_\_ call _BuildModule defined at cxx side, 创建 RelayBuildModule ret. 其余函数有通过 Module __get_item\_\_调用 GetFunction获取 cxx端返回PackedFunc;
  * key function **BuildRelay in build_module.cc**
  * **BuildRelay**: 
    * **Optimize** ---> 图级别的优化，最重要的可能是 fuse_ops，从算法上看是基于 dataflow分析;在 graph level 优化后，CallNode中对应的 Op不为 OpNode，因其被fuse，为 FunctionNode;
    * **Codegen** ---> MakeExecutorCodegen (ret GraphCodegen, AOTCodegen), 因此 Codegen也为对应GraphCodegen 或 AOTCodgen的方法;
    * **GraphExecutorCodgen::Codegen**: GraphPlanMemory -> convert input params -> Visit func body -> update metadata & ret.
    * **GraphExecutorCodgen::CallNode**: 这里仅可为 FunctionNode，获取func，在其上作用 _make_CCacheKey 以及 **_CompileEngineLower**，经过 lower后会选择合适的 Op实现，并且将 FunctionNode最终转换为 PrimFunc，其内部主要调用 **LowerInternal**;
    * **LowerInternal**: CreateSchedule -> LowerSchedule, 主要创建 Schedule，这里就会将FuncNode中全部变为 Operation; 如下可以展示，中间到底发生了啥;
    ```c++
    // Just relay.nn.dense & relay.sigmoid
    
    //************** bef create schedule **************
    FunctionNode([Var(p0, ty=TensorType([5, 5], float32)), Var(p1, ty=TensorType([1, 5, 5], float32))], TensorType([5, 5], float32), CallNode(Op(sigmoid), [CallNode(Op(nn.contrib_dense_pack), [Var(p0, ty=TensorType([5, 5], float32)), Var(p1, ty=TensorType([1, 5, 5], float32))], relay.attrs.DenseAttrs(0x218cde8), [TensorType([5, 5], float32), TensorType([1, 5, 5], float32)])], (nullptr), [TensorType([5, 5], float32)]), [], {"Primitive": 1, "hash": "a2d3f5d197085d29"})

    //**************aft create schedule, the sch **************
    stage(placeholder, placeholder(placeholder, 0x1fb22c0))
    stage(placeholder, placeholder(placeholder, 0x1f7ae60))
    stage(compute.global, compute(compute.global, body=[reduce(combiner=comm_reducer(result=[(x + y)], lhs=[x], rhs=[y], identity_element=[0f]), source=[(placeholder[y.c, k]*placeholder[floordiv(x.c, 5), k, floormod(x.c, 5)])], init=[], axis=[iter_var(k, range(min=0, ext=5))], where=(bool)1, value_index=0)], axis=[iter_var(y.c, range(min=0, ext=5)), iter_var(x.c, range(min=0, ext=5))], reduce_axis=[iter_var(k, range(min=0, ext=5))], tag=dense_pack, attrs={"workload": ["dense_pack.x86", ["TENSOR", [5, 5], "float32"], ["TENSOR", [1, 5, 5], "float32"], (nullptr), "float32"]}))
    stage(compute, compute(compute, body=[compute.global[y, x]], axis=[iter_var(y, range(min=0, ext=5)), iter_var(x, range(min=0, ext=5))], reduce_axis=[], tag=dense_pack, attrs={"workload": ["dense_pack.x86", ["TENSOR", [5, 5], "float32"], ["TENSOR", [1, 5, 5], "float32"], (nullptr), "float32"]}))
    stage(T_sigmoid, compute(T_sigmoid, body=[tir.sigmoid(compute[ax0, ax1])], axis=[iter_var(ax0, range(min=0, ext=5)), iter_var(ax1, range(min=0, ext=5))], reduce_axis=[], tag=elemwise, attrs={}))

    //*** lower primfunc name: fused_nn_contrib_dense_pack_sigmoid ***
    IRModule({GlobalVar(fused_nn_contrib_dense_pack_sigmoid): PrimFunc([placeholder, placeholder, T_sigmoid]) attrs={"global_symbol": "fused_nn_contrib_dense_pack_sigmoid", "tir.noalias": (bool)1} {
        parallel (ax1.outer.ax0.outer.fused, 0, 5) {
            // attr [compute.global] storage_scope = "global"
            allocate compute.global[float32 * 5]
            compute.global[ramp(0, 1, 5)] = x5(0f)
            for (k.outer, 0, 5) {
                compute.global[ramp(0, 1, 5)] = (compute.global[ramp(0, 1, 5)] + (x5(placeholder[((ax1.outer.ax0.outer.fused*5) + k.outer)])*placeholder[ramp((k.outer*5), 1, 5)]))
            }
            for (ax1.inner.inner.s, 0, 5) {
                T_sigmoid[((ax1.outer.ax0.outer.fused*5) + ax1.inner.inner.s)] = tir.sigmoid(compute.global[ax1.inner.inner.s])
            }
        }
    }
    })
    ``` 
    * **ScheduleGetter**: the CreateSchedule impl, call Create, convert and add inputs/outputs, visit body;
    * **ScheduleGetter::CallNode**: 主要使用 relay.backend.lower_call 来进行 impl的选择，该函数定义在 python side， **select_implementation** 完成 op 的选择;
    * 在select implementation中首先通过 get_valid_implementations获取当前对应target下所有注册strategy;
    * 通过调用 fstrategy -> ret op_strategy，注意这里Op对应的**fstrategy**通过op.get_attr("FTVMStrategy")获取，这玩意返回的是个**GenericFunc**，看下GenericFunc其中**CallPacked**定义：
    ```c++
    void GenericFunc::CallPacked(TVMArgs args, TVMRetValue* ret) const {
        auto node = static_cast<const GenericFuncNode*>(get());
        auto target = Target::Current(true);
        PackedFunc func;

        if (target.defined()) {
            for (auto& k : target->GetKeys()) {
                auto iter = node->dispatch_dict_.find(k);
                if (iter != node->dispatch_dict_.end()) {
                    func = iter->second;
                    break;
                }
            }
        }
    ``` 
    注意从中可以看到在target定义的情况下，将从GenericFunc的dispatch_dict_中选取对应平台的实现，那么对于fstrategy而言也会选在对应平台的实现。
    * 那么剩下的问题就在于 fstrategy何时被注册到op中，以及 其中 dispatch_dict_中何时注册了对应target函数;
    * **register_strategy**： relay/op/op.py中对应函数，**tvm.ir.register_op_attr(op_name, "FTVMStrategy", fstrategy, level)**， 注意这里将完成把 GenericFunc注册到Op中的;
    * 对应register_strategy何时被调用？ 其通常那个在op中对应 _xxx.py文件中调用，如dense op在 _nn.py中调用reg.register_strategy("nn.dense", strategy.dense_strategy); 注意这个是将 dense_strategy(GenericFunc)注册进入op的属性中，而同GenericFunc的其他平台实现这之前就已被注册进入 dispatch_dict_中，通过调用GenericFunc的register; 默认的strategy的定义在 relay/op/strategy/generic.py中;
    ```c++
    @override_native_generic_func("dense_strategy")
    def dense_strategy(attrs, inputs, out_type, target):
        """dense generic strategy"""
        logger.warning("dense is not optimized for this platform.")
        strategy = _op.OpStrategy()
        strategy.add_implementation(
            wrap_compute_dense(topi.nn.dense),
            wrap_topi_schedule(topi.generic.schedule_dense),
            name="dense.generic",
        )
        return strategy
    ``` 
    * 这里为对应dense_strategy定义的地方，而装饰器将创建名为 **dense_strategy** 的GenericFunc， 并将fdefault设置为这里的dense_strategy;
    ```c++
    @dense_strategy.register("cpu")
    def dense_strategy_cpu(attrs, inputs, out_type, target):
        """dense x86 strategy"""
        strategy = _op.OpStrategy()
        same_type = inputs[0].dtype == inputs[1].dtype == out_type.dtype
        dtype = inputs[0].dtype
        u8s8s32 = dtype == "uint8" and inputs[1].dtype == "int8" and out_type.dtype == "int32"
        print("call the dense_strategy_cpu", attrs, inputs, out_type, target)

        strategy.add_implementation(
            wrap_compute_dense(topi.x86.dense_nopack),
            wrap_topi_schedule(topi.x86.schedule_dense_nopack),
            name="dense_nopack.x86",
            plevel=5,
        )

        strategy.add_implementation(
            wrap_compute_dense(topi.x86.dense_pack),
            wrap_topi_schedule(topi.x86.schedule_dense_pack),
            name="dense_pack.x86",
            plevel=10,
        )
    ``` 
    举个例子在strategy/x86.py中定义了dense_strategy_cpu实现，这里会将通过调用GenericFunc的register将"cpu"作为dispatch_dict的key，dense_strategy_cpu作为对应PackedFunc value添加进入dispatch_dict_中; 至此可以看到合适注册strategy GenericFunc以及对应平台实现，到最中怎么调用到具体target的fstrategy函数，从而返回对应平台的op_strategy.
    * Comments:到这可以看出对于 tvm relay来说，和 TIR 层级不同点在于其计算图组成有很多的 CallNode组成，在使用 relay描述计算时，其对应为Op，而在 optmize, lower等过程中，其通过fuse变换为 FunctionNode，此也可以看作将 graph 划分为多个子图，而每个FunctionNode则会被 lower为对应的 primfunc, 此时主要通过将 FunctionNode中对应 Call选择合适的 Op实现(i.e., schedule)，并且对于 CallNode 添加 inputs/outputs tensors，其实这里是将其转为 TIR 层级的计算图 (Op-Tensor graph).
* **TODO fuse_ops**

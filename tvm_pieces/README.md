# Boring TVM Code

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
* *** TIR Buffer ***
    * 看了下TIR中Buffer的管理，因利用IRBuilder将Tensor创建buffer_ptr时出错。其实因为Tensor会在lower的过程中最后转换为Buffer，因此这种行为相当于将lower过程中搞得提前到了Tensor赋值的过程中来，故报错，若非要解决这个问题只能根据Tensor的dtype, shape重新构建Buffer才行。
    * TIR Buffer在 C++端(src/tir/ir/buffer.cc, include/tir/buffer.h) 和 Python端(python/tvm/tir/buffer.py)行为基本一致，因Python Buffer class 直接调用C++ register的类构造函数 以及 成员函数，其中decl_buffer也是直接调用C++端的函数。BufferNode中基本三个member为 Var data -> 指向数据的指针，DataType dtype -> 数据类型 以及 Array<PrimExpr> shape -> buffer的shape大小。那关于Buffer的内存分配，看了下codegen_llvm.cc中，llvm的IRBuilder只有在Allocate Stmt的时候，才会调用 CreateAlloca 即分配内存，故Buffer的内存管理主要还是由 lower的过程当中，适时地插入Allocate Stmt时才完成。那么关于 Store / Load 基本是先获取到 对应Buffer，后利用 llvm IRBuilder 创建指针访问。
    * python端 ir_builder 使更容易的获取buffer元素 ---> BufferVar 类，类构建irbuilder, buffer_var(Var), content_type；通过__setitem__ (emit Store) 和 __getitem__ (emit Load)来创建IR，同时注意 buffer_ptr函数接收一个Buffer对象，将其 data 和 dtype提取并创建BufferVar对象，就可以A[i]这种方式获取Buffer中的元素，免去了很多麻烦。pointer函数根据类型先创建 Var，后创建 BufferVar绑定到刚创建的Var上。allocate函数会直接emit allocate stmt. 
  
### *2021.2.19*
* ***Storage Flatten Pass***
    * 这个pass的主要作用是将 multi-dim 的访问 变换为 1-d buffer array的访问StorageFlattener 构造时针对 primfunc 外部传入的 buffer_map (Var-Buffer) 构造内部 buf_map_ (Buffer-BufferEntry)继承自  StmtExprMutator，重载其中对于 Stmt 以及 Expr的访问，对于primfunc中的body进行变换；
    * 在经过 schedule_ops 及 schedule_postproc_to_primfunc两个pass之后，body的形式在 MakePipeline的时候在每个 OP 对应的外层部分一般为 attr_stmt realize_scope；ProducerRealizeNode，ProducerStoreNode，ProducerLoadNode变换为对应的 BufferRealize, BufferStore, BufferLoad；
    * 对于 AttrStmt解析，realize_scope (每个op均有)，获取stage 所对应的 storage_scope后对body进行递归分析，因为在MakePipeline的时候，每个producer的最外层为realize_scope，因此定会对当前op中所对应的stmt body进行遍历，主要针对其中的 BufferRealizeNode；
    * 对于 BufferRealizeNode，若对应的buffer在 buf_map_中找到(首次为外部buf_map)，则返回即可，因默认外部buffer 已经分配好；否则创建BufferEntry，获取对应bounds (extent)创建shape，获取storage_scope；计算buffer aligmnet信息前会先依据shape计算allocate的const_size(即shape乘积) 后获取 align对齐(默认128字节对齐)；而dim级别的align信息仅针对compute，此之前对 buffer_dim_align 的解析，可以获取到dim_info_(map)后对每一维度创建strides(其他op可为空)，依据之前信息创建新的Buffer，并添加进入 buf_map_中；并创建对应的 AllocateNode；从这个角度来看，BufferRealizeNode主要目的在于创建 整个Pipeline中的 Buffer，总感觉放在 StorageFlatten Pass中怪怪的。
    * 对于 BufferLoadNode，check buffer存在后对于bounds和indices对于Index重新变换，之后主要返回Buffer vload (tir::Load)，Flatten体现，计算偏移通过 BufferOffset -> ElemOffset；同理对于BufferStoreNode，也是一样的流程，不过为 Buffer vstr::Store)，偏移计算也是通过 BufferOffset -> ElemOffset将多维访问变为一维；
    * 对于VarNode，LoadNode，StoreNode则判断是否需要对其中 Var进行 remap(仅当ExternOp时需要)，不需要则直接返回即可；
    * 对于 ExternOp则调用 handle_bind_scope进行处理，因为其在Provide的时候会对Op 传入的外部Buffer和该Op output tensor 和 input tensor 添加 buffer_bind_scope, 经过先前的处理, tensor已经为对应的buffer, 主要作用是先做 begin, extents的变换，后将对应buffer make Slice为对应的 shape 后调用 ArgBinder 对于将 buffer 中 Var bind 至 在buf_map中找到的 target buffer对应的 Var上，具体其实是在var_remap中添加对应 Var的替换关系后，Visit AttrStmt中的 body, 而 body中的 VarNode, LoadNode, StoreNode中对应的 Var则会被替换为对应的 target buffer 的 Var, 对 buffer_bind_scope AttrStmt处理完成之后，则清除掉对应的 var_remap关系；
### *2021.3.7*
* ***Some notes***
    * Writing a Customized Pass: https://tvm.apache.org/docs/tutorials/dev/low_level_custom_pass.html#sphx-glr-tutorials-dev-low-level-custom-pass-py以前还真没注意怎么在 python 端写 tvm lower pass的，基本还是和 cxx side 差不多， python 端更加不灵活吧stmt_functor 中 post_order_visit 获取想 modify 的 IR， stmt_functor.ir_transform进行变换， stmt_functor.substitue替换，将 Pass可通过在 tvm.transform.PassContext 中添加
    * 从 compile models来看，首先是 relay.frontend.from_xxx， 后创建 graph executor， relay.build_module.create_executor那其实 第一步主要就是解析 不同前端框架 保存模型参数的一个过程，然后匹配成 relay_func 的格式；其中不同前端中的OP(conv, pool)之类的会被的等价转换为 relay 中 op，relay中 build_module过程中调用cxx端的 RelayBuildModule 中 build，后调用  BuildRelay， 其中会创建 graph_codegen 通过  _GraphRuntimeCodegen 创建  GraphRuntimeCodegenModule 调用其中 Codegen方法，在 visit 其中 CallNode时候，_CompileEngineLower，其中会根据 OpStrategy完成对不同OP实现的选择
### *2021.3.13*
* ***Tensor IR First Impression***
    * 粗略地看了下RFC(https://discuss.tvm.apache.org/t/rfc-tensorir-a-schedulable-ir-for-tvm/7872)上的讨论, TensorIR是TIR可优化的增强版, 主要提出 Block statement structure来warp IR, Block会作为最小的scheduling and tensorization的单元，其中包含 Iter Vars, Block读写region, allocated buffer 以及 body stmt. 从这点来看，是在已有的TIR的基础上新增了更加粗粒度的访问块，在Block unit中附加上schedule所需的信息，从而实现直接对于TIR的schedule；
    * 几点好处: 1). schedule直接作用在 TIR, 而不需要通过在schedule tree调度后拼接 stmt形成 TIR，TF/PyTorch/ONNX -> Relay -> TIR -> schedule -> TIR -> scheudle -> TIR -> C++/CUDA; 2). 更加灵活的描述，相对于TE; 3). 更好地支持 memory hierarchy 以及 execution hierarchy; 4). 对于 tensorization 支持更加灵活; 5). 可以在每次schedule对于IR进行验证
    * 目前来看, 关于Block 的 PR已经提了;
### *2021.3.15 & 16*
* ***Storage Rewrite Pass***
    * Function: Memory access pattern analysis and optimization, re-write data access to enable memory shareing as mush as possible.
    * Rewrite calls: LinearAccessPatternFinder -> LivenessAnalysis -> PlanMemory -> PrepareNewAlloc -> rewrite by operator() -> MakeAttach 
    * LinearAccessPatternFinder ---> 主要作用是:  
      * 1). 对每个storage_scope所对应的 Var 在 alloc_info_ info中添加, 并set storage scope.  
      * 2). 对于For/IfThenElse/Assert等stmt, thread_extent/extern_scope/virtual_thread对应的attr stmt, 调用 VisitNewScope 将其分割开, 每个VisitNewScope 会在 linear_seq_ 添加对应的 before_scope, after_scope, 而该 scope 中的分析递归 visit op即可, 因此会有 nested_scope的情形出现；每次进入会在 scope_ 中 push StmtEntry, 而 after_scope后则 pop, 借用 scope_ stack 可以获取到当前的 Stmt Node处于 哪个 scope 中, 使用 scope_level 来标识. 每次的 touched vars 仅在 after_scope时候更新，因此 touched size 不为 0的 offset均为负.  
      * 3). 那么对于 Allocate Stmt, 获取对应 buffer_var 的 AllocEntry, 更新对应 Allocate Stmt以及 scope_level 为当前 Allocate Stmt所在位置.
      * 4). 对于其他会使用到 Var的 Stmt (Store, Load, Var), 在alloc_info中获取 Var 对应的 Alloc Stmt所在的 scope_level, 并在对应 level scope_ 中添加该 Var 为 touched, 即在此level的 scope中会有这些 Var访问到Allocate所分配的buffer_var.
      * 5). 两个重要的数据结构: StmtEntry & AllocEntry, StmtEntry 主要标识当前 scope 对应 Stmt, scope在 linear_seq_中范围, 以及其牵扯到的 Var访问(R/W);     AllocEntry 主要标记每个 Allocate 所在的 scope 层级, 以及对应 Allocate Stmt.
      * 6). 两个被后续使用成员: linear_seq_ & alloc_info_
      * linear_seq_: vector<StmtEntry\>, 其中包含了所有scope的起始和终止(bef_scope, aft_scope), 可获取每个scope中对应访问(R/W)过的所有 Vars;
      * alloc_info_: unordered_map<const VarNode*, AllocEntry\>, 可获取得到所有 Var所对应的 Alloc (stmt,scope_level, storage_scope).
    * LivenessAnalysis:  find gen and kill point of each variable by filling the event_map_ ---> unordered_map<const Object*, EventEntry\>
      * EnventEntry: vector<cosnt VarNode*\> gen, vector<const VarNode*\> kill   
      * kill point: 逆序遍历 linear_seq_, 即从最后的 nested scope开始向前, 即从最后使用 touched vars的scope开始, 此即为其中scope touch vars的生命周期, 故 对应Var 因在之后被kill. 每次在 event_map_ 中 对scope stmt 对应的 EnventEntry的 kill 中添加 touched vars (not visited yet).
      * gen point: 顺序遍历 linear_seq_, 从最初的nested_scope 开始，即最先使用该 Var的 scope 开始, 故对应 Var因在此前被 gen. 每次在event_map_ 中对 scope stmt 对应的 EventEntry 的 gen中添加 touch vars (not visited yet).
    * PlanMemory: Utilize linear_seq_ and alloc_info_ to do memory planning  
      *  PlanMemory所使用信息: linear_seq_ -> scope 对应 stmt 及 scope 中 touched vars; alloc_info_ -> VarNode 所对应 Alloc Stmt, storage_scope and scope_level; event_map_ -> 每个 scope 对应 stmt 所包含的 gen vars & kill vars;
      * ```c++
        for (i = 0; i < seq.size(); i++)  
            find seq[i].stmt in event_map_  
                // begin scope process, offset >= 0
                // stmt denotes seq[i].stmt for simplicity
                if found && offset >= 0
                    for (var : stmt related gen vars)
                        detect inplace if only gen < 2
                            for (var : stmt related kill)
                                if exist gen var in kill vars (may exist in place)
                                    InplaceOpVerifier
                                    if pass inplace check
                                        StorageEntry of this dst_entry(gen) is src_entry(kill)
                        dst_entry = FindAlloc(gen var related alloc ...)
                        dst_entry->allocs add related alloc
                        add <var, dst_entry\> in alloc_map_
                // enter/exit new_scope
                if AttrStmt
                    if thread_extent || virtual_thread
                        PlanNewScope -> update thread_scope_ to op
                else if For
                    if Parallel type For
                        PlanNewScope -> udpate thread_scope_ to op
                // exit the scope, we can free the stmt related kill vars
                if found && offset <= 0
                    for (var : stmt reated kill vars)
                        Free(var) if not inplace substitute
        ```
     * 这个 PlanMemory还是相当复杂啊，1).首先对event_map_中对应的**gen vars**(此时外层先被access)进行**内存分配**即创建对应的 StorageEntry 并添加进入 alloc_map_中; 分配前做 InplaceOp检测，主要会借助 InplaceOpVerifier 检测 inplace 操作，若是对var进行合并，复用之前 StorageEntry即可；若不是inplace，则调用 FindAlloc 创建新的 StorageEntry 并进行内存分配；之后将对应分配结果添加进入 alloc_map_中. 2). 之后判断是否 enter/exit new_scope，这个主要跟 thread强相关，在 AttrStmt为 thread_extent 和 virtual_thread的时候会对 thread_scope_ 更新，在 For 为 parallel(即做了 parallel) 优化后的也会更新 thread_scope_. 3). 最后对 event_map_中对应 **kill vars**, offset为负(内层先被access, also free first). 针对对应的 kill var，若其没有被 inpalce操作，则 Free, 其实是在const_free_map_ & sym_free_list_中添加 var 对应 StorageEntry，从而被之后的 FindAlloc时可复用.
     * PlanMemory -> InplaceOpVerifier
     * PlanMemory -> FindAlloc
     * PlanMemory -> FindAlloc -> NewAlloc
     * PlanMemory -> Free
     * PlanMemory -> PlanNewScope
     * PrepareNewAlloc
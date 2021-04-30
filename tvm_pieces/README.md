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
* ***TIR Buffer***
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
    * TODO(添加 ProducerRealizeNode -> Postproc_to_Primfunc 中 Buffer -> Flatten 中 Buffer中间的差异，以及为什么整这么复杂)
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
    * Function: Memory access pattern analysis and optimization, re-write data access to enable memory shareing as mush as possible.
    * Rewrite calls: LinearAccessPatternFinder -> LivenessAnalysis -> PlanMemory -> PrepareNewAlloc -> rewrite by operator() -> MakeAttach 
    * **LinearAccessPatternFinder** ---> 主要作用是:  
      * 1). 对每个storage_scope所对应的 Var 在 alloc_info_ info中添加, 并set storage scope.  
      * 2). 对于For/IfThenElse/Assert等stmt, thread_extent/extern_scope/virtual_thread对应的attr stmt, 调用 VisitNewScope 将其分割开, 每个VisitNewScope 会在 linear_seq_ 添加对应的 before_scope, after_scope, 而该 scope 中的分析递归 visit op即可, 因此会有 nested_scope的情形出现；每次进入会在 scope_ 中 push StmtEntry, 而 after_scope后则 pop, 借用 scope_ stack 可以获取到当前的 Stmt Node处于 哪个 scope 中, 使用 scope_level 来标识. 每次的 touched vars 仅在 after_scope时候更新，因此 touched size 不为 0的 offset均为负.  
      * 3). 那么对于 Allocate Stmt, 获取对应 buffer_var 的 AllocEntry, 更新对应 Allocate Stmt以及 scope_level 为当前 Allocate Stmt所在位置.
      * 4). 对于其他会使用到 Var的 Stmt (Store, Load, Var), 在alloc_info中获取 Var 对应的 Alloc Stmt所在的 scope_level, 并在对应 level scope_ 中添加该 Var 为 touched, 即在此level的 scope中会有这些 Var访问到Allocate所分配的buffer_var.
      * 5). 两个重要的数据结构: StmtEntry & AllocEntry, StmtEntry 主要标识当前 scope 对应 Stmt, scope在 linear_seq_中范围, 以及其牵扯到的 Var访问(R/W); AllocEntry 主要标记每个 Allocate 所在的 scope 层级, 以及对应 Allocate Stmt.
      * 6). 两个被后续使用成员: linear_seq_ & alloc_info_
      * linear_seq_: vector<StmtEntry\>, 其中包含了所有scope的起始和终止(bef_scope, aft_scope), 可获取每个scope中对应访问(R/W)过的所有 Vars;
      * alloc_info_: unordered_map<const VarNode*, AllocEntry\>, 可获取得到所有 Var所对应的 Alloc (stmt,scope_level, storage_scope).
      * 在这里需要注意理解 scope, scope主要在这里用以来划定 程序局部区域, 其包含了范围那些变量的信息, 可以来帮忙界定对应 alloc 的生存范围, 比如若存在某个变量在此区域后就未被使用了, 那么其可在该 scope被 kill, 因此后续则会被添加到 scope 对应的 kill vars中; 若存在某个变量在该区域第一次使用, 那么其应在这里被 gen, 也就是分配内存的操作直到这里才开始进行; 而 linear_seq_ 中 scope 往往是 nested scope, 对应 offset很可能出先类似 [8, [3, [1, -1], -3], [3, [1, -1], -3], -8], 注意这里的 从 + 进后 从对应 - 出的时候,  若有 kill vars, 则 free掉; 在后续和该 scope 同级的 scope 就有复用 被释放掉 vars的机会 (storage rewrite 的主要目的). 
    * **LivenessAnalysis**:  find gen and kill point of each variable by filling the event_map_ ---> unordered_map<const Object*, EventEntry\>
      * EnventEntry: vector<cosnt VarNode*\> gen, vector<const VarNode*\> kill   
      * kill point: 逆序遍历 linear_seq_, 即从最后的 nested scope开始向前, 即从最后使用 touched vars的scope开始, 此即为其中scope touch vars的生命周期, 故 对应Var 因在之后被kill. 每次在 event_map_ 中 对scope stmt 对应的 EnventEntry的 kill 中添加 touched vars (not visited yet).
      * gen point: 顺序遍历 linear_seq_, 从最初的nested_scope 开始，即最先使用该 Var的 scope 开始, 故对应 Var因在此前被 gen. 每次在event_map_ 中对 scope stmt 对应的 EventEntry 的 gen中添加 touch vars (not visited yet).
    * **PlanMemory**: Utilize linear_seq_ and alloc_info_ to do memory planning  
      *  PlanMemory所使用信息: 
         * linear_seq_ ---> scope 对应 stmt 及 scope 中 touched vars; 
         * alloc_info_ ---> VarNode 所对应 Alloc Stmt, storage_scope and scope_level; 
         * event_map_ ---> 每个 scope 对应 stmt 所包含的 gen vars & kill vars;
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
                        add <var, dst_entry> in alloc_map_
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
     * **PlanMemory**比较复杂...  
       * 1).首先对event_map_中对应的**gen vars**(此时外层先被access)进行**内存分配**即创建对应的 StorageEntry 并添加进入 alloc_map_中; 分配前做 InplaceOp检测，主要会借助 InplaceOpVerifier 检测 inplace 操作，若是对var进行合并，复用之前 src的 StorageEntry即可；若不是inplace，则调用 FindAlloc 创建新的 StorageEntry 并进行内存分配；之后将对应分配结果添加进入 alloc_map_中. 
       * 2). 之后判断是否 enter/exit new_scope，这个主要跟 thread强相关，在 AttrStmt为 thread_extent 和 virtual_thread的时候会对 thread_scope_ 更新，在 For 为 parallel(即做了 parallel) 优化后的也会更新 thread_scope_. 
       * 3). 最后对 event_map_中对应 **kill vars**, offset为负(内层先被access, also free first). 针对对应的 kill var，若其没有被 inpalce操作，则 Free, 其实是在const_free_map_ & sym_free_list_中添加 var 对应 StorageEntry，从而被之后的 FindAlloc时可复用. 比如 [8, [3, [1, -1], -3], [3, [1, -1], -3], -8], 进行是否可 free 的判断顺序会为, -1, -3, -1, -3, -8; 可以看出 nested_scope 中 free 后的 var 的复用也是同 free scope 所同级别的 scope, 不被 free 当前所在的scope所包含在内; 
     * **PlanMemory -> InplaceOpVerifier**  
       * 主要验证刚被 kill 的 var(src) 是否 可被下一个 gen 的 var(dst)所复用(也就是当前 scope 存在gen var后 立马释放 对应var, 那么对应的op很有可能就是 inplace opeartion); 这个 Verifier主要验证  dst[index\] = f(src[index\]), 具体判断规则看代码;
     * **PlanMemory -> FindAlloc**   
       * 在没有 inplace 复用的情况下, 调用 FindAlloc 进行 StorageEntry 的分配; 
       * FindAlloc/Free 和 linear_seq_中scope 对应 event_map_中 gen/kill vars进行联动易求完成对前一 scope free的 vars 的复用; 
       * Free 的 Vars 会首先找到 alloc_map_ 中对应 StorageEntry, 并根据alloc const_n_bits 是否为 0 将对应 StorageEntry 添加进入 const_free_map(const size) 及 sym_free_list(dynamic size)中; 
       * 若在 const_free_map / sym_free_list 中找到满足条件的 StorageEntry, 则复用对应的 StorageEntry, 且返回后在对应的牵扯 allocs中添加到该 var对应 alloc, 即该alloc会被合并; 没找到就很直接，NewAlloc就行了, NewAlloc create 的 new StorageEntry 会被添加进 alloc_vecs_中; 
       * 此时的 attach_scope_ 为外部传入, 而该 attach_scope 为 thread_scope_ 尽在 PlanNewScope的时候会被更新;
     * **PlanMemory -> PlanNewScope**: 这个还需更深入理解下 thread_scope_ 切换的时机, 目前来看 AttrStmt(thread_extent/virtual_thread) / For(parallel) 层级的 scope 的 gen vars是在这之前完成的, gen vars成功分配 StorageEntry之后才会进入这个 PlanNewScope, 所以 update thread_scope_ 之后 gen vars 对应的 scope 应包含在当前这个 scope之内, 这之内的 gen vars就希望进行局部 alloc/free, 且不可被其他 var复用分配内存;
     * **PrepareNewAlloc** -> 在上述 alloc_map_, alloc_vecs_这些准备好了之后 fill attach_map_  
       * 首先将 alloc_vecs中所有的 StorageEntry 中 <attach_scope_, StorageEntry*\>作为 kv 对添加进入 attach_map_中
       * 遍历 attach_map_ 中 kv对, 遍历同一 key (attach_scope) 中所有 StorageEntry 更新其中 new_alloc 为新创建的 Allocate; 这当中会根据 StorageEntry中 allocs的个数判断是否进行 allocate合并. allocs.size() > 0 就是有多个 alloc 共享这个 StorageEntry 此时新创建 Allocate 分配大小为 几个 alloc中最大; 若为0, 直接create Allocate即可.
     * **Rewrite**: 使用 attach_map_ 对 当前 stmt进行 rewrite, 因在之前的处理当中, inplace op被合并, 可被复用的内存已被复用; 针对 thread 的 thread_extent, virtual_thread, For parallel, 对应的 Alloc 被 attach 到 当前的 op (AttrStmt, For op); 因此, 此时 attach_map_ 含有 nullptr 对应 global allocs, others attached op 对应的 local allocs. 那么当前的策略是:  
       * 1). 对 thread 相关的 thread_extent, virtual_thread, For, 若 attach 到对应 op, 则 MakeAttach 拼接 该 attach_scope 对应的 所有内存分配 StorageEntry对应 alloc 和 op body; 原地 alloc, 且 alloc 对应的 body为该 op body, 此时其作用范围仅限 op_body 对应部分, 超出即释放.
       * 2). 对于其他 Stmt Node, 如 StoreNode, LoadNode, VarNode, CallNode 等，则在 alloc_map_中查询对应新 alloc 的 buffer_var 并更新其中 var的部分后 return modified stmt 即可. 而 attach_scope 为 nullptr 的 global 内存分配 (StorageEntry), 会在所有 stmt更新完成之后, 调用 MakeAttach 拼接 该 attach_scope(global) 对应的所有 allocs 和 stmt, 即此时会将这部分 alloc提升至全局.

### *2021.3.23 & 24 & 25 & 26 & 27 & 28*
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

### *2021.4.20 & 4.21*
* ***Ansor reading notes***
    * **Contributions**:
        * A mechanism to generate a **large hierarchical search space** of tensor programs for a **computational graph**.
        * An **evolutionary strategy** with a learned cost model to fine-tune the performance of tensor programs.
        * A **scheduling algorithm** based on gradient descent to **prioritize important subgraphs** when optimizing the end-to-end performance of DNNs.
        * An implementation and comprehensive evaluation of the Ansor system demonstrating that the above techniques outperform state-of-the-art systems on a variety of DNNs and hardware platforms.  
    * **Previous work**:
        * **Templated based search**: require the user to write the template for several parameters first, autotvm in tvm. Also, limited to single operator optmization.
        * **Sequential construction based search**: sequentially unfold each node in the computational graph based on decision making, select the top-k candidates via learned cost model, others are pruned(incomplete search space). The auto-scheduler in Halide. The cost model estimateing the perf of incomplete program is difficult.
        * **hierarchical approach**: this paper.
    * **Framwork design**:
        * 3 major components: (1) a program sampler that constructs a large search space and samples diverse programs from it; (2) a performance tuner that fine-tunes the performance of sampled programs; (3) a task scheduler that allocates time resources for optimizing multiple subgraphs in the DNNs.
        * **Program Sampler**: two levels: sketchs -> high-level structure of program, annotations -> low-levle details of program. 
        * **Performance tuner**: At each iteration, Ansor uses re-sampled new programs as well as good programs from previous iterations as the initial population to start the evolutionary search.
        * **Task Scheduler**: allocate time resource for searching for each subgraph extracted via relay program, not the end-to-end search for the whole graph.
    * **Program Sampling**:
        * **Sketch generation**: Based on the derivation rules(six rules in paper) to iteratively apply node from end to the first (topological order of DAG).
        * **Random Annotation**: Randomly sample the details of the program based on the generated sketch, including: tile sizes and loop annotations, such as parallel, unroll, and vectorization.
    * **Perforamce Tuning**:
        * **Evolutionary Search**:  starts from the sampled initial generation. **mutations**: tile size, parallel, pragma, computation location. Along with the mutation, contains **node-level crossover**.
        * **Cost Model**: Based on xgboost, to use the cost model select the candidates programs first and then evaluate these programs on the real machine to update the parameters of the cost model. In this paper, use the L2 loss to train the boosting tree.
    * **Task scheduler**: the schedule algorithm based on gradient-descent to allocate time resources to sub-graphs of the DNN, i.e., tasks.
    * **Limitations**:
        * ***Dynamic shape supprot***.
        * ***Only works well on dense operators, fail on sparse operator like sparse matrix multiplication and GCN.***
        * ***Perform optimization on high level, rely the LLVM & NVCC to do machine-depedent optimizations.***
        * ***Short usage of the special instructions(tensorcore, arm).***, this may due to the weakness of the current tensorization way to utilize the special instruction ? 
        * ***Combination of ansor and graph-level optimization***, i.e., the end-to-end optimization for the whole network graph.

### *2021.4.22 & 4.24 & 4.25 & 4.26 & 4.30 & 5.1 & 5.2 & 5.3*
* ***auto-schedule user-defined operator tracing***:
    * **workload_registry.py**: Workload registration and serialization.
        * WORKLOAD_FUNC_REGISTRY = {} -> Global workload function and hash key registry; 1). user registerd task via decorator **register_workload**. 2). extract tasks from relay program via function **register_workload_tensors**. **register_workload** will register the function_name & func_pointer to the WORKLOAD_FUNC_REGISTRY.
    * **search_task.py**: create the SearchTask object via registered func, arguments of func(static size, no dynamic shape), target. Contains the computation information and hardware parameters for a schedule search task.
        * _ffi.register_object int the cxx side, auto_scheduler.SearchTask.
        * Create the search task either via function or via workload_key.
        * For function input, call the **make_workload** to serialize the function_name with args to workload_key via json dump. Create **ComputeDAG** object via workload_key and utilize the default layout_rewrite_options. Finally, call the cxx constructor to construct the object.
        * The **HardwareParams** object and **TuningOptions** object just call the constructor in cxx side.
        * **tune** func create the search_policy if input one is None. search policy firt construct **XGBModel** cost model and then create **SketchPolicy** object. Finally, can the **_ffi_api.AutoSchedule** in cxx size via params(search_policy and tuning_options).
    * **compute_dag.py**: auto-scheduler's computational graph and related program analyses.
        * **ComputeDAG** object create, input to cxx side is the compute_tensors(out_tensor ReadGraph postorder visit) and the sch(None). Or compute(None), the (tvm.te.schedule) is not none.
        *  Currently omit the detals of the such as the loop_state related functions.
        *  Function: It keeps the input/output tensors, all operations in the DAG, and some static analysis results for the DAG (e.g. the total float operation count, consumer/producer relations of operations, whether an operation stage should be tiled/compute inlined ...). These analyses can help the search policy to make decisions during the search.
    * **search_task.h/search_task.cc**
        * In search_task.h, define the class HardwareParams, HardwareParamsNode, SearchTask, SearchTaskNode. just same members consistent with python side.
        * search_task.cc, just constructor method and register the method. The **GetDefaultHardwareParams** decrible the how to set default params in different targets.
    * **compute_dag.h/compute_dag.cc**
        * Members of ComputeDAGNode: Array<te::Tensor> tensors(only input/output); Array<te::Operation> ops(all ops in the schedule); double flop_ct; State init_state; AccessAnalyzer access_analyzer(do the static analysis);
        * Brief overview the ComputeDAGNode, can find the auto-schedule do lots of analysis on the compute_dag.
        * Create object: from in/out tensors, topo order -> te.create_schedule. from sch, obtain placeholders and the stage op marked as the output.
        * (TO FILL) -> placeholder
    * **auto_schedule.h/auto_schedule.cc**
        * **TuningOptionsNode** definition, and the interface for python to call the AutoSchedule.
        * The AutoSchedule function create the **ProgramMeasurer** based on the tuning_options. Then call **SearchPolicy** search method with tunining options and created measurer. If the loop_state ret is valid, apply the transform_steps of the loop_state and ret the te::Schedule. The input SearchPolicy is created in python side via the SketchPolicy.
    * **search_policy.py**
        * the python side actually is just interface to call the cxx side files, also provide several funcs for debuging and testing.
        * SearchPolicy -> EmptyPolicy, SearchPolicy -> SketchPolicy.
        * Note **PreloadCustomSketchRule** can be used to register user-defined sketch rule satisfied with requirement.
    * **search_policy.h/search_policy.cc**
        * Declare the base class of search policies. **SearchPolicyNode**, the search related method is virutal to be overloaded. The **SearchCallbackNode** can be applied on SearchPolicy object to do some extra processing for schedule search.
    * **sketch_policy.h/sketch_policy.cc**
        * class **SketchPolicy**:  The search policy that searches in a hierarchical search space defined by sketches. The policy randomly samples programs from the space defined by sketches and use evolutionary search to fine-tune them.  
            ```c++
            /*! \brief The cost model to estimate the complete schedules. */
            CostModel program_cost_model;
            /*! \brief The parameters map for this search policy. */
            Map<String, ObjectRef> params;
            /*! \brief The rules to generate sketches. */
            std::vector<SketchGenerationRule*> sketch_rules;
            /*! \brief The rules to generate initial population. */
            std::vector<PopulationGenerationRule*> init_rules;
            /*! \brief The rules to mutate states in the evolutionary search. */
            std::vector<std::shared_ptr<PopulationMutationRule>> mutation_rules;
            /*! \brief Random generator. */
            std::mt19937 rand_gen;
            /*! \brief Memorize split space for Split. */
            SplitFactorizationMemo split_memo;
            ```
        * Note the these methods: **GenerateSketches**, **SampleInitPopulation**, **EvolutionarySearch** and **PickStatesWithEpsGreedy**. Also, attention to SketchPolicy constructor, the **sketch rules for cpu and gpu differs**, the cpu & gpu & mali are specialized.
            ```c++
            // CPU task 
            node->sketch_rules.push_back(&rule_always_inline);
            node->sketch_rules.push_back(&rule_simplify_compute_with_const_tensor);
            node->sketch_rules.push_back(&rule_add_rfactor);
            node->sketch_rules.push_back(&rule_add_cache_write_stage);
            node->sketch_rules.push_back(&rule_multi_level_tiling_with_fusion);
            node->sketch_rules.push_back(&rule_multi_level_tiling);
            node->sketch_rules.push_back(&rule_skip_stage);
            // GPU(cuda)
            node->sketch_rules.push_back(&rule_add_cache_read_stage);
            node->sketch_rules.push_back(&rule_special_compute_location_gpu);
            node->sketch_rules.push_back(&rule_always_inline);
            node->sketch_rules.push_back(&rule_simplify_compute_with_const_tensor);
            node->sketch_rules.push_back(&rule_cross_thread_reduction);
            node->sketch_rules.push_back(&rule_add_cache_write_stage);
            node->sketch_rules.push_back(&rule_multi_level_tiling_with_fusion);
            node->sketch_rules.push_back(&rule_multi_level_tiling);
            node->sketch_rules.push_back(&rule_skip_stage);
            ```  
        * The details of the **sketch generation rules** and **init population rules** are defined in sketch_policy.cc. The base classes are **SketchGenerationRule** and **PopulationGenerationRule** respectively. 
            ```c++            
            /********** Sketch generation rules **********/
            static RuleSkipStage rule_skip_stage;
            static RuleAlwaysInline rule_always_inline;
            static RuleMultiLevelTiling rule_multi_level_tiling;
            static RuleMultiLevelTilingWithFusion rule_multi_level_tiling_with_fusion;
            static RuleAddCacheRead rule_add_cache_read_stage;
            static RuleAddCacheWrite rule_add_cache_write_stage;
            static RuleAddRfactor rule_add_rfactor;
            static RuleCrossThreadReduction rule_cross_thread_reduction;
            static RuleSimplifyComputeWithConstTensor rule_simplify_compute_with_const_tensor;
            static RuleSpecialComputeLocationGPU rule_special_compute_location_gpu;

            /********** Init population rules **********/
            static InitFillTileSize init_fill_tile_size;
            static InitChangeComputeLocation init_change_compute_location;
            static InitParallel init_parallel;
            static InitUnroll init_unroll;
            static InitVectorization init_vectorization;
            static InitThreadBind init_thread_bind;
            ``` 
        * The **Search** method of sketchpolicy show the workflow of how the ansor works. 
            * Train cost model(not first round)
            * SearchOneRound 
            * InferBound of the best_states and random_states 
            * PickStatesWithEpsGreedy(typically, the num equals to the num_measures_per_round), i.e., candidates selection 
            * Meaure the selected candidate states(loop state) 
            * continue to next round... 
        * For **SearchOneRound**: 
           * **GenerateSketches** 
           * **SampleInitPopulation** 
           * **EvolutionarySearch**, here insert the preivous measured good states to init_population controled by para: sample_init_use_measured_ratio.
        * **GenerateSketches**: generate the sketches based on the sketch rules of the search policy. 
           * start with the init_state <-> stage.size()-1(stage_id), push the state to currently working Array\<State\>
           * for the stage in array, try all sketch_rules, if **MeetCondition** ret is not skip cond, note the **stage_id** indicates the position of the rule apply at. 
           * Generally, the **stage_id** decreases when one rule applied on the state, but schedule primitive like cache_read/cache_write will add stage in the computeDAG, thus the stage_id remains. 
           * Also, some rules(inline, tiling) will change the loop state(CopyOnWrite way), thus state may changes during the process. 
           * The **order** in the sketch rules directly influences the sketch generation, since one rule can affect the condition checking for other rules. 
        * **GenerateSketches** based on the **SketchGenerationRule**, the two key methods are: **MeetCondition**, **Apply**.
           * the **Apply** method will do the primitives on the **State** and return the **State** with position id of the stage, the **State** schedule primitive do the **CopyOnWrite** to change the current State by add **transform_steps**, the added step contains the stage_id to indicate in which stage.
        * **SampleInitPopulation**: to generate sample_init_min_pop out_states
           * for the init_rules, **randomly** choose a **sketch**, and apply init_rule with **random factors**. 
           * Then filter the invalid generated states, call the **program_cost_model->Predict** to select candidate states to ret out_states.
        * The **PopulationGenerationRule** key is **Apply** method, typically, use **CopyOnWrite** way to rewrite the loop state, the goal is to generate **random factors** for each specialized init_rule.
        * **EvolutionarySearch**:  
            * Use **heap** to keep the best states in the search process, the compare obj is the score returned by the program_cose_model. 
            * For all mutation rules, based on rule_weights, **ComputePrefixSumProb** to get the prob for applying each mutation rule.
            * Start iterately search
            * First, prune the invalid state, use program_cost_model predict the init/prev_population states. Based on the init states and scores, construct the heap. If the heap is not empty, update the heap with previous searched states.
            * Also, **ComputePrefixSumProb** for population scores to get the pop_selection_prob (state).
            * **NodeCrossOver not supported now**
            * Do mutation, randomly choose the state, and then randomly select the mutation rule to apply rule on the state (if uniform_dist < mutation_prob). Until the population size reach the defined param.
            * Continue search.
            * Sort the heap, add state to best_states and ret. 
        * **PickStatesWithEpsGreedy**: for simply, when inputs < num_good(since will have some random states, **eps_greedy** param), pick best_states first, otherwise pick the random_states first. Then add the picked states(candidates) in **measured_states_set_** and **measured_states_vector_** for next round search won't re-pick again.
    * loop_state.h/loop_state.cc/transform_step.h/transform_step.cc
        * **StageNode** class: lightweight state in tvm/te/schedule, the members are listed as follows: 
            ```c++
            class StageNode : public Object {
            public:
            /*! \brief The operator of this stage */
            te::Operation op;
            /*! \brief The iterators in this stage. */
            Array<Iterator> iters;
            /*! \brief The type of this stage. */
            StageKind op_type;
            /*! \brief The compute location of this stage. */
            ComputeAtKind compute_at;
            /*! \brief Other stage-level attributes. */
            StageAttributes attrs;
            ```
        * **StageKey** -> int, i.e., stage_id to represent the stage;
        * **IterKey** -> pair\<int, int\>, i.e., stage_id & iter_id to represent a iterator;
        * **AttachMapNode**: stores the compute_at relation between stages
            ```c++
            /*! \brief A Map to store the mapping of stage to its attached iterator. */
            std::unordered_map<StageKey, IterKey> stage_to_attach_iter;
            /*! \brief A Map to store the mapping of iterator to the stages attached to it. */
            std::unordered_map<IterKey, std::vector<StageKey>, IterKeyHash> iter_to_attached_stages;
            ``` 
        * **StateNode** ---> the state in search process, consists of current loop structure and list of transformation steps, each state corresponds to a specific schedule for its ComputeDAG. The state similar to the schedule in tvm::te::Schedule.
            ```c++
            class StateNode : public Object {
            public:
            /*! \brief Current stages and loop structures. */
            Array<Stage> stages;
            /*! \brief History transformation steps. */
            Array<Step> transform_steps;
            /*!
            * \brief The attach relations of stages and iterators. This is used to track the compute at operation.
            */
            AttachMap attach_map;
            /*! \brief The up-to-date ComputeDAG of this state.
            * (e.g., CacheReadStep/CacheWriteStep) can modify the ComputeDAG.
            */
            Optional<ObjectRef> current_compute_dag;
            /*!
            * \brief Indicate whether this state has unfilled tile sizes. Only concrete state can be apply to TVM schedule.
            */
            bool concrete;
            ``` 
        * **State** ref support schedule primitives: bind, parallel, unroll, vectorize, fuse, pragma, reorder, split, storage_align. new two: follow_split, follow_fused_split, these two use split factors from previous steps; compute_at, compute_inline, compute_root; cache_read, cache_write, rfactor. Use stage_id to get the stage to be applied on.
        * **State** construct from the **Ops(tvm::te::Operation)**, the **stages** added to the StateNode is auto_schedule defined **Stage**.
        * The impl of the schedule primitives for the State:
            * General Process: (TODO) 
            * **bind, parallel, unroll, vectorize** almost same impl ---> **AnnotationStep**
                * 1). get the Stage via stage_id
                * 2). AnnotationStep with the right annotation_type
                * 3). CopyOnWrite to add the step in transform_steps 
                * 4). call the step->ApplyToState(this) to change the related stage in the State. For AnnotationStep, change the Iterator annotation.
            * **fuse**: -> FuseStep
            * **pragam**: -> PragmaStep
            * **reorder**: -> ReorderStep
            * **split**: -> SplitStep
            * **follow_split**: -> FollowSplitStep
            * **follow_fused_split**: -> FollowFusedSplitStep
            * **storage_align**: -> StorageAlignStep
            * **compute_at**: -> ComputeAtStep
            * **compute_inline**: -> ComputeInlineStep
            * **compute_root**: -> ComputeRootStep
            * **cache_read**: -> CacheReadStep
            * **cache_write**: -> CacheWriteStep
            * **rfactor**: -> RfactorStep
    * sketch_policy_rules.h/sketch_policy_rules.cc  
        * TODO 
    * TODO ---> as, Downcast, copy_on_write, operator->(), GetRef等
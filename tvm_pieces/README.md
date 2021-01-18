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

### *2021.1.19*
* ***The TIR structure***
    * 重新扫了下object, objectptr, objectref, 后续又看了下 tir/expr下的一些东西，对于Var产生兴趣，每个Var由其地址所唯一区分，之前被误导了淦，在Allocate, Let, For, LetStmt中是每个Var仅被bind一次，之前看成只在这几个中bind。。。
    * 结合llvm codegen看，这几个Node会维护一个 VarNode* -> llvm::Value* 的var_map表，对于Let, For, LetStmt的Var 基本是通过Make_value得到，Allocate则会调用到 llvm irbuilder 的 alloca 并convert成指针类型，其实这个过程就将抽象的Var 转为了具体的 数 或者 内存指针。
    * 在compute时候，会根据compute传入的参数进行解析，shape被解析为 vector<IterVar> axis 同时从中会获取到 类型为 vector<Var> 的args参数，而计算函数FCompute为函数模板对象，根据args调用，则会得到 类型为PrimExpr body，即内部的计算描述。  
    * 目前来看，tir这块理解最欠缺的是内存管理相关的IR，之后着重分析。

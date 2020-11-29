# Boring TVM Code

## *2020.11.17*
* use *parallel*, *vectorize* to create schedule which accelerate the execution time of operator.
* Use *split* for splitting the for loop, reorder can change the axis orders, quite like Halide.
* Identify the the bottleneck is memory bandwidth or computation. vector_add and broadcast_add are all memory-bound element-wise calculation.
* A good schedule needs to consider multiple performance-related factors together.

## *2020.11.18*
* for matrix multiplication, *reorder* the sum axis to avoid column based memory access
* choose the right tiling size to improve the cache effcience, *tile* primitive tile blocks, *fuse* can fuse two axes into one to be further parallelized.
* The non-continuous write issue is severer than the non-continuous read, so we can first write the results to a local buffer for each sub-matrix computation, and then write them back to the final matrix C.

## *2020.11.19*
* 可按照优化矩阵乘法的思路，对卷积操作中，宽和高的维度上进行*tile*操作，来提高cache利用率，但随着channel size的增大，性能会下降
* 仅对宽高维度做*tile*提升有限，采用相同思想对于channel维度也进行*tile*优化，需改变data layout

## *2020.11.24*
* depthwise conv和普通conv的优化思路相同，区别在于depthwise conv仅需tile单个维度的channel，因其输入输出channel相等同
* pooling这种 memory-bound 的算子，optimize的方式有限，可使用te.schedule.AutoInlineInjective(sch) 避免重复计算，类似于Halide的中inline的计算方式
  
## *2020.11.25*
* batch normalization 和 pooling的优化方法差不多，也是用AutoInlineInjective使计算全部inline，外层两axis做parallel，而内层使用vectorize优化
* *TOPI* (TVM Opeartor Invertory) provides numpy-style generic operations and schedules with higher abstractions than TVM. *TOPI* also provides higher level scheduling recipes depending on a given context to generate optimized kernel codes. *TOPI* also provides common neural nets operations such as _softmax_ with optimized schedule. 
* *TOPI* provide generic functions, different backends have different implementations to optimize the performance, thus should utilize the right backend implementation and schedule.

## *2020.11.26*
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

## *2020.11.28*
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

## *2020.11.29*
* Auto-tuner: 
    * 1). Define the search space: @autotvm.template, get config object: cfg = autotvm.get_config(), cfg.define_split, cfg.define_reorder etc. then apply the configEntity.
    * 2). Search through the space: RandomTuner, GridSearchTuner, GATuner, XGBTuner. *First*, create a tunning task. *Second*, define how to measure (autotvm.measure_option) the generated code and pick a tuner, build and run two steps. *Finally*, apply history best from teh cache file and check its correctness by autotvm.apply_history_best.
* Auto-scheduler:
    

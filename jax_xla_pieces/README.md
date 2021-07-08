# 简介

## JAX

JAX = Numpy on CPU/GPU/TPU + 自动微分 + XLA JIT Optimization，JAX为前端，使用XLA作为其后端编译器。

JAX不仅是Numpy on GPU，更多的是对于代码的转换，目的是获取更高效的数值计算代码。虽然其从接口层级上看起来和numpy别无二致，其真正的能力来自于函数变换的组合。主要核心变换方式为：1). jit，用于加速函数执行，通过tracing完成对于程序的变换；2). grad，用于反向求导；3). vmap 即vectorizing map，用于自动向量化/批量化。如下图所示，可以看出JAX主要针对数值计算，提供一个针对可组合函数变换的可扩展系统。

<div align="center">
<img src="https://github.com/hehesnail/Boring_code/blob/main/imgs/jax_position.png" width="50%" height="50%" /> 
</div>


JAX v.s. Numpy:
1. JAX提供了和numpy类似的接口;
2. 通过duck type，JAX array 和 numpy array可互相使用；
3. JAX array是不可修改对象，而numpy array可以修改。

JAX API Layering：
1. jax.numpy：提供类似numpy一样的接口，为高层次的包装器
2. jax.la: 更低层次的api，使用更加受限但同时更强大，对类型检查更为严格，不会自动进行类型转换；
3. 所有jax operations 通过 xla operations实现，通过xla使能JIT。

JAX 约束：
1. Pure function：JAX的变换以及编译被设计为均作用于pure function之上，要求函数无side-effect，否则其无法保证正确性；
2. In-Place Updates：JAX不支持in-place地修改变量，因这会使程序分析和变换变得困难；通过index_udpate, index_add等接口，将返回新的变量；
3. 越界索引： numpy中对于越界访问将报错，而jax中此为undefined-behaviour；
4. 非array输入：List，tuple等primitives不能被jax func的输入；
5. 控制流：
   * python control_flow + auto-diff 无约束；
   * python control_flow + jit：jit在 ShapeArray 的抽象层级上 trace python代码，此要求每个array value有固定shape以及数据类型，对python code更高层次的view提供了对于优化更多的机会，但同时限制越多；
   * 若控制流基于traced value，则会出错，故对于控制流条件变量需标记为static，而static args变动时将导致re-compilation；static args不变动，则每次可节省重复编译时间，若变动较小，代价可接受，若每次static args变动大，则带来很大的性能损耗；
   * 若想使control-flow被trace，同时又避免重复编译，则可使用jax中primitives，lax.cond (differentiable)，lax.while_loop (fwd-mode-differentiable)，lax.fori_loop (fwd-mode-differentiable)，lax.scan (differentiable)，这几个primitives应直接对应于xla中对应的ops；

JAX添加Primitive方法：

在jax tracing的过程中，其中primitve operations是可被其trace，因此可施加jit/grad/vmap等变换。而用户可添加自定义的primitive；
1. 首先在core.Primitive添加对应的operation primitive，后将对应函数绑定至此primitive；
2. 添加基本eval规则即对应numpy实现，通过def_impl添加；
3. 添加abstract eval规则，在使用jit等变换时，其抽象层级在ShapedArray层级，即不需要实际数据，因此需定义对应规则，主要是该primitve的shape，dtype规则，通过def_abstract_eval添加；
4. 在使用jit前，还需定义对应的xla编译规则，需使用xla_client添加对应xla op规则，通过xla.backend_specific_translations在指定backend中添加对应的xla编译规则；
5. 后续若还需使用 grad, jvp, vmap等变换，则需分别实现对应的 ad.primitive_jvps，ad.primitive_transposes 以及 batching.primitive_batchers等规则。

除了在Jax中可以添加自定义primitive外，也可添加对应自定义的函数变换规则即custom interpreter:
1. trace a function： 通过jax.make_jaxpr可trace一个function，并将其转换为 jaxpr
2. evalute a jaxpr： 对jaxpr变换的规则，可参考 core.eval_jaxpr，从jaxpr的表示出发，绑定对应输入参数及constants，后解析jaxpr中的equations，主要通过对primitive采用不同的解析方式以实现不同的功能。因jaxpr为pure function，且不存在副作用，因此整体就是逐步解析equations的过程。

## XLA

tensorflow的计算图中粒度非常小，一个常见的网络为存在大量的op相连，存在大量不必要的访存，因此需要对其进行优化，而XLA则完成了此功能。整体过程如下图所示：

<div align="center">
<img src="https://github.com/hehesnail/Boring_code/blob/main/imgs/tf_to_xla.png" width="100%" height="100%" /> 
</div>

XLA：Accelerated Linear Algebra Compiler，XLA目标：
1. 提高执行速度。编译子计算图以减少短暂运算的执行时间，从而消除 TensorFlow 运行时的开销；融合流水线运算以降低内存开销；并针对已知张量形状执行专门优化以支持更积极的常量传播。
2. 提高内存使用率。分析和安排内存使用，原则上需要消除许多中间存储缓冲区。
3. 降低对自定义运算的依赖。通过提高自动融合的低级运算的性能，使之达到手动融合的自定义运算的性能水平，从而消除对多种自定义运算的需求。
4. 减少移动资源占用量。通过提前编译子计算图并发出可以直接链接到其他应用的对象/头文件对，消除 TensorFlow 运行时。这样，移动推断的资源占用量可降低几个数量级。
5. 提高可移植性，方便为新硬件编写新后端。

XLA 的输入语言称为“HLO IR”或仅为“HLO”(High Level Optimizer)，可参考下图为典型HLO IR：
<div align="center">
<img src="https://github.com/hehesnail/Boring_code/blob/main/imgs/hlo_ir.png" width="60%" height="60%" /> 
</div>

XLA 接受在 HLO 中定义的计算图并将其编译为适用于各种架构的机器指令。XLA 采用模块化设计，可以轻松融入其他后端以针对某些新颖的硬件架构。TensorFlow 源代码树中包含适用于 x64 和 ARM64 架构的 CPU 后端，以及 NVIDIA GPU 后端。

<div align="center">
<img src="https://github.com/hehesnail/Boring_code/blob/main/imgs/xla.png" width="30%" height="30%" /> 
</div>

如上图所示，XLA 提供了多种与目标无关的优化和分析过程（例如 CSE）、与目标无关的运算融合，以及用于为计算分配运行时内存的缓冲区分析。完成与目标无关的步骤之后，XLA 会将 HLO 计算发送到后端。后端可以执行进一步的 HLO 级优化，而此时将考虑目标特定的信息和需求。最后针对特定目标生成代码。XLA 所含的 CPU 和 GPU 后端使用 LLVM 进行低级 IR、优化和代码生成。这些后端发出有效表示 XLA HLO 计算所需的 LLVM IR，然后调用 LLVM 以从此 LLVM IR 中发出原生代码。从提供的新增硬件的链接来看，当前主要基于LLVM来扩展新的硬件，非LLVM方式则工作量最大。

# 内部机制
## Basic workflow
JAX的主要流程如下图所示：
<div align="center">
<img src="https://github.com/hehesnail/Boring_code/blob/main/imgs/jax_life_cycle.png" width="60%" height="60%" /> 
</div>

### Tracer & Trace & Primitive

补图，类定义，Trace，Primitive交互方式
class Primitive
class Tracer
class Trace
class MainTrace
class TraceStack

def new_main, primitive defs -> bind

def bind -> find_top_trace, full_raise, process_primitive, full_lower

### Jaxpr
对于jvp，vmap变换而言，其实实现的对应Tracer需要很少的局部信息。而对于jit、vjp变换而言，对应即时编译和反向传播需要更加全面的上下文信息，而完成此功能的正是jaxpr，其为jax内部对于函数的中间表示，其为有类型，函数式的表示，对应syntax如下
```python
jaxpr ::=
  { lambda <binder> , ... .
    let <eqn>
        ...
    in ( <atom> , ... ) }
binder ::= <var>:<array_type>
var ::= a | b | c | ...
atom ::= <var> | <literal>
literal ::= <int32> | <int64> | <float32> | <float64>
eqn ::= <binder> , ... = <primitive> [ <params> ] <atom> , ...

jaxpr_type ::= [ <array_type> , ... ] -> [ <array_type> , ... ]
array_type ::= <dtype>[<shape>]
dtype ::= f32 | f64 | i32 | i64
shape ::= <int> , ...
```

Jax中的transformation将python function转换为了jaxpr IR，其为函数式语言，因此可作为函数变换的IR。。jax通过tracing生成对应的jaxpr，在tracing时，jax将函数每个参数包装为tracer object，这些tracer objects记录所有在其之上执行的jax operations，最后通过jax tracer objects重构整个输出即jaxpr。注意在trace的过程中，所有side effect操作将不会被记录，因此在生成的jaxpr不会记录对应的操作。获取到jaxpr后，不同的transformation依据不同的解释规则完成对代码的变换。

一个jaxpr对象表示具有单个或多个类型输入，单个或多个有类型输出的函数，通过调用jax.make_jaxpr可以看到对应python function的jaxpr形式如下所示：
```python
jaxpr ::= { lambda Var* ; Var+.
            let Eqn*
            in  [Expr+] }
Eqn  ::= let Var+ = Primitive [ Param* ] Expr+
Primitive := add | sub | sin | mul | ...
```

High-order primitives处理：
* Conditionals：控制流在jaxpr中不会被记录，在调用函数生成jaxpr的时候，因其基于trace的机制，对于python的控制流会正常执行，因此在最后生成的jaxpr中无需重复capture；但若使用lax.switch, lax.cond通过lax实现动态执行，jaxpr将其绑定为primitive cond进行处理，cond有两个输入即index和args。
* While：python loops在tracing的时候被默认inline，若想动态执行需要使用jax.lax.while_loop或jax.lax.fori_loop，此时jaxpr中对应为while primitive；
* Scan & xla call & xla pmap：同理对应为jaxpr对应primitives，分别对应为scan primitive, xla_call 以及xla_pmap，其中xla_call和xla_pmap同将jit编译的函数表示为call_jaxpr，同时包含backend，device等信息。

class Jaxpr, Var, Atom, Literal, Eqn defs
class DynamicJaxTracer, DynamicTrace, JaxprStackFrame
def make_jaxpr ---> Generate the jaxprs

## JIT -> jaxpr to xla
相对于transform来说，jit更类似于high-order primitive，即高阶原语，其输入为一个函数。jit采取逐步处理(staged processing)方法，bind对应的输入为jaxpr，而非python可调用对象

补图，jit调用图

jit -> _python_jit -> f_jitted -> xla_call

```python
xla_call_p: core.CallPrimitive = core.CallPrimitive('xla_call')
xla_call = xla_call_p.bind
xla_call_p.def_impl(_xla_call_impl)
```

这里xla_call为primitive即CallPrimitive，注意其bind函数中对应的primitive.impl目前对应了_xla_call_impl，在单独调用@jit的情况下，此时tracer stack中仅包含默认的EvalTracer，其通过调用bind调用对应的_xla_call_impl

_xla_callable ---> 其为xla转换的核心函数，先通过 trace_to_jaxpr_final 来trace python函数并获取到最终转换后的jaxpr，并通过xla_bridge模块中对应函数创建XlaBuilder，后使用jaxpr_subcomp将jaxpr转换为对应的HLO IR，此时对应的HLO指令将插在对应XlaBuilder中并通过Build创建HLO计算图。此时在xla中translation table中查询对应jaxpr.eqns中primitive对应的规则，对应规则在lax中注册，最终调用backend_compile通过对应平台的XLA Compiler编译对应HLO Graph，_xla_callable被cache装饰器装饰，以避免对同样的函数重复编译。

## xla_ops & xla_builder & compiler

<div align="center">
<img src="https://github.com/hehesnail/Boring_code/blob/main/imgs/xla_while.png" width="100%" height="100%" /> 
</div>
上图为xla中while op对应的流程，从左到右依次为xla op定义，XlaBuilder插入对应op的过程，以及对应CpuCompiler处理while的流程。这里选择while op进行说明，因其为control flow类语句，主要探明xla compiler如何处理控制流。可以看到对应XlaBuilder中对于While其输入的cond以及body类型需为XlaComputation，调用XlaBuilder build获取，因此其对应XlaComputation和此XlaBuilder不同。同时其会调用AddCalledComputation将其添加到embedded_中，注意RunBackend中首先会对embedded_中进行处理，因此在CpuCOmpiler IREmitter时处理while时，cond/body对应的llvm function已存在llvm module中。而HandleWhile，主要调用EmitGlobalCall分别调用condition/body对应的函数，同时while中判断语句生成，则是通过load condition func call对应的返回值，并创建比较语句ICmpNE获取。

# 总结
## 表示能力
## 技术路线

# References

XLA Operations： [xla ops](https://www.tensorflow.org/xla/operation_semantics)，各个Ops语义描述，输入输出定义；

XLABuilder定义：[xla builder](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/xla/client/xla_builder.h)，主要通过XLABuilder添加对应HLO指令，xla_op以及xla_computation定义；

XLA 新增硬件方法：[xla new hardware](https://tensorflow.google.cn/xla/developing_new_backend?hl=zh-cn)，目前支持cpu/gpu均通过LLVM完成；

XLA python端接口：[xla python interface](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/xla/python/xla_client.py)，这里 xla python interface通过python module xla_extension引入，xla_extension是通过pybind11调用cxx端实现；

JAX XLA：[lax](https://github.com/google/jax/blob/master/jax/lax.py)，
[xla_bridge](https://github.com/google/jax/blob/master/jax/lib/xla_bridge.py)，
[xla_exec](https://github.com/google/jax/blob/master/jax/interpreters/xla.py)

JAX Docs: [jax_doc](https://jax.readthedocs.io/en/latest/index.html)
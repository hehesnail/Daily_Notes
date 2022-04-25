# Pytorch-torchscript

随着深度学习技术的发展，出现了许多框架如caffe, tensorflow, pytorch等等。而其近几年，pytorch因其pythonic的编程模式，eager execution，及动态图等特性在编写模型和训练上提供了极大的灵活性，也因此逐渐在深度学习框架之争中后来居上，同tensorflow分庭抗礼以至有反超之势。根据No free lunch，其灵活性必然是建立在牺牲性能的基础上。如下图所示，在部署层面，pytorch表现的并不是十分理想。
在获得大量用户基础后，TorchScript和其对应JIT Compiler被 pytorch开发者提出以解决其在部署上的劣势。而其值得注意的一点是pytorch是如何将如此灵活的表示(大量control flow及python primitives) 结构化成易于推理的表示(IR)，以及在此基础上进行进一步优化。
本节首先对TorchScript进行简要的介绍，后续着重分析了torchscript中各阶段IR的转换，最终简要地总结了torchscript为支持灵活python表示所做出的支持。

<div align="center">
<img src="https://github.com/hehesnail/Boring_code/blob/main/imgs/pytorch_capability.png" width="60%" height="60%" /> 
</div>

## Torchscript简介

### 动机

为什么需要TorchScript？

1. TorchScript代码可以在自己的解释器中调用，它基本上是一个受限的Python解释器。此解释器不获取全局解释器锁，因此可以在同一实例上同时处理多个请求；
2. TorchScript格式可以将整个模型保存到磁盘，并将其加载到另一个环境中，比如用Python以外的语言编写的服务器中；
3. TorchScript提供了新的表示IR，在此IR中，可以对代码进行编译器优化，以提供更高效的执行；
4. TorchScript可以同许多后端/设备运行时进行交互。

### TorchScript IR及JIT Compiler

TorchScript IR形式为图级别IR，并且IR具有SSA(静态单赋值)性质。其IR由Aten(Pytorch C++后端) Operator以及其他内置运算符如循环，条件分支等。
下图为TorchScript IR形态的例子，可看到nn.Module类被转换为了torchscript表示的Graph，Graph由Node及Block构成，而其中If语句被转换为两个block，其形式类似于LLVM IR中If表示形式，其SSA的特性决定在Block输出合并时应会插入Phi Node。其中组成元素prim对应内置类型，aten则为对应pytorch aten算子库，其在后续lower过程中会转换为函数调用。


<div align="center">
<img src="https://github.com/hehesnail/Boring_code/blob/main/imgs/torchscript_ir_eg.png" width="60%" height="60%" /> 
</div>

JIT Compiler: JIT Compiler为TorchScript对应的即时编译器(Just-in-Time)，其会将python源码编译为对应的Torchscript IR，并可加载进入解释器执行。

使用方式：

Pytorch提供两种方式完成TorchScript IR的转换，Trace以及Script，当然两种方式可混合使用:
1. torch.jit.trace: 此模式下基于输入，trace顾名思义就是运行代码，记录运行过程中发生的操作，并基于此构造ScriptModule。Trace模式的优点在于：其仅trace实际运行的操作，并将对应的path转换为Script IR同时编译优化，编译开销相对较小。缺点在于：trace到的操作取决于输入，因此对于控制流无法完整获取；对于inplace的操作如tensor view，其无法trace到。
2. torch.jit.script: 直接将python模块转换为TorchScript IR, 可确保对于控制流等结构完整的获取，不取决于外部输入。其规则为：Parameters(self.weight) -> 保留；Submodules(self.layer1) -> 对于子模块会递归进行转换；Methods -> 转换为torchscript，从最上层 forward method开始，递归地转换；Model structure -> 保留，包括其中函数调用，类，以及控制流语句；

另外需要注意的一点是，ScriptModule继承自nn.Module，其也可以在训练时使用。

TorchScript优化:

1. 典型编译器优化: 冗余代码消除(DCE), 共同子表达式消除(CSE), 循环展开(loop unrolling), 常量传播(const propagation).
2. 张量优化: Algebraic层级窥孔优化, 多batch矩阵乘优化, element-wise 运算融合；
3. 运行时优化: 无全局解释器锁, fork/wait parallelism；

同时torchscript提供算子间并行性，可通过调用torch.jit._fork 以及 torch.jit._wait接口完成。

## TorchScript IR变换

### 整体流程

<div align="center">
<img src="https://github.com/hehesnail/Boring_code/blob/main/imgs/torchscript_workflow.png" width="60%" height="60%" /> 
</div>

上图展示了python code转换为TorchScript IR，最终被加载进入解释器进行执行的流程。总体上说，当调用torch.jit.script后，其会创建对应AST，并将AST转换为对应的图层级IR，创建Graph Executor并加载图层级IR，转换为对应的解释器指令并执行。其中编译优化的部分集中在Graph Executor部分，其会根据当前Graph信息，完成相应的优化后，翻译为解释器指令并执行。下面为同一段代码在不同阶段的IR表示形态，分别对应source code，jit ast以及 IR Graphd的形态。

如下代码片段为对应示例python代码：
```python
@torch.jit.script
def foo(len):
    # type: (int) -> torch.Tensor
    rv = torch.zeros(3, 4)
    for i in range(len):
        if i < 10:
            rv = rv - 1.0
        else:
            rv = rv + 1.0
    return rv
```
如下代码片段对应为示例pytho代码经解析得到的AST形态：
```c++
// JIT AST
(def
(ident foo)
(decl
    (list
    (param
        (ident len)
        ....
(list
    (assign
    (list (variable (ident rv)))
    ...
    (for
    (list (variable (ident i)))
    ...
    (list
        (if
        (<
            (variable (ident i))
            (const 10))
        ...
        (list
            (assign
            ...
    (return (variable (ident rv)))))
```
上述AST中节点编译后可得到对应IR Graph的表示，形式上为SSA形式IR，其中针对python List, if, while等处理为新增对应prim::ListConstruct、prim::If、prim::Loop等，而原有支持的Op则会派发对应aten中op，如aten::sub, aten::add等。基于IR Graph，torchscript进行一系列优化后将其编译为对应解释器指令执行。
```c++
// IR Graph
graph(%len.1 : int):
%24 : int = prim::Constant[value=1]()
%17 : bool = prim::Constant[value=1]() # test.py:10:5
%12 : bool? = prim::Constant()
%10 : Device? = prim::Constant()
%6 : int? = prim::Constant()
%1 : int = prim::Constant[value=3]() # test.py:9:22
%2 : int = prim::Constant[value=4]() # test.py:9:25
%20 : int = prim::Constant[value=10]() # test.py:11:16
%23 : float = prim::Constant[value=1]() # test.py:12:23
%4 : int[] = prim::ListConstruct(%1, %2)
%rv.1 : Tensor = aten::zeros(%4, %6, %6, %10, %12) # test.py:9:10
%rv : Tensor = prim::Loop(%len.1, %17, %rv.1) # test.py:10:5
    block0(%i.1 : int, %rv.14 : Tensor):
    %21 : bool = aten::lt(%i.1, %20) # test.py:11:12
    %rv.13 : Tensor = prim::If(%21) # test.py:11:9
        block0():
        %rv.3 : Tensor = aten::sub(%rv.14, %23, %24) # test.py:12:18
        -> (%rv.3)
        block1():
        %rv.6 : Tensor = aten::add(%rv.14, %23, %24) # test.py:14:18
        -> (%rv.6)
    -> (%17, %rv.13)
return (%rv)
```

### JIT AST

Tree & TreeRef:表示树节点，同时进行前后类型检查, Compound 继承自Tree, 在AST中节点大部分为此类型，其相比父类增加 SourceRange, TreeList数据成员。

TreeView: AST中Node父类，包括 TreeRef data member, 同时提供了基本 methods。

Stmt & Expr继承自TreeView, 而其他语句(If, For, While...)继承自Stmt，Expr同理。其上下文无关文法(CFG)定义在Tree_view.h中，阐明了AST语法规则。 

构建AST存在两种方式：
1. Python端, 存在StmtBuilder & ExprBuilder类, 其首先对python代码调用ast模块进行分析，后递归地将其转换为AST，此时会通过pybind调用C++端方法。
2. C++端,对于传入的soure code (python)，调用 Lexer 及Parser创建JIT AST

### IR Graph

IR Graph主要由Graph & Node & Value & Block构成, 定义于ir.h/ir.cpp中

Graph: 为计算的抽象，常用于表示函数层级，其由Node串联组成；

Node: IR Graph 基础类，表示计算以及其依赖的Values, 其可包含多个Blocks（用来定义嵌套control flow, For/If）。注意 Graph 的连接关系是通过 Node 串联起来， Graph中插入新节点，实际作用在Node为元素的双向链表(拓扑排序)上，可通过 next / prev指针获取 Node；

Value: 用以表示 node 的输入输出，为Tensor类型或者模糊类型的指针；

Block: 由Node组成的列表,同时包含对应输入输出。

结合后面 AST to IR的转换来看，IR Graph 更类似于提供了AST Node 中对应到的图中编排方式(重新组织)，分析AST时以拓扑排序组织，以prim::xxx, aten::xxx为粒度。 通过以图的方式组织，结合Node多为 Operator的性质，因此可做基于图层级的优化, 如bmm, fused_op。 同时，因其会转换为SSA形式，基于此做更多传统编译器优化。 

### JIT AST变换为IR Graph

<div align="center">
<img src="https://github.com/hehesnail/Boring_code/blob/main/imgs/torchscript_to_ir.png" width="60%" height="60%" /> 
</div>


JIT AST转换为IR Graph的功能由ir_emitter.cpp中to_ir函数完成，上图为to_ir函数中整体流程，其中emitDef函数调用完成了JIT AST向IR Graph转换的过程，对应为AST文法定义中Def为root节点，其主要调用emitExpr及emitStatements完成IR Graph生成，转换过程中相关函数依据节点类型递归调用。

**emitDef**: 一般来说AST的根节点为Def class, 主要根据 AST IR 的定义递归处理， IR转换过程类似 Codegen，其内部则会调用emitStatements/emitExpr转换语句和表达式。

**emitStatements**: 根据 stmt类型, 调用对应的 emitXXX函数。emit stmt时，一般会在graph中添加对应的Node(topological order)。 比如，对于If语句，调用emitIf时先分别调用emitCondExpr和emitIfElseBlocks, 后在graph中添加对应 Node(prim::If), 同时对于If Node来说，其添加了两个Block (true block 和 false block)。

**emitExpr**: 主要根据AST对应表达式类型，调用 emitSugaredExpr, emitApplyExpr, emitSubscript等。

**emitSimpleExpr**。对于Exp而言，其返回类型为 Value*，即作为Stmt (Node)的输入输出。
总体而言，其根据 AST Node定义，递归处理，而在此每个Node对应的处理中， stmt会在 graph中添加对应的Node(以kind区分)，expr则作为stmt的输入输出，被预先处理。

### IR Graph变换为GraphExecutor

经过上一步得到IR Graph后，对Graph进行分析(此时节点按顺序排列)，此时构建图执行器，其针对Graph 中Node的类型，会将其翻译成Graph Executor中对应解释器的指令，并进行执行。此时，对于解释器而言，并不需要设计十分复杂的指令集，因IR Graph中大部分Node为对应Aten算子调用(包括+/-/*等基本运算符)，因此需处理的集合也仅为Prim对应的子集。

## 小结

### python支持能力

<div align="center">
<img src="https://github.com/hehesnail/Boring_code/blob/main/imgs/torchscript_sup_python.png" width="100%" height="100%" /> 
</div>


上图展示了目前torchscript支持的python语法，可以看出已经相当丰富，包括List/ Tuple /Dict等，还有灵活的控制流语句(while/if/for/break/continue/return)等，也支持灵活的函数调用，同时其也针对nn.Module类的相关成员变量和成员函数进行支持。上述仅为大致总结，关于详细约束可参考官方language ref。

### 技术路线

TorchScript目的在于支持的 python语法为和描述神经网络相关的语法，亦可看成同 nn.Module强相关的python语法，从 torchscript.jit.script 直接作用于 nn.Module -> ScriptModule, function -> ScriptFunction主要为 nn.Module服务。但从目前支持的语法来看，其已具有相当高的灵活性，逻辑跳转，循环，List，Dict，Tuple等均支持。支持如此多的 python语法 以及 已有pytorch特性，建立在：

1. 综合需支持语法，专为 JIT 设计的 Lexer, Parser, CFG, AST；
2. 设计IR Graph 用以组织AST的节点，且该Graph IR 易于优化且易于转换为解释器指令；
3. 综合IR Graph 及 图中节点已有的 Kind, 设计了GraphExecutor 即图执行器其中包含优化Passes 以及Interpreter， 而Interpreter中设计了对应的 Instruction 以及不同OpCode的对应行为，同时解释器中支持插入Profile节点，并以此运行结果来引导进一步优化；
4. Python -> JIT AST -> IR Graph -> Optim Passes -> Interpreter Instruction -> Interpreter Exec；针对上述不同层级的路径，支持的其间的转换以及类型匹配；
5. pytorch原生强大的operator库 (ATen), 因此相当多的对于tensor的操作可通过调用operator的方式调用 ATen 中对应的 operator实现包括cpu, gpu。无论从 pass还是解释器的角度, 这种方式可以通过比较粗粒度方式的处理，如 +,-,*,/等运算符不用设计对应的 OpCode, 通过operator调用方式；
6.  JIT优化不用lowered 到更底层的目标平台相关的IR描述，比如从IR Graph 层级lower 到 LLVM IR，这种方式降低了分析Op内部的复杂性。

这里我们以List为例，简明地描述上述不同层级间的转换：

1. 在python阶段调用build_List创建 ListLiteral Object，其对应在JIT AST中ListLiteral节点(tree_views.h), 类型为TK_LIST_LITERAL(kind)；
2. 转换为IR Graph时, emitSimpleExpr 中依据 TK_LIST_LITERAL类型处理调用 createList 并在 graph 中添加类型为prim::ListConstruct的Node;
3. Interpreter生成对应对应指令时，调用emitNode, 依据prim::ListConstruct 调用emitContainerConstruct, 此时 OpCode 为 LIST_CONSTRUCT;
4. Interpreter 执行时根据 LIST_CONSTRUCT OpCode, call listConstruct, 其会生成 c10::List 实例化对象并push入栈中。(注：List 和 Stack 中元素类型为 IValue, 此为类型无关)。

### 演进方向

<div align="center">
<img src="https://github.com/hehesnail/Boring_code/blob/main/imgs/torchscript_plan.png" width="60%" height="60%" /> 
</div>

上图为对应的torchscript提出的roadmap，可以看出其之后主要发力点为：
1. 在torchscript中提供对于模型量化的支持，其思路可能类似TVM量化方言；
2. 更加轻量级的解释器，增强torchscript在移动端的竞争力；
3. 支持更多的后端，而其这里对于后端的定义则是graph compilers，比如TVM, XLA等。从目前JIT相关代码的提交中可以发现，TorchScript内部也新增了TensorExpr(NNC)，其中NNC代表神经网络编译器，其IR形态大幅基于TVM IR/ Halide IR。因此可看出，其在之后很大可能会采用类似schedule/auto-schedule的调度实现。


# JAX-XLA

## JAX简介

JAX = Numpy on CPU/GPU/TPU + 自动微分 + XLA JIT Optimization，JAX为前端，使用XLA作为其后端编译器。

JAX不仅是Numpy on GPU，更多的是对于代码的转换，目的是获取更高效的数值计算代码。虽然其从接口层级上看起来和numpy别无二致，其真正的能力来自于函数变换的组合。主要核心变换方式为：1). jit，用于加速函数执行，通过tracing完成对于程序的变换；2). grad，用于反向求导；3). vmap 即vectorizing map，用于自动向量化/批量化。如下图所示，可以看出JAX主要针对数值计算，提供一个针对可组合函数变换的可扩展系统。

<div align="center">
<img src="https://github.com/hehesnail/Boring_code/blob/main/imgs/jax_position.png" width="60%" height="60%" /> 
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

## XLA简介

tensorflow的计算图中粒度非常小，一个常见的网络为存在大量的op相连，存在大量不必要的访存，因此需要对其进行优化，而XLA则完成了此功能。整体过程如下图所示：

<div align="center">
<img src="https://github.com/hehesnail/Boring_code/blob/main/imgs/tf_to_xla.png" width="60%" height="60%" /> 
</div>

XLA：Accelerated Linear Algebra Compiler，XLA目标：
1. 提高执行速度。编译子计算图以减少短暂运算的执行时间，从而消除 TensorFlow 运行时的开销；融合流水线运算以降低内存开销；并针对已知张量形状执行专门优化以支持更积极的常量传播。
2. 提高内存使用率。分析和安排内存使用，原则上需要消除许多中间存储缓冲区。
3. 降低对自定义运算的依赖。通过提高自动融合的低级运算的性能，使之达到手动融合的自定义运算的性能水平，从而消除对多种自定义运算的需求。
4. 减少移动资源占用量。通过提前编译子计算图并发出可以直接链接到其他应用的对象/头文件对，消除 TensorFlow 运行时。这样，移动推断的资源占用量可降低几个数量级。
5. 提高可移植性，方便为新硬件编写新后端。

XLA 的输入语言称为“HLO IR”或仅为“HLO”(High Level Optimizer)，XLA 接受在 HLO 中定义的计算图并将其编译为适用于各种架构的机器指令。XLA 采用模块化设计，可以轻松融入其他后端以针对某些新颖的硬件架构。TensorFlow 源代码树中包含适用于 x64 和 ARM64 架构的 CPU 后端，以及 NVIDIA GPU 后端。

<div align="center">
<img src="https://github.com/hehesnail/Boring_code/blob/main/imgs/xla.png" width="60%" height="60%" /> 
</div>

如上图所示，XLA 提供了多种与目标无关的优化和分析过程（例如 CSE）、与目标无关的运算融合，以及用于为计算分配运行时内存的缓冲区分析。完成与目标无关的步骤之后，XLA 会将 HLO 计算发送到后端。后端可以执行进一步的 HLO 级优化，而此时将考虑目标特定的信息和需求。最后针对特定目标生成代码。XLA 所含的 CPU 和 GPU 后端使用 LLVM 进行低级 IR、优化和代码生成。这些后端发出有效表示 XLA HLO 计算所需的 LLVM IR，然后调用 LLVM 以从此 LLVM IR 中发出原生代码。从提供的新增硬件的链接来看，当前主要基于LLVM来扩展新的硬件，非LLVM方式则工作量最大。

## 内部机制

### Basic workflow

JAX的主要流程如下图所示：

<div align="center">
<img src="https://github.com/hehesnail/Boring_code/blob/main/imgs/jax_life_cycle.png" width="60%" height="60%" /> 
</div>

上图为JAX对应的整体流程，可以看出其主要机制为trace + transform，通过trace获取到python函数中对应的primitive，依据当前trace的类型完成对于函数变换。若为JIT变换，则首先通过trace转换为Jaxpr中间表示后，compile对应Jaxpr表示完成xla op的变换，从而利用XLA完成JIT优化后返回对应python函数进行执行。其中jvp，vmap变换可直接通过trace实现，而对应jit，vjp等变换需首先trace变换为Jaxpr，后对Jaxpr进行变换。

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

### JIT -> jaxpr to xla

相对于transform来说，jit更类似于high-order primitive，即高阶原语，其输入为一个函数。jit采取逐步处理(staged processing)方法，bind对应的输入为jaxpr，而非python可调用对象。

<div align="center">
<img src="https://github.com/hehesnail/Boring_code/blob/main/imgs/jax_jit.png" width="60%" height="60%" /> 
</div>

上图为JAX jit对应的实现流程，jit通过装饰器实现，jax中存在_cpp_jit以及_python_jit，其中_cpp_jit直接调用xla中c++ 端jit完成对于jax中trace函数的转换，目前处于实验状态，功能不稳定。因此主要还是采用_python_jit在python端完成xla ops的转换及编译。

```python
xla_call_p: core.CallPrimitive = core.CallPrimitive('xla_call')
xla_call = xla_call_p.bind
xla_call_p.def_impl(_xla_call_impl)
```

这里xla_call_p为primitive即CallPrimitive，注意其bind函数中对应的primitive.impl目前对应了_xla_call_impl，在单独调用@jit的情况下，此时tracer stack中仅包含默认的EvalTracer，其通过调用bind调用对应的_xla_call_impl。

**_xla_callable**: 其为xla转换的核心函数
1. 先通过trace_to_jaxpr_final通过调用DynamicJaxprTrace将python函数并获取到最终转换后的jaxpr；
2. 并通过xla_bridge模块中对应函数创建XlaBuilder；
3. 后使用jaxpr_subcomp将jaxpr转换为对应的HLO IR，此时对应的HLO指令将插在对应XlaBuilder中并通过Build创建HLO计算图。此时在xla中translation table中查询对应jaxpr.eqns中primitive对应的规则，对应规则在lax中注册；
4. 最终调用backend_compile通过对应平台的XLA Compiler编译对应HLO Graph，_xla_callable被cache装饰器装饰，以避免对同样的函数重复编译。

通过上述过程JAX完成了将python函数转换为XLA Computation Graph的过程，其中backend_compile调用对应平台compiler完成JIT编译后返回对应executable。

### Complie XLA to Executable

<div align="center">
<img src="https://github.com/hehesnail/Boring_code/blob/main/imgs/run_backend.png" width="60%" height="60%" /> 
</div>

上图为RunBackend对应流程，以CpuCompiler为例，主要流程对应加载MLIR方言，创建LLVM JIT对象，后针对当前HLOMoudle中HLOInstructions进行调度排序以降低内存开销，同时根据调度结果进行Buffer分配，之后针对所有的子函数即没有fuse掉的op形成的子图对应的XlaComputation生成对应的LLVM Function并添加进入emitted_functions之中。后续生成entry function对应的函数体，也通过EmitComputation函数完成，而在生成entry function函数体时候会在emitted_functions中查找对应函数，并完成对应函数调用生成。

### While Op

下图为伪码对应while graph形式的示例，condition和body1分别对应到图中子图以及XlaComputation对象，init为Tuple Op封装的标量及向量其为while op的operand亦即输入，作为condition, body的初始输入即parameter。

<div align="center">
<img src="https://github.com/hehesnail/Boring_code/blob/main/imgs/while_graph.png" width="60%" height="60%" /> 
</div>

下图为xla中while op对应的流程，从左到右依次为xla op定义，XlaBuilder插入对应op的过程，以及对应CpuCompiler处理while的流程。这里选择while op进行说明，因其为control flow类语句，主要探明xla compiler如何处理控制流。可以看到对应XlaBuilder中对于While其输入的cond以及body类型需为XlaComputation，调用XlaBuilder build获取，因此其对应XlaComputation和此XlaBuilder不同。同时其会调用AddCalledComputation将其添加到embedded_中，注意RunBackend中首先会对embedded_中进行处理，因此在CpuCOmpiler IREmitter时处理while时，cond/body对应的llvm function已存在llvm module中。而HandleWhile，主要调用EmitGlobalCall分别调用condition/body对应的函数，同时while中判断语句生成，则是通过load condition func call对应的返回值，并创建比较语句ICmpNE获取。

<div align="center">
<img src="https://github.com/hehesnail/Boring_code/blob/main/imgs/xla_while.png" width="100%" height="100%" /> 
</div>

### Tuple & GetTupleElement

<div align="center">
<img src="https://github.com/hehesnail/Boring_code/blob/main/imgs/xla_tuple.png" width="60%" height="60%" /> 
</div>

如上图所示为XLA中Tuple Op以及GetTupleElement Op对应LLVM IR的生成过程，XLA基于此解决了在计算图中Op存在多输入多输出的问题。对于Tuple Op而言，其主要通过对应HandleTuple函数对HLO Instr进行处理，首先为对应Tuple分配对应内存，后获取操作数Value，其将多个操作数打包为Tuple类型，而在EmitTuple中，将每个操作数Store进入对应分配好的内存中。

与此相对应的GetTupleElementOp获取Tuple中对应index的值，其首先通过GetEmittedValue获取到对应Tuple的地址，并在EmitGetTupleElement中，依照index的值，创建对应Load IR。

### Slice Op

<div align="center">
<img src="https://github.com/hehesnail/Boring_code/blob/main/imgs/xla_slice.png" width="60%" height="60%" /> 
</div>

上图为Slice Op LLVM IR生成大致流程，在HandleSlice中首先调用EmitTargetAddrForOp为Slice Op分配空间，后续构建For循环，并依据Shape大小添加对应循环轴，后续针对Slice以及Operand操作数分别生成对应target_array和source_array地址，通过调用EmitTransferElments完成元素的拷贝。在此可以看出，对于Slice操作，XLA选择下沉到LLVM层级进行解决，而方式也是类似的生成循环，并在循环体内部完成元素的拷贝。

### Dot Op

<div align="center">
<img src="https://github.com/hehesnail/Boring_code/blob/main/imgs/xla_dot.png" width="60%" height="60%" /> 
</div>

如上图所示为XLA Compiler对于Dot Op的处理，根据配置不同其会调用不同函数生成对应的LLVM IR，其中EmitNaiveLlvmIrGemm，EmitTiledLlvmIrGemv, EmitTiledLlvmIrGemm等均直接通过LLVM IRBuilder生成对应的IR语句，而EmitCallToRuntime则通过调用外部函数如MKL等。这里值得注意的为EmitLinalgMatmul函数，该函数产生MLIR函数，并生成对此函数的调用，而在MLIR函数生成过程中，其复用了Affine，Linalg，Vector等方言中的优化配置并最终生成对应的函数。这里也可以看到在XLA中，MLIR方言体系已得到实际应用。

## 小结
### 表示能力
可参考下图典型HLO IR分类：

<div align="center">
<img src="https://github.com/hehesnail/Boring_code/blob/main/imgs/hlo_ir.png" width="60%" height="60%" /> 
</div>

通过上图部分HLO IR分类并综合其他IR可以看出，XLA是完全以表示神经网络为中心，并基于此设计了HLO IR，其中Op基本涵盖Tensorflow中Op。其本身并不关心对于python代码直接转译得到XLA Op的过程，JAX通过Trace机制完成了部分功能，但对于控制流等无法完整转换。

从分类中可以看出，其类型较为丰富，值得注意的为控制流类型Op以及数据重组织类型Op。其中控制流类型Op中直接支持While，Conditional Op，亦即在XLA中有表示控制流的能力。而Reshape，Broadcast，Slice等Op XLA直接在LLVM层级予以支持，因此其前端可较为灵活地支持语法糖，如类似numpy slice之类灵活的描述。因此，HLO IR的表示能力上是相对丰富的。

### 技术路线

<div align="center">
<img src="https://github.com/hehesnail/Boring_code/blob/main/imgs/mlir_dialect.png" width="60%" height="60%" /> 
</div>

MLIR作为Google发起的开源项目，隶属于整个LLVM社区，其目的在于复用不同层级IR的优化能力。上图所示为当前MLIR中已存在方言，其中TF为Tensorflow dialect, HLO、MHLO、LMHLO为HLO对应不同层级dialect。从中可以看出，tensorflow已逐步地完成向MLIR体系中的迁移，而从其中Dot Op emit的过程看出其已支持对加载部分方言并复用其优化能力从而生成高效代码。XLA设计出发点主要针对图层级的优化，而缺乏多层级优化能力的支持，将其接入整个MLIR方言体系可复用各个方言的优化能力，从而获取更大的性能受益，这也为Google未来的技术路线。

# TVM Relay

## Relay简介

Relay IR是TVM社区用于描述神经网络的IR，其前身为NVVM，主要负责模型层的表示以及图层级优化，在后续Lower及Codegen过程中会转换到TIR层级的对应实现。

Relay IR是一种面向张量的静态类型的函数式IR，其设计源于Meta Language家族中语言所使用的函数式IR可以较为容
易地调整并支深度学习领域的描述。通过其可表达的语义，包括控制流、数据结构和一级函数，Relay IR可以当前最新的深度学习模型。

深度学习框架中的常见特性，如量化和shape推断，可以在编译器过程中重新定义对应的方言如QNN Dialect。通过使用这种重构，可以利用传统编译器的优化手段进行优化。与此同时，Relay IR是与平台无关的运算符表示和基于特定域的优化，通过TVM后端的支持性，可协同完成模型层面以及算子层面的优化。

本小节将先对Relay IR类型进行描述，后针对其编译流程，Op选择机制进行进一步分析。

## IR Infra

关于Relay IR的定义在relay/ir/expr.cc, ir/expr.cc中。

基础 Expr类型为 tvm::RelayExpr，继承自BaseExpr，同 PrimExpr平级，主要区别在于RelayExpr的类型及类型推断方式，此点在论文中提过。其class定义如下：
```c++
class RelayExprNode : public BaseExprNode {
    public:
    /*!
    * \brief Stores the result of type inference(type checking).
    * \note This can be undefined before type inference.
    *       This value is discarded during serialization.
    */
    mutable Type checked_type_ = Type(nullptr);
    /*!
    * \return The checked_type
    */
    inline const Type& checked_type() const;
    ...
}
```
注意其中的Type以及checked_type用于类型推断，而PrimExpr中仅为runtime是DataType就足够，因其针对POD Value。
而函数类型的父类，BaseFunc则继承自RelayExpr，这是因为函数类型并不是POD Value，其为Relay类型系统中新增加的类型，需继承子RelayExpr以提供类型检查。后续 RelayFunc和PrimFunc均继承子BaseFunc。

Relay FunctionNode ---> BaseFuncNode ---> RealyExprNode; 其中RelayExprNode的描述，BaseNode of all-non primitive expressions, 而依照Paper中说法，DL type system一个特殊的点在于要将 function call等操作看作 primitive 同等级的，RelayExprNode支持 tensor types, functions, ADT作为第一等公民;

PrimExprNode 中 DataType 同 RelayExprNode 中 tvm::Type的区别; PrimExpr的数据类型匹配是通过runtime时候进行check，这对于 POD Value是足够的; 对于 Tensor Type等类型，需要借助其文中提到的type system进行 type inference; 因此，BaseFunc 继承 RelayExpr， PrimFunc 以及 RelayFunc继承自BaseFunc;

整体的继承关系图如下：

<div align="center">
<img src="https://github.com/hehesnail/Boring_code/blob/main/imgs/relay_ir_infra.png" width="60%" height="60%" /> 
</div>

**ConstantNode**: constant tensor，内部有runtime::NDArray类型数据成员data，当data->ndim为0时，即rank-0 tensor其对应常量;

**TupleNode**： relay为支持多输出引入类型，成员变量 tvm::Array<relay::Expr> fields;

**VarNode**：变量类型，用于let expr中，其语义类似于tir中tvm.Var;每个Var仅被bind一次且imutable，成员 Id vid 及 Type type_annotation;

**CallNode**: relay中为支持神经网络中op调用引入的类型，在计算图中即为op，其成员变量： 1). Expr op，这个op可以是tvm::Op(primitive opeartors)，也可为Function(fuse_op后均为function)等; 2). tvm::Array<relay::Expr> args，输入参数; 3). Attrs attrs，额外attributes; 4). tvm::Array::<Type> type_args;

**LetNode**: Let可将程序变换为 A-normal形式，其中每个表达式均对应一个let binding，每个let可被看作计算图中一个 operator node， 其对应 Var var即待绑定变量，Expr value绑定变量值，Expr body;

**IfNode**: 条件表达式，跟TIR中If不同之处在于，if会求值为对应分支的结果，更类似与 c 中三元表达式;其成员变量 Expr cond, Expr true_branch, Expr false_branch;

**TupleGetItemNode**: 从tuple类型中获取指定index值， Expr tuple，int index;

**RefCreateNode**：创建对于初始值的引用，其成成员变量为 Expr value(initial_value of ref);

**RefReadNode**: 获取Ref中值，Expr ref;

**RefWriteNode**: Set value of ref, whole expression evaluates to an Empty Tuple; 成员变量为 Expr ref（ref expression）， Expr value，待set值;

**TempExprNode**： temporary expression基类，TempExprs主要用于rewriting pass如layout，type transform中间结果的定义; TempExprNode子类使pattern match时可使用特定类型的TempExpr并在expression rewriting时使用;

## 编译流程

下图所示描述了Relay GraphExecutor的JIT编译流程，其AOT流程以及VMExecutor的流程与其类似。总体而言，其也是通过调用C++端实现的编译过程，首先整体对图级别进行优化，而其中较为关键的FuseOps，因计算图中存在大量Op调用所对应的CallNode，若不fuse成整体的FunctionNode会存在较大的内存开销。

后续调用Codegen完成对应IRModule以及其中对应PrimFunc的生成。其中GraphPlanMemory这一步完成图中内存的分配，注意当前GraphPlanMemory中不支持IfNode，这也意味着当前GraphExecutor无论是JIT还是AOT编译模式目前仍无法支持灵活灵活控制流，即使If在Relay IR整体设计中是支持的。因此，为支持此功能TVM引入了VMExecutor并实现对应VM runtime来支持灵活的控制逻辑。此思路上比较类似于Torchscript设计对应的解释器来实现对于python较高的支持，而XLA通过在LLVM层级生成对应的支持。

<div align="center">
<img src="https://github.com/hehesnail/Boring_code/blob/main/imgs/relay_build_flow.png" width="100%" height="100%" /> 
</div>

一般来说，先创建 Relay **Function**(relay.ir.Function)，通过传入params(tvm.relay.Var)以及body(tvm.relay.Expr)创建; 后将该func添加进入**IRMoulde**中; 可通过BindParamsByName绑定Function中对应Var为外部参数（如权重），针对其中VarNode直接替换为对应Const即可;

首先调用build, 其为将relay function编译至graph executor的方式，主要参数为IRModule，params(str->NDArray)，以及target信息，对于直接传入Func而言，先绑定params后create IRModule;

后续创建BuildModule对象，调用build方法 ---> 其返回值为graph_json(标识图结构，最后多函数串联关系)/runtime::Module/ graph params; 后根据executor，返回对应AOTExecutorFactoryModule或者 GraphExecutorFactoryModule.

python端BuildModuled类__init\_\_ 调用C++端定义 _BuildModule, 创建RelayBuildModule并返回。 其余函数通过Module __get_item\_\_调用GetFunction获取C++端返回的PackedFunc;

核心函数 **BuildRelay**其定义在build_module.cc中，对应流程如下:

**Optimize** ---> 图级别的优化，最重要的可能是 fuse_ops，从算法上看是基于 dataflow分析;在graph level优化后，CallNode中对应的Op不为OpNode，因其被fuse，为FunctionNode;

**Codegen** ---> MakeExecutorCodegen (ret GraphCodegen, AOTCodegen), 因此 Codegen也为对应GraphCodegen 或 AOTCodgen的方法;

**GraphExecutorCodgen::Codegen**: GraphPlanMemory -> 输入参数转换 -> 函数Body遍历生成 -> update metadata & ret。

**GraphExecutorCodgen::CallNode**: 这里仅可为 FunctionNode，获取func，在其上作用 _make_CCacheKey 以及 **_CompileEngineLower**，经过 lower后会选择合适的 Op实现，并且将 FunctionNode最终转换为 PrimFunc，其内部主要调用 **LowerInternal**;

**LowerInternal**: CreateSchedule -> LowerSchedule, 主要创建 Schedule，这里就会将FuncNode中全部变为 Operation; 如下可以展示，中间IR形式的变换，具体可看出为从Relay IR，至schedule到最终 TIR Primfunc的流程。

如下对应代码片段为，在经过fuse_ops pass后，在创建schedule之前的Relay IR形态。此时计算图仅为全连接层+sigmoid层，因此在fuse的时候对应的dense op 和 sigmoid op的CallNode被合并至同一FunctionNode中，消除了dense op和sigmoid op之间拷贝的内存开销，而该FunctionNode后续转换为对应的PrimFunc。
```c++
// Just relay.nn.dense & relay.sigmoid

//************** bef create schedule **************
FunctionNode([Var(p0, ty=TensorType([5, 5], float32)), Var(p1, ty=TensorType([1, 5, 5], float32))], TensorType([5, 5], float32), CallNode(Op(sigmoid), [CallNode(Op(nn.contrib_dense_pack), [Var(p0, ty=TensorType([5, 5], float32)), Var(p1, ty=TensorType([1, 5, 5], float32))], relay.attrs.DenseAttrs(0x218cde8), [TensorType([5, 5], float32), TensorType([1, 5, 5], float32)])], (nullptr), [TensorType([5, 5], float32)]), [], {"Primitive": 1, "hash": "a2d3f5d197085d29"})
```

下述代码片段对应为上述fuse后的到FunctionNode创建schedule后其中个Stage的表示，可以看出对应PrimFunc中被划分为了5个Stage，其中前两Stage为Placeholder对应输入，而中间两个Stage为ComputeOp，其对应nn.dense的实现并因其配置了schedule后被优化为两个Stage，最后Stage对应sigmoid操作。从中也看出当前fuse_ops后生成的compute/schedule描述中并没有融合计算，仅可针对内存访问上做出优化。而融合nn.dense与sigmoid elementwise算子计算定位于auto-schedule解决。
```c++
//**************aft create schedule, the sch **************
stage(placeholder, placeholder(placeholder, 0x1fb22c0))
stage(placeholder, placeholder(placeholder, 0x1f7ae60))
stage(compute.global, compute(compute.global, body=[reduce(combiner=comm_reducer(result=[(x + y)], lhs=[x], rhs=[y], identity_element=[0f]), source=[(placeholder[y.c, k]*placeholder[floordiv(x.c, 5), k, floormod(x.c, 5)])], init=[], axis=[iter_var(k, range(min=0, ext=5))], where=(bool)1, value_index=0)], axis=[iter_var(y.c, range(min=0, ext=5)), iter_var(x.c, range(min=0, ext=5))], reduce_axis=[iter_var(k, range(min=0, ext=5))], tag=dense_pack, attrs={"workload": ["dense_pack.x86", ["TENSOR", [5, 5], "float32"], ["TENSOR", [1, 5, 5], "float32"], (nullptr), "float32"]}))
stage(compute, compute(compute, body=[compute.global[y, x]], axis=[iter_var(y, range(min=0, ext=5)), iter_var(x, range(min=0, ext=5))], reduce_axis=[], tag=dense_pack, attrs={"workload": ["dense_pack.x86", ["TENSOR", [5, 5], "float32"], ["TENSOR", [1, 5, 5], "float32"], (nullptr), "float32"]}))
stage(T_sigmoid, compute(T_sigmoid, body=[tir.sigmoid(compute[ax0, ax1])], axis=[iter_var(ax0, range(min=0, ext=5)), iter_var(ax1, range(min=0, ext=5))], reduce_axis=[], tag=elemwise, attrs={}))
```

如下代码片段对应IRModule中lower后primfunc形态，其在IRModule中以fused_nn_contrib_dense_pack_sigmoid为标识，其中dense op的实现选择为dense_pack，op选择机制在创建schedule时完成，在下小节有详细描述。Primfunc中对应输入为两placeholder，输出为对应的T_sigmoid，后续graph_executor将IRModule会编译为runtime::Module。若IRModule中存在多个Primfunc之间的串联关系(输入输出对应)将通过json文件来说明。
```c++
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

## Op选择机制

这里为在上述lower过程中，在LowerInternal过程中会完成对fuse_ops后function_node的转换，选择function_node中对应op实现，并最终将其lower为Primfunc。

**ScheduleGetter**: CreateSchedule对应实现, 其通过调用类中Create方法，对输入输出进行转换并添加后，递归遍历Body完成转换;

**ScheduleGetter::CallNode**: 主要使用 relay.backend.lower_call 来进行 impl的选择，该函数定义在 python side， **select_implementation** 完成 op 的选择;

在select implementation中首先通过 get_valid_implementations获取当前对应target下所有注册strategy;通过调用 fstrategy -> ret op_strategy，注意这里Op对应的**fstrategy**通过op.get_attr("FTVMStrategy")获取，其返回为**GenericFunc**，看下GenericFunc其中**CallPacked**定义：
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

那么剩下的问题就在于 fstrategy何时被注册到op中，以及 其中 dispatch_dict_中何时注册了对应target函数;

**register_strategy**： relay/op/op.py中对应函数，**tvm.ir.register_op_attr(op_name, "FTVMStrategy", fstrategy, level)**， 注意这里将完成把 GenericFunc注册到Op中的;

对应register_strategy何时被调用？ 其通常那个在op中对应 _xxx.py文件中调用，如dense op在 _nn.py中调用reg.register_strategy("nn.dense", strategy.dense_strategy); 注意这个是将 dense_strategy(GenericFunc)注册进入op的属性中，而同GenericFunc的其他平台实现这之前就已被注册进入 dispatch_dict_中，通过调用GenericFunc的register; 默认的strategy的定义在 relay/op/strategy/generic.py中;
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
这里为对应dense_strategy定义的地方，而装饰器将创建名为 **dense_strategy** 的GenericFunc， 并将fdefault设置为这里的dense_strategy;
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

可以看出对于 tvm relay来说，和 TIR 层级不同点在于其计算图组成有很多的 CallNode组成，在使用 relay描述计算时，其对应为Op，而在 optmize, lower等过程中，其通过fuse变换为 FunctionNode，此也可以看作将 graph 划分为多个子图，而每个FunctionNode则会被 lower为对应的 primfunc, 此时主要通过将 FunctionNode中对应 Call选择合适的 Op实现(i.e., schedule)，并且对于 CallNode 添加 inputs/outputs tensors，其实这里是将其转为 TIR 层级的计算图 (Op-Tensor graph)。

## 小结

### 表示能力

Relay同XLA一样定位于面向神经网络的描述，因此其不直接提供于将python解析为对应IR的方法。从社区提交来看，其tensorscript也为将python通过AST解析转译为TIR层级的描述，聚焦层面为算子层级灵活的描述，因此其对应层级也为Op层级，尚不明确其是否有跨层级的描述能力。

从Relay IR本身来看，可以支持灵活的控制流以及Op调用，同时也类似与XLA通过Tuple/TupleGetItem支持Op的多输出，因此其本身的表示能力足够。但当前GraphExecutor对应JIT/AOT模式中无法支持到对应的表示能力如If，当前通过VMExecutor支持，但同时也引入了对runtime更高的要求。


# 总结

<div align="center">
<img src="https://github.com/hehesnail/Boring_code/blob/main/imgs/dsl_comp.png" width="100%" height="100%" /> 
</div>

上述表格中总结了文中调研对象的主要特点，从转换方式，python支持能力，定位以及其未来的技术规划出发进行了对比。从中可以看出不同框架因其本身特性选择的侧重点和规划方向也存在差异。Torchscript因Pytorch在训练领域的成功，整体仍然延续pythonic的设计思想，从python AST解析到对应JIT AST并在此层级直接支持python的控制流和List、Tuple等primitive。亦可看出Facebook作为互联网厂商，因其产品大部分线上模型，其首先支持模型JIT优化。Google则针对训练，部署，算法原型开发等方向划分十分明确并开发不同框架完成对应功能，同时其新引入的IREE作为首个端到端MLIR落地优化框架比较值得注意。Relay作为tvm的前端表示，主要聚焦于神经网络的推理，因此也可看出其对应直接从python转换过来的需求不是很强烈，反而算子层级的TIR这种需求更加强烈。

# References

torchscript文档: [torchscript_intro](https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html)，[jit_doc](https://pytorch.org/docs/stable/jit.html)，[op_parallel](https://pytorch.org/docs/stable/notes/cpu_threading_torchscript_inference.html)，[jit_code](https://zasdfgbnm.github.io/2018/09/20/PyTorch-JIT-Source-Code-Read-Note/)

torchscript python语法支持参考: [jit_lang_ref](https://pytorch.org/docs/stable/jit_language_reference.html#language-reference), [jit_builtin_funcs](https://pytorch.org/docs/stable/jit_builtin_functions.html#builtin-functions), [python_lang_cover](https://pytorch.org/docs/stable/jit_python_reference.html#python-language-reference), [jit_unsupported_pytorch_constructs](https://pytorch.org/docs/stable/jit_unsupported.html#torch-and-tensor-unsupported-attributes).

XLA Operations语义： [xla ops](https://www.tensorflow.org/xla/operation_semantics)

XLABuilder定义：[xla builder](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/xla/client/xla_builder.h)

XLA 新增硬件方法：[xla new hardware](https://tensorflow.google.cn/xla/developing_new_backend?hl=zh-cn)

XLA python端接口：[xla python interface](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/xla/python/xla_client.py)

JAX同XLA接入方法：[lax](https://github.com/google/jax/blob/master/jax/lax.py)，[xla_bridge](https://github.com/google/jax/blob/master/jax/lib/xla_bridge.py)，[xla_exec](https://github.com/google/jax/blob/master/jax/interpreters/xla.py)

JAX官方文档: [jax_doc](https://jax.readthedocs.io/en/latest/index.html)

TVM官方文档: [tvm_doc](http://tvm.apache.org/docs/dev/index.html)

JAX源码：[jax_code](https://github.com/google/jax)

tensorflow XLA源码：[xla_code](https://github.com/tensorflow/tensorflow/)

TVM源码：[tvm_code](https://github.com/apache/tvm)

Torchscript源码：[ts_code](https://github.com/pytorch/pytorch/tree/master/torch/jit)
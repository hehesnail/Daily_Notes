# Pytorch Notes 

### *2021.5.10 & 5.11 & 5.12 & 5.13*
* [Quickstart to pytorch internals](http://blog.ezyang.com/2019/05/pytorch-internals/)
## Tensor
* Tensor/TensorImpl --- Storage/StorageImpl
  * Tensor/TensorImpl (bridge pattern), Tensor提供对外的接口(Abstraction), 内部成员变量为指向TensorImpl的指针(Impl)，接口同实现解耦，有更好的扩展性.
  * Storage/StorageImpl 同理.
  * TODO -> more details.
* Methods for a tensor:
    * Each tensor has an associated torch.Storage, which holds its data. The tensor class also provides multi-dimensional, strided view of a storage and defines numeric operations on it.
    * torch.as_strided: create the view of torch.Tensor via size, stride, storage_offset. Warning: More than one element of a created tensor may refer to a single memory location, since in as_stride make_tensor use the self.storage as the Storage mem in TensorImpl.
    * torch.expand -> as_strided, just new view, same data.
    * torch.repeat -> call the empty (empty_cpu) -> allocate new memory.
    * storage() to get the Storage of the tensor, the data_ptr() to obtain the data pointer.
    * TensorView:
        * Views share underlying data with its base tensor, if you edit the data in the view, it will be reflected in the base tensor as well.
        * Why ? ---> Supporting View avoids explicit data copy, thus allows us to do fast and memory efficient reshaping, slicing and element-wise operations.
        * View ops: Basic slicing and indexing op, e.g. tensor[0, 2:, 1:7:2] returns a view of base tensor / as_strided() / detach() / diagonal() / expand() / expand_as() / movedim() / narrow() / permute() / select() / squeeze() / transpose() / t() / T / real / imag / view_as_real() / view_as_imag() / unflatten() / unfold() / unsqueeze() / view() / view_as() / unbind() / split() / split_with_sizes() / swapaxes() / swapdims() / chunk() / indices() (sparse tensor only) / values() (sparse tensor only)
        * reshape(), reshape_as() and flatten() can return either a view or new tensor, user code shouldn’t rely on whether it’s view or not.
        * TensorView类型算子实现时经由 Dispatch 到 对应 aten::native operator, 相当一部分由 at::as_stride实现, 其会创建新的 tensor 返回, 需注意的是此时TensorImpl的构造函数传入的Storage相同, 因此返回 tensor 和 被作用 tensor 共用同一内存。因此如 reshape, view, indexing返回的tensor若修改其中元素，其余tensor也会被修改。
    * TensorIterator -> ElementWise Op Impl
      * TODO Broadcast analysis.
* Refs: [tensor_view]( https://pytorch.org/docs/stable/tensor_view.html), [tensor class](https://pytorch.org/docs/stable/tensors.html), [tensor_internals_1](https://zhuanlan.zhihu.com/p/54896021), [tensor_internals_2](https://zhuanlan.zhihu.com/p/64135058), [tensor_internals_3](https://zhuanlan.zhihu.com/p/69530008), [tensoriterator](https://labs.quansight.org/blog/2021/04/pytorch-tensoriterator-internals-update/).
## Dispatch
* [Quickstart to dispatch](http://blog.ezyang.com/2020/09/lets-talk-about-the-pytorch-dispatcher/)
## 动态图 & Autograd
* [Dynamic graph overview_1](https://zhuanlan.zhihu.com/p/61765561), [Dynamic graph overview_2](https://zhuanlan.zhihu.com/p/65822256).
## JIT & TorchScript
* **Why TorchScript**:
    * TorchScript code can be invoked in its own interpreter, which is basically a restricted Python interpreter. This interpreter does not acquire the Global Interpreter Lock, and so many requests can be processed on the same instance simultaneously.
    * This format allows us to save the whole model to disk and load it into another environment, such as in a server written in a language other than Python
    * TorchScript gives us a representation in which we can do compiler optimizations on the code to provide more efficient execution
    * TorchScript allows us to interface with many backend/device runtimes that require a broader view of the program than individual operators.
* **TorchScript**:
    * **SSA form IR** of the graph. The instructions in this format consist of ATen (the C++ backend of PyTorch) operators and other primitive operators, including control flow operators for loops and conditionals.
    * A **script compiler**, which does direct analysis of your **Python source code** to transform it into **TorchScript**.
* **torch.jit.trace** ---> based on the example inputs. Tracing does exactly what we said it would: run the code, record the operations that happen and construct a ScriptModule that does exactly that. Fail in 1). tracing control flow depends on inputs, 2). tracing in-place operations of tensor views.
* **torch.jit.script** ---> convert python module to TorchScript. Faithfully convert the module to ScriptModule.
    * Parameters(self.weight) -> preserved
    * Submodules(self.layer1) -> recursively converted
    * Attributes(self.training) -> converted if possible
    * Methods -> convertd to TorchScript, start from top-level forward method, recursively convert methods it reaches.
    * Model structure -> preserved including: func calls, objects, control-flow.
* **Mixing Tracing and Scripting**: In many cases either tracing or scripting is an easier approach for converting a model to TorchScript. Tracing and scripting can be composed to suit the particular requirements of a part of a model.
* **ScriptModule**: one thing needs to be mentioned here is that the ScriptModule inherits from nn.Module, thus it can still be trained.
* **Optimize TorchScript**:
    * Standard compiler passes: DCE, CSE, loop unrolling, const propagation.
    * Tensor Optimizations: Algebraic peephole optimizations, Batching of matrix multiplications, Point-wise fusions of element-wise operations.
    * Runtime Optimization: No global interpreter lock, fork/wait parallelism at the language level.
    * Use profile-guided execution of TorchScript programs with guarded optimistic optimizations.
* **Inter-Op & Intra-Op parallelism**: torch.jit._fork & torch.jit._wait.
* **script flow**: -> analysis torch.jit.script (take func input as an example)
    * **get_jit_def**: build a jit ast (treeview) from a given func, get_source_lines_and_file -> ast.parse -> ret build_def. ctx -> SourceContext -> SourceRangeFactory (SourceRange, to locate code) in cxx side.
    * **build_def**: build_param_list -> build_expr for ret_type(if exsited) -> Decl obj -> ret Def obj.
        * build_param_list -> build_param -> build_expr/Var/Ident/Param
        * Decl(SourceRange, param_list(Param), ret_type), stmt of jit ast.
        * Def(Ident(r, def_name), decl, build_stmts), stmt of jit ast.
        * build_stmts(ctx, body) -> for s in stmts -> build_stmt
        * build_stmt -> StmtBuilder -> Builder: \_\_call\_\_ in Builder overload, based on node.\_\_class\_\_ find the right build_xxx method in StmtBuilder.
        * **build_stmts**:
          * call build_stmt(StmtBuilder) and ret the list of _jit_tree_views, create the object in cxx side, in tree_views.h / tree_views.cpp. 
          * methods ret contain: ExprStmt, Assign, Delete, Return, Raise, Assert, AugAssign, While, For, If, Print, Pass, Break, Continue, With.
        * **build_expr** --> ExprBuilder
            * binop_map, unop_map, boolop_map, cmpop_map
            * Rets: Select, Apply, Dots, TrueLiteral, FalseLiteral, NoneLiteral, Var, BinOp, UnaryOp, TernaryIf, SliceExpr, Subscript, ListLiteral, TupleLiteral, DictLiteral, Const, StringLiteral, ListComp, DictComp, Starred.
        * Example of JIT AST (TreeViews)
            ```python
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
    * **_jit_script_compile**: def in script_init.cpp
        * **script_compile_function**: get_python_cu -> cu->define -> defined->setSchema -> StrongFunctionPtr ret -> didFinishEmitFunction -> ret.
        * **get_python_cu** -> ret the shared_ptr<CompilationUnit\>.
        * **cu->define** -> CompilationUnit::define (ir_emitter.cpp). In this case, the lexer and parser are not called since the JIT AST is constructed in python side via build_stmt and build_expr.
        * **CompilationUnit::define** -> creator contains **to_ir**, fn is created by **GraphFunction**.
        * py::class_<StrongFunctionPtr>(m, "ScriptFunction", py::dynamic_attr()), the ret of script_compile_function is StrongFunctionPtr, called as ScriptFunction in python side.
        * **ScriptFunction**: **\_\_call\_\_** -> **invokeScriptFunctionFromPython** -> ret of  **runAndInsertCall** is warped by toPyObject.
        * **GraphFunction::run** -> get_executor().run(stack)
        * **get_executor** -> call the GraphFunction::ensure_defined(**to_ir called at this time to emit IR**), ret the created GraphExecutor.
        * **GraphExecutor & GraphExecutorImplBase**
            * members: graph, function_name_, num_inputs, num_outputs.
            * graph -> via GraphFunction::preoptimizeGraph method, including optim passes: inlineCalls, PeepholeOptimize, ConstantPropagationImmutableTypes, ConstantPooling.
            * call GraphExecutorImplBase::run to execute
            * create ExecutionPlan via getPlanFor -> getOrCompile
              * getOrCompile -> plan_cache exists ret, call compileSpec with ArgumentSpec.
              * compileSpec -> compile the opt_graph, passes here.
              * I**Passes**:
                * **phase 0** -> Inline functions: Inline / LowerGradOf /specializeAutogradZero / LowerSimpleTuples / ConstantPooling.
                * **phase 1** -> Specialize to input definedness & run required passes: RemoveExpands / CanonicalizeOps / EliminateDeadCode.
                * **phase 2** -> Propagate detailed information through the graph: ConstantPropagation / PropagateInputShapes / PropagateRequiresGrad.
                * **phase 3** -> Run differentiable optimizations (i.e. simple graph rewrites that we can still execute using autograd): EliminateDeadCode / EliminateCommonSubexpression / PeepholeOptimize / ConstantPropagation / ConstantPooling / UnrollLoops / RemoveListMutation / PeepholeOptimize / ConstantPropagation / EliminateCommonSubexpression / CheckInplace.
                * **phase 4 & phase 5** -> slice out symbolically differentiable subgraphs & apply non-differentiable optims to the graphs.
                * **need_gradient**: CreateAutodiffSubgraphs / differentiate / PropagateInputShapes / runOptimization / runNondiffOptimization / packGradient / InlineAutodiffSubgraphs.
                * **no need_gradient**: runNondiffOptimization -> DecomposeOps / LowerSimpleTuples / BatchMM / FuseGraph / 
              * ret created ExecutionPlan via opt_graph & func_name.
              *  ExcutionPlan member **Code & CodeImpl**: constructor of CodeImpl call run method -> (emitCodeForBlock -> insertInstruction -> insertBailoutBlocks), thus here to emit instructions for Interpreter runtime.
              *  Check the instruction.h for OpCode defs. 
                 *  Note the **emitOperator** method of CodeImpl call node->getOperator to get the registered opeartors. 
                 *  The operators are registered via **RegisterOperators** class constructor -> call **registerOperator**, register to the  static OperatorRegistry. RegisterOperators in various places, note in some passes, call RegisterOperators for newly operator (i.e. fused_operator).
            * create InterpreterState via created ExecutionPlan code, call run method.
        * **InterpreterState & InterpreterStateImpl**:
          * constructor of Impl, call the enterFrame method to set base_pointer(pc) and registers.
          * call runImpl method to execute the emitted instructions, the final ret result is on the top of the stack.
          * Take a look at code_impl.h/**emitNode** method, **default emitOperator**, otherwise based on Node kind, call corresponding method. 
          * other kinds -> (prim::Drop, prim::Constant, prim::If, prim::Loop, aten::wait, prim::CallFunction, prim::CallMethod, prim::TypeCheck, prim::Bailout, prim::profile, prim::profile_ivalue, prim::GetAttr, prim::SetAttr, prim::ListUnpack, prim::TupleConstruct, prim::ListConstruct, prim::DictConstruct, prim::CreateObject, prim::isinstance, prim::TupleSlice, prim::fork, aten::warn, prim::Enter, prim::Exit).
          * Note in pytorch, add/sub are aten operators, thus for interpreter, these things are handled by OpCode (OP & OPN), the common behaviour of operator is getting the inputs from stack, running the operator, saving res back to the stack.
  * Refs: [torchscript_intro](https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html), [jit_doc](https://pytorch.org/docs/stable/jit.html), [op_parallel](https://pytorch.org/docs/stable/notes/cpu_threading_torchscript_inference.html), [jit_code](https://zasdfgbnm.github.io/2018/09/20/PyTorch-JIT-Source-Code-Read-Note/)
* **JIT AST**: tree.h / tree_views.h / tree_views.cpp
  * Tree & TreeRef: 表示树节点，同时进行 pre & post typechecking, Compound 继承自 Tree, AST 中节点大部分为此类型， 相比父类增加 SourceRange, TreeList(SmallVector<TreeRef\, 4>) 数据成员.
  * TreeView: AST中Node父类，包括 TreeRef data member, 同时提供了基本 methods. Stmt & Expr inherits TreeView, 而其他语句(If, For, While...) 继承自 Stmt，Expr同理.
  * 其 CFG 在 tree_views.h 中定义.
  * Two ways construct the AST: 
    * 1). Python side, StmtBuilder & ExprBuilder, 递归地将 Python code转换为 AST， 依据 python ast分析.
    * 2). CXX side, 对于传入的 soure code(python) 调用 Lexer 及 Parser 创建 JIT AST.
* **Graph IR**: Graph & Node & Value & Block, defs in ir.h/ir.cpp
  * Graph: 表示 one "function" of computation, 其由Node串联组成.
  * Node: IR Graph 基础类， 表示计算 以及其依赖的 Values, 其可包含多个Blocks（用来定义 nested control flow, For/If）. 注意 Graph 的连接关系是通过 Node 串联起来，Graph 中插入新节点，实际作用在 Node为元素的双向链表(拓扑排序)上，可通过 next/prev获取 Node.
  * Value: 用以表示 node 的输入输出， type: Tensor or opaque Handle object.
  * Block: List of Nodes, with inputs/outputs.
  * 结合后面 AST to IR 来看， Graph IR 更类似于提供了 AST Node 中对应到的图中编排方式(重新组织)，分析AST时以拓扑排序组织，以 prim::xxx, aten::xxx 为粒度. 以图的方式组织，结合 Node多为 Operator的性质，因此可做基于图层级的优化, bmm, fused_op etc. 同时，因其会ConvertToSSA，从而基于此做更多 static compiler optimization.  
* **JIT AST to IR**: mainly **to_ir** in ir_emitter.cpp
  * **Flow**: emitDef -> ConvertToSSA -> CanonicalizeModifiedLoops -> NormalizeOps -> runCleanupPasses.
  * **emitDef**: AST Root Def class, 主要还是根据 AST IR 的定义递归处理， IR转换过程也挺类似 Codegen的. 
    * **emitStatements**: 根据 stmt.kind, call 对应的 emitXXX func, emit stmt 一般会在 graph中添加对应的Node(topological order). e.g.: emitIf -> 先 emitCondExpr, 后 emitIfElseBlocks, graph中添加对应 Node(prim::If), Node add two blocks(true_block, false_block). For true & false, emitSingleIfBranch -> emitStatements again.
    * **emitExpr**: emitSugaredExpr, based tree kind, call emitSugaredExpr, emitApplyExpr, emitSubscript, emitSimpleExpr. 对于 Expr而言，其 ret 为 Value*.
  * 总体而言是根据 AST Node定义，递归处理，而在此每个Node对应的处理中， stmt会在 graph中添加对应的Node(以kind区分)，expr则作为stmt的输入输出，被预先处理.
* **Passes** & **JIT Executor**: 参考上述 _jit_script_compile 过程中简要描述.
* **TensorExpr(nnc)**: 
  * NNC stands for Neural Net Compiler. It is a component of TorchScript JIT and it performs on-the-fly code generation for kernels, which are often a combination of multiple aten (torch) operators.
  * jit interpreter automatically extracts subgraphs which or which specialized code can be JIT generated. combined kernerls avoid memory access to improve performance, i.e., fuser pass. thus can apply NNC code generation.
  * TE: borrow from halide and TVM.
  * 从注释说明来看，处于快速开发期.
## TorchScirpt Conclusion:
* **Python Support (表示能力)**: torchscript subset of python, 详细参考 [jit_lang_ref](https://pytorch.org/docs/stable/jit_language_reference.html#language-reference), [jit_builtin_funcs](https://pytorch.org/docs/stable/jit_builtin_functions.html#builtin-functions), [python_lang_cover](https://pytorch.org/docs/stable/jit_python_reference.html#python-language-reference), [jit_unsupported_pytorch_constructs](https://pytorch.org/docs/stable/jit_unsupported.html#torch-and-tensor-unsupported-attributes).
  * Types: TorchScript only supports a small set of types that are needed to express neural net models, each variable should have a single static type.
  * Supported Types: Tensor, Tuple, bool, int, float, str, List[T], Optional[T], Dict[K, V], T(torchscript class), E(torchscript enum), NamedTuple[T0, T1, ...].
  * Expressions:
    * literals
    * list construction, tuple construction, dict construction
    * arithmetic operators(+, -, *, /, ^, @)
    * comparision operators(==, !=, <, >, <=, >=)
    * logical operators(and, or, not)
    * subscripts and slicing(t[0], t[0:2], t[-1], t[1:], t[:1], t[-1, 1:, 0], t[i:j, i], t[0, 1:2])
    * funtion calls(calls to builtin-funcs, calls to other script funcs)
    * method calls(calls to methods of builtin types, compile start from forward method)
    * ternary expr(x if x > y else y)
    * casts(float(3), int(3.5)..)
    * accessing module parameters(self.xxx).
  * Statements:
    * simple assign, (a=b, a+=b, a-=b); pattern matching assign (a, b = tuple_or_list); multiple assign (a = b, c = tup)
    * print stmt (print("xxxxx", a+b))
    * if stmt (if, elif, else)
    * while loops, for loops with range, for loops over tuples, for loops over constant nn.ModuleList.
    * break, continue, return
  * Use of python values: python values are not a first class part of TorchScript. Instead they are de-sugared at compile-time into the primitive types that TorchScript supports. Ref to **emitSugaredExpr** in ir_emitter.cpp.
  * **Comments**: 按照doc描述，其目的在于支持的 python语法为和描述神经网络相关的语法，亦可看成同 nn.Module强相关的python语法，从 torchscript.jit.script 直接作用于 nn.Module -> ScriptModule, function -> ScriptFunction主要为 nn.Module服务。但从目前支持的语法来看，其已具有相当高的灵活性，逻辑跳转，循环，List，Dict，Tuple等均支持。支持如此多的 python语法 以及 已有pytorch特性，建立在：
    * 1). 综合需支持语法，专为 JIT 设计的 Lexer, Parser, CFG, AST;
    * 2). 设计Graph IR 用以组织AST的节点，且该Graph IR 易于 1). apply optim passes, 2). convert to interpreter instruction.
    * 3). 综合Graph IR 及 图中节点已有的 Kind, 设计的 GraphExecutor -> 其包含优化 Passes 以及 Interpreter，而 Interpreter 中设计了对应的 Instruction 以及 不同OpCode的对应行为;
    * 4). Python -> JIT AST -> Graph IR -> Optim Passes -> Interpreter Instruction -> Interpreter Exec; 针对上述不同层级的路径，所支持的其间的转换以及类型匹配。
    * 5). pytorch原生强大的operator库(ATen), 因此相当多的对于 tensor的操作可通过 call operator 的方式调用 ATen 中对应的 operator(cpu, gpu)。 无论从 pass 还是 解释器的角度, 可以不用过细粒度的分析，如 +,-,*/,等不用设计对应的 OpCode, just operator call.
    * 6). JIT 优化不用 lowered 到更底层的 target device 的 IR 描述，比如从 GraphIR 层级 lowered 到 LLVM IR.
* DSL：
  * python语法：可行性, 工作量, JIT AST 绑定
  * Graph IR 层级: 可行性，工作量，同当前 Op 设计冲突性， 相当于 TIR 前置层
  * Passes: 基于 Graph IR
  * JIT Exectcutor: JIT Graph Executor 支持, VM实现工作量
  * torchscript & jit 同 pytorch 整体框架的高绑定性
  * inter_op parallelism: 算子间并行
  * C++ vs C 可行性


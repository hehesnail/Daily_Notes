# Pytorch Notes 

### *2021.5.10 & 5.11 & 5.12*
## Overview the pytorch
* [Quickstart to pytorch internals](http://blog.ezyang.com/2019/05/pytorch-internals/)
## Tensor
* Tensor/TensorImpl --- Storage/StorageImpl
  * Tensor/TensorImpl (bridge pattern), Tensor提供对外的接口(Abstraction), 内部成员变量为指向TensorImpl的指针(Impl)，接口同实现解耦，有更好的扩展性.
  * Storage/StorageImpl 同理.
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
        * **ScriptFunction**: __call\__ -> invokeScriptFunctionFromPython -> ret runAndInsertCall -> run.
        * **GraphFunction::run** -> get_executor().run(stack)
        * **get_executor** -> call the GraphFunction::ensure_defined(**to_ir called at this time to emit IR**), ret the created GraphExecutor.
        * **GraphExecutor & GraphExecutorImplBase**
            * members: graph, function_name_, num_inputs, num_outputs.
            * graph -> via GraphFunction::preoptimizeGraph method, including optim passes: inlineCalls, PeepholeOptimize, ConstantPropagationImmutableTypes, ConstantPooling.
            * call GraphExecutorImplBase::run to execute
            * create ExecutionPlan via getPlanFor -> getOrCompile
              * getOrCompile -> plan_cache exists ret, call compileSpec with ArgumentSpec.
              * compileSpec -> compile the opt_graph, passes here.
                * phase 0 -> Inline functions: Inline / LowerGradOf /specializeAutogradZero / LowerSimpleTuples / ConstantPooling.
                * phase 1 -> Specialize to input definedness & run required passes: RemoveExpands / CanonicalizeOps / EliminateDeadCode.
                * phase 2 -> Propagate detailed information through the graph: ConstantPropagation / PropagateInputShapes / PropagateRequiresGrad.
                * phase 3 -> Run differentiable optimizations (i.e. simple graph rewrites that we can still execute using autograd): EliminateDeadCode / EliminateCommonSubexpression / PeepholeOptimize / ConstantPropagation / ConstantPooling / UnrollLoops / RemoveListMutation / PeepholeOptimize / ConstantPropagation / EliminateCommonSubexpression / CheckInplace.
                * phase 4 & phae 5 -> slice out symbolically differentiable subgraphs & apply non-differentiable optims to the graphs.
                * need_gradient: CreateAutodiffSubgraphs / differentiate / PropagateInputShapes / runOptimization / runNondiffOptimization / packGradient / InlineAutodiffSubgraphs.
                * no need_gradient: runNondiffOptimization -> DecomposeOps / LowerSimpleTuples / BatchMM / FuseGraph / 
              * ret created ExecutionPlan via opt_graph & func_name.
              *  ExcutionPlan member **Code & CodeImpl**: constructor of CodeImpl call run method -> (emitCodeForBlock -> insertInstruction -> insertBailoutBlocks), thus here to emit instructions for Interpreter runtime.
              *  Check the instruction.h for OpCode defs. 
                 *  Note the **emitOperator** method of CodeImpl call node->getOperator to get the registered opeartors. 
                 *  The operators are registered via **RegisterOperators** class constructor -> call **registerOperator**, register to the  static OperatorRegistry. RegisterOperators in various places, note in some passes, call RegisterOperators for newly operator (i.e. fused_operator).
            * create InterpreterState via created ExecutionPlan code, call run method.
  * Refs: [torchscript_intro](https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html), [jit_doc](https://pytorch.org/docs/stable/jit.html), [op_parallel](https://pytorch.org/docs/stable/notes/cpu_threading_torchscript_inference.html), [jit_code](https://zasdfgbnm.github.io/2018/09/20/PyTorch-JIT-Source-Code-Read-Note/)
* **JIT AST**: python support -> 为其适配的 JIT TreeViews
* **IR Infra**:
* **JIT AST to IR**:
* **Passes**:
* **JIT Executor**: 
## TorchScirpt Conclusion:
* Python support (表示能力):
* JIT AST:
* Graph IR:
* Passes:
* JIT Executor & VM:
* tensorexpr(nnc):
* 设计思路：
* DSL：
    * python语法：可行性, 工作量, JIT AST 绑定
    * Graph IR 层级: 可行性，工作量，同当前 Op 设计冲突性， 相当于 TIR 前置层
    * Passes: 基于 Graph IR
    * JIT Exectcutor: JIT Graph Executor 支持, VM实现工作量
    * torchscript & jit 同 pytorch 整体框架的高绑定性
    * inter_op parallelism: 算子间并行
    * C++ vs C 可行性

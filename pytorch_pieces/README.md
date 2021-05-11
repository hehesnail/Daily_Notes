# Pytorch Notes 

### *2021.5.10 & 5.11*
* **torch.jit**
    * **torch.jit.trace**: invoked the Module, recorded the operations that occured when the Module was run, and created an instance of torch.jit.ScriptModule (of which TracedModule is an instance)
    * **TorchScript** records its definitions in an Intermediate Representation (or IR), commonly referred to in Deep learning as a graph.
    * Why **TorchScript**:
        * TorchScript code can be invoked in its own interpreter, which is basically a restricted Python interpreter. This interpreter does not acquire the Global Interpreter Lock, and so many requests can be processed on the same instance simultaneously.
        * This format allows us to save the whole model to disk and load it into another environment, such as in a server written in a language other than Python
        * TorchScript gives us a representation in which we can do compiler optimizations on the code to provide more efficient execution
        * TorchScript allows us to interface with many backend/device runtimes that require a broader view of the program than individual operators.
        * **torch.jit.script**, a **script compiler**, which does direct analysis of your Python source code to transform it into TorchScript.  
        * Mixing Tracing and Scripting: In many cases either tracing or scripting is an easier approach for converting a model to TorchScript. Tracing and scripting can be composed to suit the particular requirements of a part of a model.
        * **TorchScript**: uses a static single assignment **(SSA)** intermediate representation (IR) to represent computation. The instructions in this format consist of ATen (the C++ backend of PyTorch) operators and other primitive operators, including control flow operators for loops and conditionals.
        * **Tracer Fail Cases**: 
            * Tracing of control flow that is dependent on inputs (e.g. tensor shapes)
            * Tracing of in-place operations of tensor views (e.g. indexing on the left-hand side of an assignment)
            * Use the **torch.jit.script**.
    * **script**:
        * **get_jit_def**: build a jit ast (treeview) from a given func, get_source_lines_and_file -> ast.parse -> ret build_def. ctx -> SourceContext -> SourceRangeFactory (SourceRange) in cxx side. 
        * **build_def**: build_param_list -> build_expr for ret_type(if exited) -> Decl obj -> ret Def obj. 
            * build_param_list -> build_param -> build_expr/Var/Ident/Param
            * Decl(SourceRange, param_list(Param), ret_type).
            * Def(Ident(r, def_name), decl, build_stmts)
            * build_stmts(ctx, body) -> for s in stmts -> build_stmt
            * build_stmt -> StmtBuilder -> Builder: __call__ in Builder overload, based on node.__class__ find the right build_ method in StmtBuilder.
            * **build_stmts** ret the list of _jit_tree_views, create the object in cxx side, in tree_views.h / tree_views.cpp. methods ret contain: ExprStmt, Assign, Delete, Return, Raise, Assert, AugAssign, While, For, If, Print, Pass, Break, Continue, With.
            * **build_expr** --> ExprBuilder
                * binop_map, unop_map, boolop_map, cmpop_map
                * Rets: Select, Apply, Dots, TrueLiteral, FalseLiteral, NoneLiteral, Var, BinOp, UnaryOp, TernaryIf, SliceExpr, Subscript, ListLiteral, TupleLiteral, DictLiteral, Const, StringLiteral, ListComp, DictComp, Starred.
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
            * **cu->define** -> CompilationUnit::define (ir_emitter.cpp). In this case, the lexer and parser are not called. 
            * **CompilationUnit::define** -> creator contains **to_ir**, fn is created by **GraphFunction**.
            * py::class_<StrongFunctionPtr>(m, "ScriptFunction", py::dynamic_attr()), the ret of script_compile_function is StrongFunctionPtr, called as ScriptFunction in python side.
            * **ScriptFunction**: __call\__ -> invokeScriptFunctionFromPython -> ret runAndInsertCall -> run.
            * GraphFunction::run -> : get_executor().run(stack)


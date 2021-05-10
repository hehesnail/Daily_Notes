# LLVM Notes

### *2020.11.30*
* Write self-defined add LLVM IR generator
    * 将三元组信息 及 数据布局对象放入模块并创建 Module实例
    * 定义函数签名，参数及返回类型，FunctionType 及 SmallVector<Type*, 2> FuncTyArgs
    * 使用Function::Create()静态方法创建函数
    * 存储参数的Value指针，可使用函数参数迭代器获得
    * 使用entry标签(或值名称)创建第一个基本块(BasicBlock)，并存储指针
    * 在entry基本块中添加对应的IR指令，也可使用IRBuilder来构建IR指令
    * main函数中调用函数创建Module，并使用verifyModule()验证IR构造，通过WriteBitcodeToFile函数将模块的位码写入磁盘
### *2020.12.6*
* Add IR generator LLVM 10.0同书中示例3.4的差异还是非常大的，基本都改了一套接口，很多类的定义也完全变掉...总而言之，进行修改还是非常蛋疼的事...
* Pass类为实现代码优化的主要类，常见子类：ModulePass(作用于整个模块), FunctionPass(一次处理一个函数,需重载runOnFunction), BasicBlockPass(作用在每个基本块)，继承Pass子类写自定义pass后RegisterPass
* LLVM后端：通用API抽象后端任务并特化为不同平台后端；主要流程：指令选择(SelectionDAG)，指令调度(pre-register allocation and post-register allocation)，寄存器分配，代码输出；CodeGen, MC, TableGen and Target.

### *2020.12.10*
* Value is the class to represent a "Static Single Assignment(SSA) register" or “SSA value” in LLVM. The most distinct aspect of SSA values is that their value is computed as the related instruction executes, and it does not get a new value until (and if) the instruction re-executes.
* The *Builder* object is a helper object that makes it easy to generate LLVM instructions. Instances of the *IRBuilder* class template keep track of the current place to insert instructions and has methods to create new instructions.
* The *Module* is an LLVM construct that contains functions and global variables. 
* In the LLVM IR, *numeric constants* are represented with the *ConstantFP* class, which holds the numeric value in an APFloat internally (APFloat has the capability of holding floating point constants of Arbitrary Precision). 
* The LLVM IRBuilder knows where to insert the newly created instruction, all you have to do is specify what instruction to create (e.g. with *CreateFAdd*), which operands to use (L and R here) and optionally provide a name for the generated instruction.
* LLVM instructions are constrained by strict rules: for example, the Left and Right operators of an add instruction must have the *same type*, and the result type of the add must *match* the operand types. 
* Once we have the function to call, we recursively codegen each argument that is to be passed in, and create an LLVM call instruction.
* The call to *FunctionType::get* creates the *FunctionType*. Note that *Types* in LLVM are uniqued just like *Constants* are, so you don’t “new” a type, you “get” it. *Function::Create* creates the IR function. “external linkage” means that the function may be defined outside the current module and/or that it is callable by functions outside the module.
* *Module::getFunction* search Module’s symbol table for an existing version of this function.
*  Basic blocks in LLVM are an important part of functions that define the Control Flow Graph. Use *BasicBlock::Create* to create the basic block and tell insert point to the IRBuilder.

### *2020.12.14*
* With LLVM, you don’t need this support in the AST. Since all calls to build LLVM IR go through the LLVM IR builder, the builder itself checked to see if there was a constant folding opportunity when you call it. If so, it just does the constant fold and return the constant instead of creating an instruction.
* In practice, use IRBuilder when generate codes, but IRBuilder does all of analysis inline. Thus, no amount of local analysis will be able to detect and correct expression like this (1+2+x)*(x+(1+2)). This optimization needs two transforms: 1). reassociation of expressions and 2). Common Subexpression Elimination(CSE) to delete the redundant add instruction. Use Passes.
* In order to get per-function optimizations going, we need to set up a *FunctionPassManager* to hold and organize the LLVM optimizations that we want to run. Need to add a new *FunctionPassManager* for each module. LLVM supports both "whole module" passes and also "per-function" passes which just operate on a single function at a time.
* Add *JIT* compiler, 1). first prepare the environment to create code for the current native target and declare and initialize the JIT. This is done by calling some InitializeNativeTarget functions and adding a global variable TheJIT, and initializing it in main. 2). setup data layout for the JIT.
* JIT has a straightforward symbol resolution rule that it uses to find symbols that aren’t available in any given module: First it searches all the modules that have already been added to the JIT, from the most recent to the oldest, to find the newest definition. If no definition is found inside the JIT, it falls back to calling “dlsym("sin")” on the Kaleidoscope process itself. Since “sin” is defined within the JIT’s address space, it simply patches up calls in the module to call the libm version of sin directly.
* How to call function definitions more times: 
    * The easiest way to fix this is to put the anonymous expression in a separate module from the rest of the function definitions. The JIT will happily resolve function calls across module boundaries, as long as each of the functions called has a prototype, and is added to the JIT before it is called. By putting the anonymous expression in a different module we can delete it without affecting the rest of the functions.
    * In fact, we’re going to go a step further and put every function in its own module. Doing so allows us to exploit a useful property of the KaleidoscopeJIT that will make our environment more REPL-like: Functions can be added to the JIT more than once (unlike a module where every function must have a unique definition). When you look up a symbol in KaleidoscopeJIT it will always return the most recent definition.

### *2020.12.22*
* Add Control Flow to the language.
    * If/Then/Else. The *first* two things is adding lexer extensions, AST exntensions and Parser extensions for If/Then/Else. *Then*, we need to generate the LLVM IR. However, once the then/else blocks finished, we need help the code to know which expression to return. The overall codegen process is quite like the way when we writing the assembly language. gen the cond code, branch to two basic blocks(then, else) based on the cond, gen code for each block(note here we need to SetInsertPoint).  *Finally*, the most important thing should be taken away is to CreatePHI and addIncoming of the phi node.
    * For loop. The *first* two things is adding lexer extensions, AST exntensions and Parser extensions for For loop. The remaining thing is quite like the If/Then/Else control expression. The different thing here is where to insert the phi node, after draw the cfg for the FOR loop, we need to insert the phi node for the initial value and updated loop variable. Since the for loop always return 0 this tutorial, thus there is no need to insert phi node at the loop exits.

### *2021.1.5*
* LLVM IR生成变量探索，写了几个基本的测试函数：变量赋值，简单加法函数(返回参数的和，无其他局部变量)，数组访问。每个都会使用alloca instruction来在stack frame上申请空间，并进行变量的store 和 load操作。变量赋值：alloca mem, store constant value, ret。导致这个原因之一也是因为函数里内的为局部变量。简单加法函数还是会先alloca, add, ret. 数组访问，就是每次alloca一片空间，根据偏移load对应的数据即可。LLVM Module中管理a list of globals variables, a list of functions, a list of libraries, a symbol table还有target的特性。从其构造函数可以看出，主要需要提供名字和context来保证线程安全。而BasicBlock，Function其实不是LLVM Module所必需的，对于IRBuilder来说，其可以直接依赖于Context，常见用法是针对BasicBlock创建，因为BasicBlock是一推指令集合。将变量创建为全局变量，则就不需在创建Func及BasicBlock，可以利用extern在其他文件中直接访问到。

### *2021.1.6*
* LLVM Programmer's Manual - *Important and useful LLVM APIs*
    * isa<>, cast<>, dyn_cast<> templates: llvm源码使用了大量custom form of RTTI,且无需类必须含有虚函数表。这几个定义在llvm/Support/Casting.h下。参考https://llvm.org/docs/HowToSetUpLLVMStyleRTTI.html 使自定义class支持这几个模板。
        * isa<>: 判断ref或者pointer指向对象是否为特定class的实例。
        * cast<>: "check cast" operation, 将基类指针or引用转为派生类，若其不是正确类型的实例则产生assertion failuer。
        * dyn_cast<>: "check cast" operation, 不接收引用，只接收指针，和C++ dyn_cast<> 功能很类似
        * isa_and_nonnull<>: 类似于isa<>,不过接收空指针输入.
        * cast_or_null<>: 类似于cast<>, 不过接收空指针输入.
        * dyn_cast_or_null<>: 类似于 dyn_cast<>, 不过接收空指针输入.
    * Passing strings(StringRef and Twine classes): 针对std::string中会产生heap allocation,因此实现对应的类来高效地传递字符串.
        * StringRef: 支持类似string的操作,但不需要堆分配, 主要用于参数传递, 并且passed by value.
        * Twine: 主要用于高效的处理接收 拼接的字符串, 为轻量级的rope数据结构并指向临时对象,可通过C string, std::string, StringRef构建.
        * StringRef 和 Twine对象指向外部内存, 因此, 最好仅在定义需要高效接收or拼接字符串的时候使用.


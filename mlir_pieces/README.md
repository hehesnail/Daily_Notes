# MLIR Notes
## What is MLIR?
* 1). MLIR is a toolbox for building and integrating compiler abstractions. You are able to define your own set of operations
(or instructions in LLVM), your own type system, and benefit from the pass management, diagnostics, multi-threading, serialization/deserialization, and all of the other boring bits of infra. 
* 2). The MLIR project is also “batteries-included”: on top of the generic infrastructure, multiple abstractions and code transformations are integrated. 
* 3). The “multi-level” aspect is very important in MLIR: adding new levels of abstraction is intended to be easy and common. 
* The below figure demonstrates that by defining mult-level IRs in different design space and lower from high-level IR to low-level IR, one can quickly define an IR and reuse the infra of MLIR to do remaining things.
<div align="center">
<img src="https://github.com/hehesnail/Boring_code/blob/main/imgs/mlir_intuition.PNG" width="70%" height="70%" /> 
</div>

## Tutorial chapter 2: Emitting Basic MLIR  
  * MLIR is designed to be a completely extensible infrastructure; there is no closed set of attributes (think: constant metadata), operations, or types. MLIR supports this extensibility with the concept of **Dialects** . **Dialects** provide a grouping mechanism for abstraction under a unique namespavce. 
  * **Operations**: core unit of abstraction and computation, similar in many ways to LLVM instructions. Operations can have application-specific semantics and can be used to represent all of the core IR structures in LLVM: instructions, globals (like functions), modules, etc.
  * **Concepts of Operation**: 
    * 1). A name for the operation; 
    * 2). A list of SSA operand values; 
    * 3). A list of attributes; 
    * 4). A list of types for result values; 
    * 4). A source location for debugging purposes; 
    * 5). A list of successors blocks (for branches, mostly); 
    * 6). A list of regions (for structural operations like functions).
  * A dialect inherits from **mlir::Dialect** and registers custom attributes, operations, and types. It can also override virtual methods to change some general behavior.
  * MLIR also supports defining dialects declaratively via **tablegen**. Using the declarative specification is much cleaner as it removes the need for a large portion of the boilerplate when defining a new dialect. It also enables easy generation of dialect documentation, which can be described directly alongside the dialect. run the *mlir-tblgen* command with the *gen-dialect-decls*. 
  *  An **operation class** inherits from the **CRTP mlir::Op class** which also takes some **optional traits** to customize its behavior. **Traits** are a mechanism with which we can inject additional behavior into an Operation, such as additional accessors, verification, and more.  
  *  In MLIR, there are two main classes related to operations: **Operation** and **Op**. The **Operation** class is used to generically model all operations. It is ‘opaque’, in the sense that it does not describe the properties of particular operations or types of operations. Instead, the Operation class provides a general API into an operation instance. On the other hand, each **specific type** of operation is represented by an **Op derived class**. For instance ConstantOp represents a operation with zero inputs, and one output, which is always set to the same value. **Op** derived classes act as **smart pointer wrapper** around a Operation*, provide operation-specific accessor methods, and type-safe properties of operations. A side effect of this design is that we always pass around **Op derived classes “by-value”**, instead of by reference or pointer (passing by value is a common idiom in MLIR and applies similarly to attributes, types, etc). Given a generic Operation* instance, we can always **get a specific Op instance** using LLVM’s **casting** infrastructure.
  * **Operation Definition Specification(ODS) framework**: Facts regarding an operation are specified concisely into a TableGen record, which will be expanded into an equivalent mlir::Op C++ template specialization at compile time. Using the ODS framework is the desired way for defining operations in MLIR given the simplicity, conciseness, and general stability in the face of C++ API changes.
  * The Tablegen and ODS framework make it much easier to define the dialects and operations.

## Tutorial chapter 3: High-level Language-Specific Analysis and Transformation
* MLIR Generic DAG Rewriter 
* Two ways to impl pattern-match transformations: 1). Imperative, C++ pattern-match and rewrite; 2). Declarative, rule-based pattern-match and rewrite using table-driven Declarative Rewrite Rules (DRR) ---> Operations defined using ODS.
* C++, inherits from mlir::OpRewritePattern, matchAndRewrite method.
* DDR, class Pattern<dag sourcePattern, list<dag\> resultPatterns, list<dag\> additionalConstraints = [], dag benifhitsAdded = (addBenifit 0)\>; fill the basic template with specified rules.  

## Tutorial chapter 4: Enabling Generic Transformation with Interfaces
* **Core idea**: make the MLIR infrastructure as extensible as the representation, thus Interfaces provide a generic mechanism for dialects and operations to provide information to a transformation or analysis.
* Shape inference for toy lang: based on function specialization, inline all of func calls and perform intra-procedural shape propagation.
* Provide the interfaces for the inliner to hook into. **dialect interface**: a class containing a set of virtual hooks which the dialect can override. In this case, interface is DialectInlinerInterface.
* **operation interface** -> can be used to mark an operation as being "call-like". for this case, use the CallOpInterface.
* **Intraprocedural Shape Inference**: can also define operation interfaces using ODS framework. interface defined inherits from **OpInterface**. ShapeInferencePass -> FunctionPass, i.e., run on each Function in isolation, inheriting from **mlir::FunctionPass** and override **runOnFunction()**.

## Tutorial chapter 5: Partial Lowering to Lower-Level Dialects for Optimization
* **Dialect Conversions**: **A Conversion Target** & **A set of Rewrite Patterns** & **Optional Type Converter**.
* ConversionTarget: addLegalDialect (legal targets for lowering), addIllegalDialect & addLegalOp (partial lowering).
* Conversion Patterns: use RewritePatterns to perform the conversion logic. ConversionPatterns diff from RewritePatterns, accept an additional operands parameter containing operands that have been remapped/replaced, for type conversions. 

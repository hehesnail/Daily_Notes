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

## Toy Tutorial chapter 2: Emitting Basic MLIR  
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
  * ODS的从其他项目来看还是挺高，起码对于Operation的声明可以写在这里.

## Toy Tutorial chapter 3: High-level Language-Specific Analysis and Transformation
* MLIR Generic DAG Rewriter 
* Two ways to impl pattern-match transformations: 1). Imperative, C++ pattern-match and rewrite; 2). Declarative, rule-based pattern-match and rewrite using table-driven Declarative Rewrite Rules (DRR) ---> Operations defined using ODS.
* C++, inherits from **mlir::OpRewritePattern, matchAndRewrite method**.
* DDR, class Pattern<dag sourcePattern, list<dag\> resultPatterns, list<dag\> additionalConstraints = [], dag benifhitsAdded = (addBenifit 0)\>; fill the basic template with specified rules.
* DDR主要不熟悉语法，从开源项目的情况来看，用**mlir::OpRewritePattern**的更多，DDR不是非常灵活，针对规则明确的好点。Pattern Match DAG Rewriter更适合于local optimization，获取的信息为局部子图的信息，基于此可进行优化。  

## Toy Tutorial chapter 4: Enabling Generic Transformation with Interfaces
* **Core idea**: make the MLIR infrastructure as extensible as the representation, thus **Interfaces provide a generic mechanism** for dialects and operations to provide information to a **transformation or analysis**.
* Shape inference for toy lang: based on function specialization, inline all of func calls and perform intra-procedural shape propagation.
* Provide the interfaces to hook into. **dialect interface**: a class containing a set of virtual hooks which the dialect can override. 
* **operation interface** -> provide a more refined granularity of information that is specific and core to a single operation. 
* Process for supporting inline:
  * **DialectInlinerInterface** for inlining
  * toy.generic_call be aware for inliner, use operation interface **CallOpInterface** mark op call-like.
  * add CastOp to cast shape of function ret, along with **CastOpInterface**.
  * override materializeCallConversion hook in dialect interface.
* **Intraprocedural Shape Inference**: can also define operation interfaces using ODS framework. interface defined inherits from **OpInterface**. ShapeInferencePass -> FunctionPass, i.e., run on each Function in isolation, inheriting from **mlir::FunctionPass** and override **runOnFunction()**.
* Process for supporting ShapeInfer:
  * use ODS to define the ShapeInference operation interface.
  * add ShapeInferenceOpInterface to defined ops
  * implement ShapeInferencePass inherits from **mlir::FunctionPass** and override the runOnFunction() method.

## Toy Tutorial chapter 5: Partial Lowering to Lower-Level Dialects for Optimization
* **Dialect Conversions**: **A Conversion Target** & **A set of Rewrite Patterns** & **Optional Type Converter**.
* **Convert lowering pass**:
  * define the ConversionTarget.  
  * addLegalDialect (legal targets for lowering)
  * addIllegalDialect, define Toy dialect as Illegal.
  * addLegalOp, mark PrintOp as legal to skip, partial lowering.
* **Conversion Patterns**: use RewritePatterns to perform the conversion logic. ConversionPatterns differ from RewritePatterns, accept an additional operands parameter containing operands that have been remapped/replaced, for type conversions. 
* **Partial Lowering**: applyPartialConversion

## Toy Tutorial chapter 6: Lowering to LLVM and CodeGeneration
* The std and affine dialects already provide the set of patterns needed to transform them into LLVM dialect. **transitive lowering**
* FullConversion, mlir::translateModuleToLLVMIR export to LLVM IR
* JIT: Setting up a JIT to run the module containing the LLVM dialect can be done using the mlir::ExecutionEngine infrastructure. This is a utility wrapper around LLVM’s JIT that accepts .mlir as input. 

## Toy Tutorail chapter 7: Adding a Composite Type to Toy
* **Define Type Class**:  The **Type** class in itself acts as a simple wrapper around an internal **TypeStorage** object that is uniqued within an instance of an MLIRContext. When constructing a Type, we are internally just constructing and uniquing an instance of a storage class.
* **Define the Storage Class**: When defining a new Type that contains parametric data (e.g. the struct type, which requires additional information to hold the element types), we will need to provide a **derived storage class**.
* Define the class that interface with, i.e., **StructType** (inherits from Type::TypeBase), after class definition, add StructType to ToyDialect in initialize method. 
* **Exposing to ODS**: After defining a new type, we should make the ODS framework aware of our Type so that we can use it in the operation definitions and auto-generate utilities within the Dialect. 
* **Parsing and Printing**: At this point we can use our StructType during MLIR generation and transformation, but we can’t output or parse .mlir. For this we need to add support for parsing and printing instances of the StructType. This can be done by overriding the parseType and printType methods on the ToyDialect. 

## MLIR Lang Ref
* MLIR -> graph-like data structure of nodes(Operations),and edges(Values). Value is the result of exactly one Operation or Block argument.
* Operations -> Blocks -> Regions. Operations may also contain regions, can represent different concepts.
* Define the syntanx of Opeartions, Blocks.
* **Operations**: describle many different levels of abstractions and computations, contain zero or more results/operands/attributes/successors/regions.
* **Blocks**: a list of operations, start with ^, take a list of block arguments, notated in a function-like way.
* Regions: A region is an ordered list of MLIR Blocks. Regions do not have a name or an address, only the blocks contained in a region do. Regions must be contained within operations and have no type or attributes. function body is an example of region.
* **SSACFG regions**: control flow semantics of a region is indicated by RegionKind::SSACFG . Informally, these regions support semantics where operations in a region ‘execute sequentially’. 
* **Graph regions**: graph-like semantics in a region is indicated by RegionKind::Graph . Graph regions are appropriate for concurrent semantics without control flow, or for modeling generic directed graph data structures.
* **Attributes**: the mechanism for specifying constant data on operations in places where a variable is never allowed. Each operation has an attribute dictionary, which associates a set of attribute names to attribute values. 

## Understanding the IR Structure
* **Traversing the IR Nesting**
  * Opeartion -> Regions, Regions -> Blocks, Blocks -> Opeartions
  * printOperation
    * op->getName, op->getAttrs, op->getNumRegions, op->getRegions etc.
  * printRegion
    * region->getBlocks
  * printBlock
    * block->getNumArguments, block->getNumSuccessors, block->getOperations
* **IR traversal methods**
  * **filtered iterator**: getOps<OpTy\>()
  * **Walkers**: walk() helper on Operation, Block, Region; argument: callback method, ret WalkResult::interrupt() value to stop walk.
* **Traversing the def-use chains**:
  * relation -> link Value with its users. Value: BlockArgument or res of one Operation. 
  * The users of a Value are Operations, through their arguments: each Operation argument references a single Value.
  * **get producers**: op->getOperands() --> operand->getDefiningOp or operand.cast<BlockArgument\>. operand here is Value.
  * **get users**: op->getResults() --> result.use_empty(), result.hasOneUse(), result.getUses().begin()/end(), result.getUsers(). result here is Value, getUsers() ret is Operation.

## Tutorials -> creating a dialect
* Mainly about how to configure cmake files for a mlir project.
* **TableGen targets**: 
  * **add_mlir_dialect** to declare the dialect for dialect ods files. 
  * For transforms.td files, **mlir_tablegen** & **add_public_tablegen_target** since result another 'IncGen' target.
* **Library targets**:
  * Each library is typically declared with **add_mlir_dialect_library()**
  * dialect libs depends on: 
    1. generated header files from TableGen, specified via **DEPENDS**;
    2. **LINK_LIBS** for specifing linkage to MLIR libraries, also other dialect library.
    3. **LINK_COMPONENTS** for specifing linkage to LLVM libraries.
* **Conversion**:
  * decompose the conversions with conversion passess.
  * **add_mlir_conversion_libarary** to decalre each conversion, fill the **ADDITIONAL_HEADER_DIRS** & **LINK_LIBS** if needed.

## Tutorials -> add mlir graph rewrite
* Mainly about how to define rewrite using patterns as well as using a graph walker.
* **patterns + rewrite engine** is preferred.
* **TableGen patterns**: define the pattern, DAG-DAG, source pattern args can be used in result pattern. for more advance pattern, can use native code fallback method and add c++ funcion to perform replacement. Then **register** the pattern.
* **Simple C++ matchAndRewrite style specs**: impl pattern function, add it to pattern set via **populateRewrites**.
* **Function-style canonicalization**: specify "let hasCanonicalizeMethod = 1" for operation, then implement canonicalize method.
* **General C++ RewritePattern specs**: Inherits from RewritePattern, implement match/rewrite, or matchAndRewrite, also add this pattern to RewritePatternSet.

## Dialect Conversion
* **Dialect conversion framework**:
  * a conversion target
  * a set of rewrite patterns
  * a type converter(optional)
* **Conversion modes**: 1). partial conversion(applyPartialConversion); 2). full conversion(applyFullConversion); 3). analysis conversion(applyAnalysisConversion).
* **Conversion Target**:
  * a formal definition of what is considered to be legal during the conversion process.
  * operations mark: 
    1. Legal: every instance of a given operation is legal;
    2. Dynamic: only some instances of a given operation are legal.
    3. Illegal: no instance of a given operation is legal, i.e., must be converted.
  * addLegalDialect, addLegalOp, addDynamicallyLegalDialect, addDynamicallyLegalOp, markUnknownOpDynamicallyLegal, addIllegalDialect, addIllegalOp.
* **Conversion Patterns**:
  * a set of legalization patterns must be provided to transform illegal operations into legal ones. 
  * class **ConversionPattern**, addtional args for matchAndRewrite & rewrite methods, i.e., list of operands that the operation should use after conversion. 
* **Type Conversion**:
  * A **TypeConverter** may be used to convert the signatures of block arguments and regions, to define the expected inputs types of the pattern, and to reconcile type differences in general.
  * A conversion describes how a given illegal source Type should be converted to N target types, addConversion.
  * A materialization describes how a set of values should be converted to a single value of a desired type. A materialization can produce IR.
    * Argument Materialization
    * Source Materialization
    * Target Materialization

## Pass Infrastructure
* **Operation Pass**:
  * Op-Specfic: operates explicitly on a given operation type. Inherit **OperationPass** and provide operation type, override the **runOnOperation** method.
  * Op-Agnostic: operates on the operation type of the pass manager that it is added to. Inherit **OperationPass**, override the **runOnOperation** method.
* **Analysis Management**:
  * In MLIR, analyses are not passes but free-standing classes that are computed lazily on-demand and cached to avoid unnecessary recomputation. 
  * OperationPass class provides utilities for querying and preserving analyses:
    * getAnalysis<>, getCachedAnalysis<>, getCachedParentAnalysis<>, getCachedChildAnalysis<>, getChildAnalysis<>.
* **Pass Manager**:
  * The PassManager class acts as the top-level entry point, and contains various configurations used for the entire pass pipeline. The OpPassManager class is used to schedule passes to run at a specific level of nesting.
  * **OpPassManager**: collection of passes to execute on an operation of a specfic type. **addPass** to add to pass manager, **nest** to handle nested structure.
  * **Dynamic Pass Pipelines**: In some situations it may be useful to run a pass pipeline within another pass.
  * **Pass Statistics**: Statistics are a way to keep track of what the compiler is doing and how effective various transformations are. added to a pass by using the **Pass::Statistic** class.
  * **Pass Registration**: PassRegistration class, this mechanism allows for registering pass classes.
  * PassPipelineRegistration class for pass pipeline regsiteration.
* **Declarative Pass Specification**:
  * Using the **gen-pass-decls** generator, we can generate most of the boilerplate above automatically. This generator takes as an input a -name parameter, that provides a tag for the group of passes that are being generated.
* **Pass Intrucmentation**:
  * MLIR provides a customizable framework to instrument pass execution and analysis computation, via the PassInstrumentation class.
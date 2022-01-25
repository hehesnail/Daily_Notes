## 模型中动态性
### Dynamic Control Flow Model
较常见的为RNN/LSTM等在Train/Eval的时候需要按照time-step展开的模型，因此导致Loop的控制流；所有使用此类的block的模型都算这一类。If等条件分支的模型基本上没见过，考虑到算法选型时，待分支的很可能导致模型训练无法收敛。

### Dynamic Shape
常见于Input不确定的情况，如NLP, 语音, Video, Multi-Modal等任务，输入可为变长序列，当前也可通过截断或者采样的方式固定为定长序列。从论文看，大家都喜欢比较Bert的性能，可能跟都是互联网厂商，Bert有大规模应用的需求相关。对于Transformer在Image上的应用，将Image拆成多个Patch后搞Attention，确实也不存在变长的输入。

### Dynamic Data Structure
部分模型中存在特殊的数据结构，如常用于代码表示学习的Tree-LSTM/Tree-RNN，存在递归数据结构Tree。

### Dynamics in module
在一些视觉DownStream的任务中存在较多的动态性；
* HOI(人物关系检测)中一般通过Attention或者Graph modeling的方式对单图中(person, object)对的Relation建模，后续Infter的过程，因目标个数存在变化，因其Loop这样的控制流逻辑；
* Scence Graph Parsing/Generation等任务和HOI类似特性，Video Reasioning根据做的粒度不同，存在视频长度和单帧中物体个数变化的特性；

Dynamic Conv & Gated & Selective；
* dynamic conv -> generate conv kernel depends on input, key insight: time(gen kernel) < origin conv time. 从一组卷积核中获取部分卷积核，通常做法为加权；selective kernels同理
* Swith Transformer, Mix of Experts, 根据概率都单分支，topk/mask

### Contributions of Nimble
1. First end to end system for dynamic model inference across multiple hardware platforms
2. Propose compilation & optimization techniques:
   * **dynamic type system**, ref to class AnyNode in tir/expr.h, in related relay IR type system, also related modifications in type_solver.cc/type_infer.cc along with ops.
   * **memory planning** algorithm for dynamic models, related VM instructions (invoke_mut/alloc storage/alloc tensor/kill) placement. Ref to related passes: ManifestAlloc & MemoryPlan.
   * **heterogeneous device placement**: annotate & propgate the device for all IR nodes, after lower process, ref to AnalyzeContext & ContextAnalysis passes.
   * **symbolic codegen**: Based on different shapes, generate K configurations. Codes not provided yet.
3. Propose the related VM runtime to support the dynamic model execution, ref to VM.cc.

### Contributions of DISC
1. Extend HLO with dynamic shape support, DHLO, e.g, slice op definitions from static value to dynamic tensor value.
2. Based on shape constraints from op semantics and collected from high-level framework, emit shape calculation codes and perform buffer liveness analysis and optimization, compile the runtime flow.
3. Shape constraints & shape propagation ---> more aggressive op fusion strategy.
4. Bridge from tensorflow / pytorch computational graph.

### Nimble vs DISC ?
* 表示性：Nimble, Nimble直接扩展Relay IR，相对于XLA，Relay提供更加灵活的表示能力如List，Tuple等扩展；
* 扩展性：Nimble，基于TVM已有Stack，可复用其已有前端(pytorch,tf,onnx等) + 已有优化Pass + 已支持平台，同时也考虑到ARM移动端的场景。DISC更可直接看做XLA的一个扩展，并通过MLIR Infra实现该扩展，其主要放出的结果也集中于GPU；
* 性能：DISC，主要对应提升在VM运行时的开销 + Nimble缺乏对于dynamic shape op fusion的策略；
* 侧重点: Nimble为针对动态模型(包括动态shape以及动态控制流)的端到端系统设计，包括类型系统、优化Passes、虚拟机运行时。DISC仅针对XLA做了动态shape的支持扩展；
* 嵌入式环境支持：DISC直接扩展XLA针对嵌入式场景本身就不支持，Nimble对应的 VM Runtime实现过于heavy weight。

## 机会点
**总体评估**：工程量较大，牵扯层面较多，相关工作数量不多

**框架**： IoT场景下 dynamic model的部署支持，在资源受限场景下的Runtime设计。类似当前的HIR设计，裁剪掉Nimble Runtime中的alloc_storage等指令，采用更轻量和开放的方式。这个点可以包装下，其可对接任意AI-Chip和其对应推理引擎，提供dynamic model部署能力、甚至于pythonic的编程方式，可在开发完成后，细化创新点以及针对schedule、pass等作出比较fancy的点增强contribution。

**优化手段**:  针对动态shape目前op fusion策略、内存分配策略进行扩展，基于Nimble或DISC等已有工作进行扩展。

**扩展边界**：

* 支持动态模型的自动优化方法，可针对已有的SOTA如Ansor扩展对应动态模型的支持；
* 和inter-operator等方法结合，考虑到算子间并行性，提供intra-inter op的提升，如已有针对static model的IOS, 提供动态模型的支持。

**新数据结构**:
* 针对sparse数据或传感器数据提供IR层面的支持扩展，可考虑参考Taichi(描述同Data Structure解耦)。

## Refs
1. Nimble: Efficiently Compiling Dynamic Neural Networks for Model Inference, MLSys 2021.
2. DISC : A Dynamic Shape Compiler for Machine Learning Workloads, EuroMLSys 2021.

## Cortex: A Compiler for Recursive Deep Learning Models (MLSys 2021)
**TODO**

## The CoRa Tensor Compiler: Compilation for Ragged Tensors with Minimal Padding (Arxiv)
**TODO**
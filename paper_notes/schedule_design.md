## 目标1：算子级优化接口

### 1.可视化及schedule接口
提供 show_schedule_tree, 得到当前hirfunc中所有op-block的可视化结果，其中以不同颜色标记该算子是否可被调度，对应op名标记为symid, 并表明op-block从属funcname。

提供 get_op_schedule(funcname, symid)，创建symid所在block的schedule对象，并返回hik_op_schedule。而block内其余对象不用重复创建。block新增schedule对象，用以construct_op_block时check。

调度方式的手段及使用方式应为opc/noc文档负责说明，因此可视化时无需提供此功能.

### 2. hik_op_schedule, hik_op_stage

hik_op_schedule / hik_op_stage 可参考之前版本的实现，其并不care schedule本身，而是只是提供symid或名字(之前版本)到tvm tensor转换的过程，因此其理论可兼容任一版本schedule。

考虑到opc/noc可为op-block中算子实现，而未来会同其tvm进行解耦，存在新增额外的schedule primitive的可能性。

若noc新增tvm没有的schedule手段? 
* hik_noc_op_schedule / hik_noc_stage, get_op_schedule返回，此时希望在compiler config中可提供指定opc or noc的方法。
* hik_noc_op_schedule / hik_noc_stage 中增加noc特有优化方式的接口。

assumption 1：假定单个op-block对应的op均来自于同一backend, 即opc or noc；
* 此假定在外部用户使用时合理，因opc/noc的人用的时候，必然做单算子层级支持的事情，故不会牵扯到多种不同平台的算子混用.
* 若存在op-block中存在多个backend，其实际为subgraph partition的问题.
        
### 3. 使用流程
compile/analyze or 其他接口可完成op计算流产生 -> get_op_schedule -> various schedule primitives -> compile(无需set_schedule接口，此时schedule attach于block之上)

## 目标2：优化及调度能力扩展

图层级优化不暴露于外部用户，其主要用于扩展DSL接管模型的能力，并期望基于此可实现不支持模型在平台上得以支持。故不考虑dynamic control flow，此为dsl frontend需要接管的事情。因此下文也基于此点，从op-block接管模型的角度的考虑优化和调度能力。

DSL_Optimizer隶属于Op模块，服务于Op计算图的优化调度，同时提供DSL Virtual Env使调度不受实际硬件约束，但同时也可通过register hw params的方式set vDevice & vPE的特点。

### 基本框架

<div align="center">
<img src="https://github.com/hehesnail/Daily_Notes/blob/main/imgs/dsl_opt_sys_design.png" width="100%" height="100%" /> 
</div>

总体来说，提供一下的优化能力：
* 支持子图切分，子图调度(子图并行、pipeline并行、数据并行(may))
* 支持图算融合 - co-schedule
* 引入vDevice, vPE用于屏蔽platform specific信息

IR层级并不打算使用relay ir作为此层级的IR进行优化，考虑图算融合，但图算融合和relay没啥关系？relay也并不能提供graph-operator功能调度的能力，即图层级打破算子边界进行调度。还是打算采取graph表示，因诸多框架已证明其在Ops/subgraph层级调度的可行性。

拟定输入为onnx model，此时可将onnx model同subgraph match module(onnx ir层级)交互，获取不支持模型的子图。

1. 此时存在short path即支持的模型直接通过AIC转换为平台bin，而不支持的模型交给后续流程直到编译生成DSL Ops，最终在compute engine处汇合。
2. 另一点则可 onnx-> aic ->onnx + 不支持onnx subgraph 后merge成为整体onnx model转为graph ir，并且交由后续进行调度优化。

Milestone 1: 走通short path，此时并没有多少调度能力，仅区分出当前平台可支持的子图，但其可走通整体数据流;

Milestone 2: vDevice引入，subgraph partition模块;

Milestone 3: subgraph schedule + vdevice + compute engine mapping;

Milestone 4: enable the co-schedule ablitiy.


### Graph-Operator Co-schedule (basic idea from rammer)

<div align="center">
<img src="https://github.com/hehesnail/Daily_Notes/blob/main/imgs/nnfusion.png" width="100%" height="100%" /> 
</div>

### Current Weakness

* 大体框架确定，从直觉上来说应该是可行的，但各模块具体实现细节尚不明确。
* 暂不打算处理符号维度调度，对于dynamic类算子支持性有待考虑，个人倾向于dyanmic op不做任何处理，仅支持symbolic shape，此点兼容性未知。




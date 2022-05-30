# Transformer based optimization
背景：探索大模型在单卡资源受限的情况下，通过DSL Optimizer模块接管调度优化并在多个互联的设备上进行调度优化，并基于此过程中的调研探索出调度优化research方向。

## Transformer

### Tranformer & attention module

<div align="center">
<img src="https://github.com/hehesnail/Daily_Notes/blob/main/imgs/design_imgs/transformer.png" width="70%" height="70%" /> 
</div>

### BERT

BERT仅使用transformer模型的encoder部分，通过masked language modeling task完成模型的预训练。其特点为：pre-training model + bi-directional model.

<div align="center">
<img src="https://github.com/hehesnail/Daily_Notes/blob/main/imgs/design_imgs/bert1.png" width="70%" height="70%" /> 
</div>

<div align="center">
<img src="https://github.com/hehesnail/Daily_Notes/blob/main/imgs/design_imgs/bert2.png" width="70%" height="70%" /> 
</div>

## ViT (An image is worth 16x16 words: Transformers for image recognition at scale)

ViT将输入图片分为多个patch（16x16），再将每个patch投影为固定长度的向量送入Transformer，后续encoder的操作和原始Transformer中完全相同。

<div align="center">
<img src="https://github.com/hehesnail/Daily_Notes/blob/main/imgs/design_imgs/vit1.png" width="70%" height="70%" /> 
</div>

1. patch embedding：例如输入图片大小为224x224，将图片分为固定大小的patch，patch大小为16x16，则每张图像会生成224x224/16x16=196个patch，即输入序列长度为196，每个patch维度16x16x3=768，线性投射层的维度为768xN (N=768)，因此输入通过线性投射层之后的维度依然为196x768，即一共有196个token，每个token的维度是768。这里还需要加上一个特殊字符cls，因此最终的维度是197x768。到目前为止，已经通过patch embedding将一个视觉问题转化为了一个seq2seq问题;
2. positional encoding（standard learnable 1D position embeddings）：ViT同样需要加入位置编码，位置编码可以理解为一张表，表一共有N行，N的大小和输入序列长度相同，每一行代表一个向量，向量的维度和输入序列embedding的维度相同（768）。注意位置编码的操作是sum，而不是concat。加入位置编码信息之后，维度依然是197x768;
3. LN/multi-head attention/LN：LN输出维度依然是197x768。多头自注意力时，先将输入映射到q，k，v，如果只有一个头，qkv的维度都是197x768，如果有12个头（768/12=64），则qkv的维度是197x64，一共有12组qkv，最后再将12组qkv的输出拼接起来，输出维度是197x768，然后在过一层LN，维度依然是197x768;
4. MLP：将维度放大再缩小回去，197x768放大为197x3072，再缩小变为197x768.

**Model infos**

<div align="center">
<img src="https://github.com/hehesnail/Daily_Notes/blob/main/imgs/design_imgs/vit2.png" width="70%" height="70%" /> 
</div>

## BEiT

 BEIT随机mask部分image patches，让BEIT模型预测盖住的patches是什么，不断计算预测的patches与真实的patches之间的差异，利用它作为loss进行反向传播更新参数，来达到 self supervised learning.

<div align="center">
<img src="https://github.com/hehesnail/Daily_Notes/blob/main/imgs/design_imgs/beit.png" width="70%" height="70%" /> 
</div>

**Model Infos**: 

Use a 12-layer Transformer with 768 hidden size, and 12 attention heads. The intermediate size of feed-forward networks is 3072. We employ the default 16 × 16 input patch size.

## LightSeq2

* Oriented for BERT training.

**Kernel Fusion**:

Adjacent fine-grained element-wise kernels are fused into one coarse-grained kernel, resulting in fewer kernel launches and intermediate results.

<div align="center">
<img src="https://github.com/hehesnail/Daily_Notes/blob/main/imgs/design_imgs/tr_op2.png" width="70%" height="70%" /> 
</div>

**Memory-Efficient Mixed-Precision Trainer**: FP16/FP32 mixed-precision training To reduce memory and latency without hurting accuracy.

**Memory Management**: scan the training set and estimate the upper bound of the capacity, reuse for different batches for gradient accumulation.

## Data movement is all you need

* The key bottleneck when training transformers is data movement.

<div align="center">
<img src="https://github.com/hehesnail/Daily_Notes/blob/main/imgs/design_imgs/tr_op1.png" width="70%" height="70%" /> 
</div>

**Optimization steps:**
1. Dataflow graph construction and identify common operator classes.
2. Fuse the operators to reduce data movement
3. Evaluate the perf of operators with respect to data layout to find near-optimal layouts.
4. Find the best configurations to optimize end-to-end performance.

**Fusion:**

A significant portion of the runtime in existing transformer
implementations is in statistical normalization and element-
wise operators.

Method: Two operators can be fused if their iteration space implementations are compatible: They are either the same or the
only difference is that one operator performs a reduction.

Fuse two adjacent operators and continue until we cannot fuse further.

# Detailed schedule

* 方向：从目标来看，techniques for single model on single machine不符合我们预期的目标，因此将集中于multiple machines的技术;
* 需明确主要的平台：1).单卡资源受限；2).存在多卡互联的使用场景, candidate: MLU270;
* 关于papers: distributed + schedule + parallel, 扩大阅读量，明确大家都在做什么;
* Collaborative way：discuss more frequently + summarize weekly.

**week1 (5.30 -> 6.2)**
* 明确使用模型(bert)及平台，@jxc;
* 6.3 ---> 基于single model on multiple machines, schedule v0.1 plan;
* 模型计算量，内存使用分析，模型转onnx并导入dsl optimizer模块;

**week2 (6.6 -> 6.12)**
* 硬件环境配置，模型转换，CNN demo直接load模型运行;
* dsl optimizer导出分段模型+生成C算子，外部cnn demo多段调用test;
* along with paper survey;

**week3 (6.13 -> 6.19)**
* 多机runtime实现，支持parallel及dep运行;
* along with paper survey;

**week4 (6.20 -> 6.26)**
* dsl optimizer导出切分模型及schedule strategy + runtime对接
* along with paper survey

**week5 (6.27 -> 7.3)**
* End to end running example.
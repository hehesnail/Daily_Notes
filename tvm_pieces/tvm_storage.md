## TVM 计算图框架
### 计算图 (Dataflow Graph)
* 类似于tensorflow，TVM采用计算图来描述整体计算的流程，如下图所示，图中节点(Operation)为具体的计算描述(通过compute / extern / hybrid描述)如matmul，conv2d等等，而其输入输出则为Tensor，因此不难看出TVM中计算图是以Operation为Node，以Tensor为Edge的Graph。
<div align="center">
<img src=https://github.com/hehesnail/Boring_code/blob/main/imgs/dataflow_graph.png/ width=70%, height=70%>
</div>

* 计算图也被称为数据流图(Dataflow Graph)，即数据从Placeholder流入，经过图中路径上多个Op进行一系列运算，最终得到输出Tensor。需要注意的是，其Dataflow Graph具有SSA (静态单赋值) 的性质，即每个Tensor进入一个Op，被Op消耗 (Consume) 后，该Op会产生 (Produce) 新的Tensor作为输出。此性质影响了后续 TVM的Tensor管理机制，即先对Tensor/Buffer进行变换，后续通过一个Pass合并冗余内存。

### Operation/Tensor
上述通过计算图的简略介绍，我们知道对于计算图的Node和Edge，本小节进一步介绍TVM中Operation和Tensor。
Operation是TVM对计算的抽象，用户通过其提供的接口对计算进行描述，经过一系列变换后，Operation中包含了对此部分计算的描述IR，也即Stmt；目前TVM中有几种类型的Op，分别对应不同的描述机制。

* ComputeOp 对应 tvm 编写算子时常为使用 tensor expression language (TE)，并通过te.compute构建的语句。方便计算密集型实现；
* ExternOp 对应 tvm 编写算子时常为使用 IRBuilder，通过te.extern构建的语句，其直接构建Tensor IR，方便逻辑密集型算子的实现。
* Tensor则是TVM对于数据的抽象，而其在数据流图中 SSA的性质，也使得其在Schedule前期分析的时候不用管理其与实际内存的映射关系，提供了便捷性。

TensorNode中成员变量，从中可知，对于一个Tensor其主要包含shape，dtype，以及该tensor所属的op；
对应ComputeOpNode class定义，可以看到其主要包含循环轴，Op对应的Body，其中需要注意的成员函数为 InputTensors , BuildRealzie, BuildProvide，其分别对应该 Op 输入了哪些Tensor，同时又产生(Provide)了哪些Tensor；

对应ExternOpNode class定义，相对于ComputeOp不同点在于其需要指定输入输出Buffer，同时其Body直接为Stmt；其和后续Buffer管理相关的成员函数亦为 InputTensors / BuildRealize/BuildProvide。
最后，可以看一下TVM中由Tensor/Operation构成的计算图的表示形式。有三种类型图的表示形式，其中ReadGraph / FeedGraph 相对应分为Op 读取了哪些Tensor 以及 Tensor被哪些Op所消耗，其主要用于构建整个Schedule对象以及进行Bound分析；而AttachPath则主要用于Bound分析，且仅对ComputeOp等带有循环轴的Op作用。


## Tensor管理机制
### 整体流程
<div align="center">
<img src=https://github.com/hehesnail/Boring_code/blob/main/imgs/tvm_storage.png/ width=70%, height=70%>
</div>
上图为TVM整体Tensor最终变换为Alloacte Stmt(codegen时变换为内存申请)语句的整体流程；左中右分别为当前Pass，Pass会将Tensor或Buffer转换为进一步的IR Stmt/Expr类型，以及Pass中使用到的映射关系，例如Tensor-Buffer映射；

简单梳理下该流程：
* Step 1:  首先从用户编写的Op串联中，构建整个计算图，通过对计算图拓扑排序得到Op的相对顺序关系，依此创建 Schedule对象(此也为优化原语作用的对象)，此时每个Op中语句片段独立，且并无实际Tensor的变换；此时主要是确定了Op间的顺序，同时也确定了Tensor的先后顺序；此部分内容
* Step 2:  ScheduleOps 通过对上一步获得的Ops顺序，从后往前调用Op BuildProvide / BuildRealize完成整个schedule body的拼接，此时拼接时一般有2个Op，后者即为consumer, 当前Op为producer，即当前Op为后者provide了Tensor供其使用；此时每个Op的输出tensor会被ProducerRealize(其中包含tensor对象成员)进行替代以进行进一步处理；
* Step 3: 这一步 postproc to primfunc，顾名思义是将schedule拼接完成的body进行后处理转换，从而生成primfunc的函数体，以及函数参数同函数内部语句的对应；此时会将所有语句中的Tensor替换为Buffer，ProducerRealize，ProducerLoad，ProducerStore也将被替换为相应的BufferRealize，BufferLoad，BufferStore；
* 那么为什么要进行Tensor->Buffer的转换呢？
* 对比 Tensor 和 Buffer的定义不难看出，相对于Tensor，Buffer多了许多跟内存相关的属性，如strides，elem_offset, data_alignment等等；而这些信息在schedule层级进行调度优化时并不需要，而仅在更后层进行storage flatten，inject_double_buffer等Pass内存优化Pass时才需要。这里也可以看出处于不同层级时，IR表示能力的不同，可使该层级的优化变得更加容易。值得注意的是，此时Tensor相关的Stmt/Expr节点在AST中会被全部替代。
* Step 4: Storage Flatten 的主要功能，将多维的数据访问转换为1维的数据访问，同时Buffer在这一层也会消失，后续关于内存相关的IR Node(不包含Load/Store等)仅剩Allocate Node。此Pass将在后续小节详细分析；
* Step 5: Storage Rewrite 通过 Var的生命周期进行分析，从而合并冗余的Allocate Node，尽可能增加复用已有内存。此Pass存在的原因同 2.1 小节数据流图SSA的性质高度相关，此时我们对于每个Op的Tensor(Buffer)均有对应的Allocate Stmt，其必存在大量冗余。

后续小节将对Step 3 / 4 / 5 三个Pass进行详尽的分析。

### Schedule Postproc to Primfunc
本Pass的作用在上述已经阐明，这里补充其具体实现以及需特别注意的ExternOp中对于输入输入Tensor-Buffer绑定的处理。

总体来说，TensorToBufferMapper 继承自 StmtExprMutator，其内部维护了<Tensor, Buffer\> 的buffer_map，Tensor当作字典Key时，通过其重载Operator=可以看出，两Tensor相等，当其指针相等或其为相同Operation输出；
遍历AST时，通过AttrStmt中realize_scope，此由Operation的BuildeRealize产生ProducerRealize时产生。对于realize_scope，Op会产生对应Tensor并调用GetOrAllocBuffer 在buffer_map中查询 或者 分配新的Buffer。

对于AST中，ProducerRealize/ProducerStore/ProducerLoad等节点，也通过相同的方式，即在 buffer_map中查询对应Buffer，并将当前节点替换为 BufferRealize / BufferStore / BufferLoad 等节点。
需要注意的是 ExternOp 对应 buffer_bind_scope 中对应绑定的 Buffer (Op外部传入) --- Tensor，在ScheduleOps时调用 ExternOp BuildProvide产生。此Tensor也会被替换为 buffer_map中对应 buffer 或者 分配新的Buffer。

### Storage Flatten Pass
<div align="center">
<img src=https://github.com/hehesnail/Boring_code/blob/main/imgs/storage_flatten.png/ width=70%, height=70%>
</div>

* 这个pass的主要作用是将 multi-dim 的访问 变换为 1-d buffer array的访问StorageFlattener 构造时针对 primfunc 外部传入的 buffer_map (Var-Buffer) 构造内部 buf_map_ (Buffer-BufferEntry)继承自  StmtExprMutator，重载其中对于 Stmt 以及 Expr的访问，对于primfunc中的body进行变换；

* 在经过 schedule_ops 及 schedule_postproc_to_primfunc两个pass之后，body的形式在 MakePipeline的时候在每个 OP 对应的外层部分一般为 attr_stmt realize_scope；ProducerRealizeNode，ProducerStoreNode，ProducerLoadNode变换为对应的 BufferRealize, BufferStore, BufferLoad；

* 对于 AttrStmt解析，realize_scope (每个op均有)，获取stage 所对应的 storage_scope后对body进行递归分析，因为在MakePipeline的时候，每个producer的最外层为realize_scope，因此定会对当前op中所对应的stmt body进行遍历，主要针对其中的 BufferRealizeNode；

* 对于 BufferRealizeNode，若对应的buffer在 buf_map_中找到(首次为外部buf_map)，则返回即可，因默认外部buffer 已经分配好；否则创建BufferEntry，获取对应bounds (extent)创建shape，获取storage_scope；计算buffer aligmnet信息前会先依据shape计算allocate的const_size(即shape乘积) 后获取 align对齐(默认128字节对齐)；而dim级别的align信息仅针对compute，此之前对 buffer_dim_align 的解析，可以获取到dim_info_(map)后对每一维度创建strides(其他op可为空)，依据之前信息创建新的Buffer，并添加进入 buf_map_中；并创建对应的 AllocateNode；从这个角度来看，BufferRealizeNode主要目的在于创建 整个Pipeline中的 Buffer，总感觉放在 StorageFlatten Pass中怪怪的。

* 对于 BufferLoadNode，check buffer存在后对于bounds和indices对于Index重新变换，之后主要返回Buffer vload (tir::Load)，Flatten体现，计算偏移通过 BufferOffset -> ElemOffset；同理对于BufferStoreNode，也是一样的流程，不过为 Buffer vstr::Store)，偏移计算也是通过 BufferOffset -> ElemOffset将多维访问变为一维；

* 对于VarNode，LoadNode，StoreNode则判断是否需要对其中 Var进行 remap(仅当ExternOp时需要)，不需要则直接返回即可；

* 对于 ExternOp则调用 handle_bind_scope进行处理，因为其在Provide的时候会对Op 传入的外部Buffer和该Op output tensor 和 input tensor 添加 buffer_bind_scope, 经过先前的处理, tensor已经为对应的buffer, 主要作用是先做 begin, extents的变换，后将对应buffer make Slice为对应的 shape 后调用 ArgBinder 对于将 buffer 中 Var bind 至 在buf_map中找到的 target buffer对应的 Var上，具体其实是在var_remap中添加对应 Var的替换关系后，Visit AttrStmt中的 body, 而 body中的 VarNode, LoadNode, StoreNode中对应的 Var则会被替换为对应的 target buffer 的 Var, 对 buffer_bind_scope AttrStmt处理完成之后，则清除掉对应的 var_remap关系；


***Storage Rewrite Pass***
<div align="center">
<img src=https://github.com/hehesnail/Boring_code/blob/main/imgs/storage_rewrite.png/ width=50%, height=50%>
</div>

* Function: Memory access pattern analysis and optimization, re-write data access to enable memory shareing as mush as possible.
* Rewrite calls: LinearAccessPatternFinder -> LivenessAnalysis -> PlanMemory -> PrepareNewAlloc -> rewrite by operator() -> MakeAttach
* **LinearAccessPatternFinder** ---> 主要作用是:
  * 1). 对每个storage_scope所对应的 Var 在 alloc_info_ info中添加, 并set storage scope.
  * 2). 对于For/IfThenElse/Assert等stmt, thread_extent/extern_scope/virtual_thread对应的attr stmt, 调用 VisitNewScope 将其分割开, 每个VisitNewScope 会在 linear_seq_ 添加对应的 before_scope, after_scope, 而该 scope 中的分析递归 visit op即可, 因此会有 nested_scope的情形出现；每次进入会在 scope_ 中 push StmtEntry, 而 after_scope后则 pop, 借用 scope_ stack 可以获取到当前的 Stmt Node处于 哪个 scope 中, 使用 scope_level 来标识. 每次的 touched vars 仅在 after_scope时候更新，因此 touched size 不为 0的 offset均为负.
  * 3). 那么对于 Allocate Stmt, 获取对应 buffer_var 的 AllocEntry, 更新对应 Allocate Stmt以及 scope_level 为当前 Allocate Stmt所在位置.
  * 4). 对于其他会使用到 Var的 Stmt (Store, Load, Var), 在alloc_info中获取 Var 对应的 Alloc Stmt所在的 scope_level, 并在对应 level scope_ 中添加该 Var 为 touched, 即在此level的 scope中会有这些 Var访问到Allocate所分配的buffer_var.
  * 5). 两个重要的数据结构: StmtEntry & AllocEntry, StmtEntry 主要标识当前 scope 对应 Stmt, scope在 linear_seq_中范围, 以及其牵扯到的 Var访问(R/W); AllocEntry 主要标记每个 Allocate 所在的 scope 层级, 以及对应 Allocate Stmt.
  * 6). 两个被后续使用成员: linear_seq_ & alloc_info_
  * linear_seq_: vector<StmtEntry\>, 其中包含了所有scope的起始和终止(bef_scope, aft_scope), 可获取每个scope中对应访问(R/W)过的所有 Vars;
  * alloc_info_: unordered_map<const VarNode*, AllocEntry\>, 可获取得到所有 Var所对应的 Alloc (stmt,scope_level, storage_scope).
  * 在这里需要注意理解 scope, scope主要在这里用以来划定 程序局部区域, 其包含了范围那些变量的信息, 可以来帮忙界定对应 alloc 的生存范围, 比如若存在某个变量在此区域后就未被使用了, 那么其可在该 scope被 kill, 因此后续则会被添加到 scope 对应的 kill vars中; 若存在某个变量在该区域第一次使用, 那么其应在这里被 gen, 也就是分配内存的操作直到这里才开始进行; 而 linear_seq_ 中 scope 往往是 nested scope, 对应 offset很可能出先类似 [8, [3, [1, -1], -3], [3, [1, -1], -3], -8], 注意这里的 从 + 进后 从对应 - 出的时候,  若有 kill vars, 则 free掉; 在后续和该 scope 同级的 scope 就有复用 被释放掉 vars的机会 (storage rewrite 的主要目的).

* **LivenessAnalysis**:  find gen and kill point of each variable by filling the event_map_ ---> unordered_map<const Object*, EventEntry\>
  * EnventEntry: vector<cosnt VarNode*\> gen, vector<const VarNode*\> kill  
  * kill point: 逆序遍历 linear_seq_, 即从最后的 nested scope开始向前, 即从最后使用 touched vars的scope开始, 此即为其中scope touch vars的生命周期, 故 对应Var 因在之后被kill. 每次在 event_map_ 中 对scope stmt 对应的 EnventEntry的 kill 中添加 touched vars (not visited yet).
  * gen point: 顺序遍历 linear_seq_, 从最初的nested_scope 开始，即最先使用该 Var的 scope 开始, 故对应 Var因在此前被 gen. 每次在event_map_ 中对 scope stmt 对应的 EventEntry 的 gen中添加 touch vars (not visited yet).
* **PlanMemory**: Utilize linear_seq_ and alloc_info_ to do memory planning
  *  PlanMemory所使用信息:
     * linear_seq_ ---> scope 对应 stmt 及 scope 中 touched vars;
     * alloc_info_ ---> VarNode 所对应 Alloc Stmt, storage_scope and scope_level;
     * event_map_ ---> 每个 scope 对应 stmt 所包含的 gen vars & kill vars;
  * ```c++
    for (i = 0; i < seq.size(); i++)
        find seq[i].stmt in event_map_
            // begin scope process, offset >= 0
            // stmt denotes seq[i].stmt for simplicity
            if found && offset >= 0
                for (var : stmt related gen vars)
                    detect inplace if only gen < 2
                        for (var : stmt related kill)
                            if exist gen var in kill vars (may exist in place)
                                InplaceOpVerifier
                                if pass inplace check
                                    StorageEntry of this dst_entry(gen) is src_entry(kill)
                    dst_entry = FindAlloc(gen var related alloc ...)
                    dst_entry->allocs add related alloc
                    add <var, dst_entry> in alloc_map_
            // enter/exit new_scope
            if AttrStmt
                if thread_extent || virtual_thread
                    PlanNewScope -> update thread_scope_ to op
            else if For
                if Parallel type For
                    PlanNewScope -> udpate thread_scope_ to op
            // exit the scope, we can free the stmt related kill vars
            if found && offset <= 0
                for (var : stmt reated kill vars)
                    Free(var) if not inplace substitute
    ```

 * **PlanMemory**
   * 1).首先对event_map_中对应的**gen vars**(此时外层先被access)进行**内存分配**即创建对应的 StorageEntry 并添加进入 alloc_map_中; 分配前做 InplaceOp检测，主要会借助 InplaceOpVerifier 检测 inplace 操作，若是对var进行合并，复用之前 src的 StorageEntry即可；若不是inplace，则调用 FindAlloc 创建新的 StorageEntry 并进行内存分配；之后将对应分配结果添加进入 alloc_map_中.
   * 2). 之后判断是否 enter/exit new_scope，这个主要跟 thread强相关，在 AttrStmt为 thread_extent 和 virtual_thread的时候会对 thread_scope_ 更新，在 For 为 parallel(即做了 parallel) 优化后的也会更新 thread_scope_.
   * 3). 最后对 event_map_中对应 **kill vars**, offset为负(内层先被access, also free first). 针对对应的 kill var，若其没有被 inpalce操作，则 Free, 其实是在const_free_map_ & sym_free_list_中添加 var 对应 StorageEntry，从而被之后的 FindAlloc时可复用. 比如 [8, [3, [1, -1], -3], [3, [1, -1], -3], -8], 进行是否可 free 的判断顺序会为, -1, -3, -1, -3, -8; 可以看出 nested_scope 中 free 后的 var 的复用也是同 free scope 所同级别的 scope, 不被 free 当前所在的scope所包含在内;
 * **PlanMemory -> InplaceOpVerifier**
   * 主要验证刚被 kill 的 var(src) 是否 可被下一个 gen 的 var(dst)所复用(也就是当前 scope 存在gen var后 立马释放 对应var, 那么对应的op很有可能就是 inplace opeartion); 这个 Verifier主要验证  dst[index\] = f(src[index\]), 具体判断规则看代码;
 * **PlanMemory -> FindAlloc**  
   * 在没有 inplace 复用的情况下, 调用 FindAlloc 进行 StorageEntry 的分配;
   * FindAlloc/Free 和 linear_seq_中scope 对应 event_map_中 gen/kill vars进行联动易求完成对前一 scope free的 vars 的复用;
   * Free 的 Vars 会首先找到 alloc_map_ 中对应 StorageEntry, 并根据alloc const_n_bits 是否为 0 将对应 StorageEntry 添加进入 const_free_map(const size) 及 sym_free_list(dynamic size)中;
   * 若在 const_free_map / sym_free_list 中找到满足条件的 StorageEntry, 则复用对应的 StorageEntry, 且返回后在对应的牵扯 allocs中添加到该 var对应 alloc, 即该alloc会被合并; 没找到就很直接，NewAlloc就行了, NewAlloc create 的 new StorageEntry 会被添加进 alloc_vecs_中;
   * 此时的 attach_scope_ 为外部传入, 而该 attach_scope 为 thread_scope_ 尽在 PlanNewScope的时候会被更新;
 * **PlanMemory -> PlanNewScope**: 这个还需更深入理解下 thread_scope_ 切换的时机, 目前来看 AttrStmt(thread_extent/virtual_thread) / For(parallel) 层级的 scope 的 gen vars是在这之前完成的, gen vars成功分配 StorageEntry之后才会进入这个 PlanNewScope, 所以 update thread_scope_ 之后 gen vars 对应的 scope 应包含在当前这个 scope之内, 这之内的 gen vars就希望进行局部 alloc/free, 且不可被其他 var复用分配内存;
 * **PrepareNewAlloc** -> 在上述 alloc_map_, alloc_vecs_这些准备好了之后 fill attach_map_
   * 首先将 alloc_vecs中所有的 StorageEntry 中 <attach_scope_, StorageEntry*\>作为 kv 对添加进入 attach_map_中
   * 遍历 attach_map_ 中 kv对, 遍历同一 key (attach_scope) 中所有 StorageEntry 更新其中 new_alloc 为新创建的 Allocate; 这当中会根据 StorageEntry中 allocs的个数判断是否进行 allocate合并. allocs.size() > 0 就是有多个 alloc 共享这个 StorageEntry 此时新创建 Allocate 分配大小为 几个 alloc中最大; 若为0, 直接create Allocate即可.
 * **Rewrite**: 使用 attach_map_ 对 当前 stmt进行 rewrite, 因在之前的处理当中, inplace op被合并, 可被复用的内存已被复用; 针对 thread 的 thread_extent, virtual_thread, For parallel, 对应的 Alloc 被 attach 到 当前的 op (AttrStmt, For op); 因此, 此时 attach_map_ 含有 nullptr 对应 global allocs, others attached op 对应的 local allocs. 那么当前的策略是:
   * 1). 对 thread 相关的 thread_extent, virtual_thread, For, 若 attach 到对应 op, 则 MakeAttach 拼接 该 attach_scope 对应的 所有内存分配 StorageEntry对应 alloc 和 op body; 原地 alloc, 且 alloc 对应的 body为该 op body, 此时其作用范围仅限 op_body 对应部分, 超出即释放.
   * 2). 对于其他 Stmt Node, 如 StoreNode, LoadNode, VarNode, CallNode 等，则在 alloc_map_中查询对应新 alloc 的 buffer_var 并更新其中 var的部分后 return modified stmt 即可. 而 attach_scope 为 nullptr 的 global 内存分配 (StorageEntry), 会在所有 stmt更新完成之后, 调用 MakeAttach 拼接 该 attach_scope(global) 对应的所有 allocs 和 stmt, 即此时会将这部分 alloc提升至全局.

## 总结
针对TVM Tensor的全生命周期进行分析，从计算图创建，到最后进行冗余Buffer合并的Storage Rewrite Pass进行了主要思想阐述以及详细代码分析。TVM Tensor的管理机制同其计算图设计理念高度耦合，牵扯Pass较多，从而学习曲线较为陡峭。









































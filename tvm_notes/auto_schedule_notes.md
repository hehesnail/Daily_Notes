### Ansor reading notes
* **Contributions**:
    * A mechanism to generate a **large hierarchical search space** of tensor programs for a **computational graph**.
    * An **evolutionary strategy** with a learned cost model to fine-tune the performance of tensor programs.
    * A **scheduling algorithm** based on gradient descent to **prioritize important subgraphs** when optimizing the end-to-end performance of DNNs.
    * An implementation and comprehensive evaluation of the Ansor system demonstrating that the above techniques outperform state-of-the-art systems on a variety of DNNs and hardware platforms.  
* **Previous work**:
    * **Templated based search**: require the user to write the template for several parameters first, autotvm in tvm. Also, limited to single operator optmization.
    * **Sequential construction based search**: sequentially unfold each node in the computational graph based on decision making, select the top-k candidates via learned cost model, others are pruned(incomplete search space). The auto-scheduler in Halide. The cost model estimateing the perf of incomplete program is difficult.
    * **hierarchical approach**: this paper.
* **Framwork design**:
    * 3 major components: (1) a program sampler that constructs a large search space and samples diverse programs from it; (2) a performance tuner that fine-tunes the performance of sampled programs; (3) a task scheduler that allocates time resources for optimizing multiple subgraphs in the DNNs.
    * **Program Sampler**: two levels: sketchs -> high-level structure of program, annotations -> low-levle details of program. 
    * **Performance tuner**: At each iteration, Ansor uses re-sampled new programs as well as good programs from previous iterations as the initial population to start the evolutionary search.
    * **Task Scheduler**: allocate time resource for searching for each subgraph extracted via relay program, not the end-to-end search for the whole graph.
* **Program Sampling**:
    * **Sketch generation**: Based on the derivation rules(six rules in paper) to iteratively apply node from end to the first (topological order of DAG).
    * **Random Annotation**: Randomly sample the details of the program based on the generated sketch, including: tile sizes and loop annotations, such as parallel, unroll, and vectorization.
* **Perforamce Tuning**:
    * **Evolutionary Search**:  starts from the sampled initial generation. **mutations**: tile size, parallel, pragma, computation location. Along with the mutation, contains **node-level crossover**.
    * **Cost Model**: Based on xgboost, to use the cost model select the candidates programs first and then evaluate these programs on the real machine to update the parameters of the cost model. In this paper, use the L2 loss to train the boosting tree.
* **Task scheduler**: the schedule algorithm based on gradient-descent to allocate time resources to sub-graphs of the DNN, i.e., tasks.
* **Limitations**:
    * ***Dynamic shape supprot***.
    * ***Only works well on dense operators, fail on sparse operator like sparse matrix multiplication and GCN.***
    * ***Perform optimization on high level, rely the LLVM & NVCC to do machine-depedent optimizations.***
    * ***Short usage of the special instructions(tensorcore, arm).***, this may due to the weakness of the current tensorization way to utilize the special instruction ? 
    * ***Combination of ansor and graph-level optimization***, i.e., the end-to-end optimization for the whole network graph.


### Source code analysis
* **workload_registry.py**: Workload registration and serialization.
    * WORKLOAD_FUNC_REGISTRY = {} -> Global workload function and hash key registry; 1). user registerd task via decorator **register_workload**. 2). extract tasks from relay program via function **register_workload_tensors**. **register_workload** will register the function_name & func_pointer to the WORKLOAD_FUNC_REGISTRY.
* **search_task.py**: create the SearchTask object via registered func, arguments of func(static size, no dynamic shape), target. Contains the computation information and hardware parameters for a schedule search task.
    * _ffi.register_object int the cxx side, auto_scheduler.SearchTask.
    * Create the search task either via function or via workload_key.
    * For function input, call the **make_workload** to serialize the function_name with args to workload_key via json dump. Create **ComputeDAG** object via workload_key and utilize the default layout_rewrite_options. Finally, call the cxx constructor to construct the object.
    * The **HardwareParams** object and **TuningOptions** object just call the constructor in cxx side.
    * **tune** func create the search_policy if input one is None. search policy firt construct **XGBModel** cost model and then create **SketchPolicy** object. Finally, can the **_ffi_api.AutoSchedule** in cxx size via params(search_policy and tuning_options).
* **compute_dag.py**: auto-scheduler's computational graph and related program analyses.
    * **ComputeDAG** object create, input to cxx side is the compute_tensors(out_tensor ReadGraph postorder visit) and the sch(None). Or compute(None), the (tvm.te.schedule) is not none.
    *  Currently omit the detals of the such as the loop_state related functions.
    *  Function: It keeps the input/output tensors, all operations in the DAG, and some static analysis results for the DAG (e.g. the total float operation count, consumer/producer relations of operations, whether an operation stage should be tiled/compute inlined ...). These analyses can help the search policy to make decisions during the search.
* **search_task.h/search_task.cc**
    * In search_task.h, define the class HardwareParams, HardwareParamsNode, SearchTask, SearchTaskNode. just same members consistent with python side.
    * search_task.cc, just constructor method and register the method. The **GetDefaultHardwareParams** decrible the how to set default params in different targets.
* **compute_dag.h/compute_dag.cc(basic)**
    * Members of ComputeDAGNode: Array<te::Tensor> tensors(only input/output); Array<te::Operation> ops(all ops in the schedule); double flop_ct; State init_state; AccessAnalyzer access_analyzer(do the static analysis);
    * Brief overview the ComputeDAGNode, can find the auto-schedule do lots of analysis on the compute_dag.
    * Create object: from in/out tensors, topo order -> te.create_schedule. from sch, obtain placeholders and the stage op marked as the output.
* **auto_schedule.h/auto_schedule.cc**
    * **TuningOptionsNode** definition, and the interface for python to call the AutoSchedule.
    * The AutoSchedule function create the **ProgramMeasurer** based on the tuning_options. Then call **SearchPolicy** search method with tunining options and created measurer. If the loop_state ret is valid, apply the transform_steps of the loop_state and ret the te::Schedule. The input SearchPolicy is created in python side via the SketchPolicy.
* **search_policy.py**
    * the python side actually is just interface to call the cxx side files, also provide several funcs for debuging and testing.
    * SearchPolicy -> EmptyPolicy, SearchPolicy -> SketchPolicy.
    * Note **PreloadCustomSketchRule** can be used to register user-defined sketch rule satisfied with requirement.
* **search_policy.h/search_policy.cc**
    * Declare the base class of search policies. **SearchPolicyNode**, the search related method is virutal to be overloaded. The **SearchCallbackNode** can be applied on SearchPolicy object to do some extra processing for schedule search.
* **sketch_policy.h/sketch_policy.cc**
    * class **SketchPolicy**:  The search policy that searches in a hierarchical search space defined by sketches. The policy randomly samples programs from the space defined by sketches and use evolutionary search to fine-tune them.  
        ```c++
        /*! \brief The cost model to estimate the complete schedules. */
        CostModel program_cost_model;
        /*! \brief The parameters map for this search policy. */
        Map<String, ObjectRef> params;
        /*! \brief The rules to generate sketches. */
        std::vector<SketchGenerationRule*> sketch_rules;
        /*! \brief The rules to generate initial population. */
        std::vector<PopulationGenerationRule*> init_rules;
        /*! \brief The rules to mutate states in the evolutionary search. */
        std::vector<std::shared_ptr<PopulationMutationRule>> mutation_rules;
        /*! \brief Random generator. */
        std::mt19937 rand_gen;
        /*! \brief Memorize split space for Split. */
        SplitFactorizationMemo split_memo;
        ```
    * Note the these methods: **GenerateSketches**, **SampleInitPopulation**, **EvolutionarySearch** and **PickStatesWithEpsGreedy**. Also, attention to SketchPolicy constructor, the **sketch rules for cpu and gpu differs**, the cpu & gpu & mali are specialized.
        ```c++
        // CPU task 
        node->sketch_rules.push_back(&rule_always_inline);
        node->sketch_rules.push_back(&rule_simplify_compute_with_const_tensor);
        node->sketch_rules.push_back(&rule_add_rfactor);
        node->sketch_rules.push_back(&rule_add_cache_write_stage);
        node->sketch_rules.push_back(&rule_multi_level_tiling_with_fusion);
        node->sketch_rules.push_back(&rule_multi_level_tiling);
        node->sketch_rules.push_back(&rule_skip_stage);
        // GPU(cuda)
        node->sketch_rules.push_back(&rule_add_cache_read_stage);
        node->sketch_rules.push_back(&rule_special_compute_location_gpu);
        node->sketch_rules.push_back(&rule_always_inline);
        node->sketch_rules.push_back(&rule_simplify_compute_with_const_tensor);
        node->sketch_rules.push_back(&rule_cross_thread_reduction);
        node->sketch_rules.push_back(&rule_add_cache_write_stage);
        node->sketch_rules.push_back(&rule_multi_level_tiling_with_fusion);
        node->sketch_rules.push_back(&rule_multi_level_tiling);
        node->sketch_rules.push_back(&rule_skip_stage);
        ```  
    * The details of the **sketch generation rules** and **init population rules** are defined in sketch_policy.cc. The base classes are **SketchGenerationRule** and **PopulationGenerationRule** respectively. 
        ```c++            
        /********** Sketch generation rules **********/
        static RuleSkipStage rule_skip_stage;
        static RuleAlwaysInline rule_always_inline;
        static RuleMultiLevelTiling rule_multi_level_tiling;
        static RuleMultiLevelTilingWithFusion rule_multi_level_tiling_with_fusion;
        static RuleAddCacheRead rule_add_cache_read_stage;
        static RuleAddCacheWrite rule_add_cache_write_stage;
        static RuleAddRfactor rule_add_rfactor;
        static RuleCrossThreadReduction rule_cross_thread_reduction;
        static RuleSimplifyComputeWithConstTensor rule_simplify_compute_with_const_tensor;
        static RuleSpecialComputeLocationGPU rule_special_compute_location_gpu;

        /********** Init population rules **********/
        static InitFillTileSize init_fill_tile_size;
        static InitChangeComputeLocation init_change_compute_location;
        static InitParallel init_parallel;
        static InitUnroll init_unroll;
        static InitVectorization init_vectorization;
        static InitThreadBind init_thread_bind;
        ``` 
    * The **Search** method of sketchpolicy show the workflow of how the ansor works. 
        * Train cost model(not first round)
        * SearchOneRound 
        * InferBound of the best_states and random_states 
        * PickStatesWithEpsGreedy(typically, the num equals to the num_measures_per_round), i.e., candidates selection 
        * Meaure the selected candidate states(loop state) 
        * continue to next round... 
    * For **SearchOneRound**: 
       * **GenerateSketches** 
       * **SampleInitPopulation** 
       * **EvolutionarySearch**, here insert the preivous measured good states to init_population controled by para: sample_init_use_measured_ratio.
    * **GenerateSketches**: generate the sketches based on the sketch rules of the search policy. 
       * start with the init_state <-> stage.size()-1(stage_id), push the state to currently working Array\<State\>
       * for the stage in array, try all sketch_rules, if **MeetCondition** ret is not skip cond, note the **stage_id** indicates the position of the rule apply at. 
       * Generally, the **stage_id** decreases when one rule applied on the state, but schedule primitive like cache_read/cache_write will add stage in the computeDAG, thus the stage_id remains. 
       * Also, some rules(inline, tiling) will change the loop state(CopyOnWrite way), thus state may changes during the process. 
       * The **order** in the sketch rules directly influences the sketch generation, since one rule can affect the condition checking for other rules. 
    * **GenerateSketches** based on the **SketchGenerationRule**, the two key methods are: **MeetCondition**, **Apply**.
       * the **Apply** method will do the primitives on the **State** and return the **State** with position id of the stage, the **State** schedule primitive do the **CopyOnWrite** to change the current State by add **transform_steps**, the added step contains the stage_id to indicate in which stage.
    * **SampleInitPopulation**: to generate sample_init_min_pop out_states
       * for the init_rules, **randomly** choose a **sketch**, and apply init_rule with **random factors**. 
       * Then filter the invalid generated states, call the **program_cost_model->Predict** to select candidate states to ret out_states.
    * The **PopulationGenerationRule** key is **Apply** method, typically, use **CopyOnWrite** way to rewrite the loop state, the goal is to generate **random factors** for each specialized init_rule.
    * **EvolutionarySearch**:  
        * Use **heap** to keep the best states in the search process, the compare obj is the score returned by the program_cose_model. 
        * For all mutation rules, based on rule_weights, **ComputePrefixSumProb** to get the prob for applying each mutation rule.
        * Start iterately search
        * First, prune the invalid state, use program_cost_model predict the init/prev_population states. Based on the init states and scores, construct the heap. If the heap is not empty, update the heap with previous searched states.
        * Also, **ComputePrefixSumProb** for population scores to get the pop_selection_prob (state).
        * **NodeCrossOver not supported now**
        * Do mutation, randomly choose the state, and then randomly select the mutation rule to apply rule on the state (if uniform_dist < mutation_prob). Until the population size reach the defined param.
        * Continue search.
        * Sort the heap, add state to best_states and ret. 
    * **PickStatesWithEpsGreedy**: for simply, when inputs < num_good(since will have some random states, **eps_greedy** param), pick best_states first, otherwise pick the random_states first. Then add the picked states(candidates) in **measured_states_set_** and **measured_states_vector_** for next round search won't re-pick again.
* **loop_state.h/loop_state.cc/transform_step.h/transform_step.cc**
    * **StageNode** class: lightweight state in tvm/te/schedule, the members are listed as follows: 
        ```c++
        class StageNode : public Object {
        public:
        /*! \brief The operator of this stage */
        te::Operation op;
        /*! \brief The iterators in this stage. */
        Array<Iterator> iters;
        /*! \brief The type of this stage. */
        StageKind op_type;
        /*! \brief The compute location of this stage. */
        ComputeAtKind compute_at;
        /*! \brief Other stage-level attributes. */
        StageAttributes attrs;
        ```
    * **StageKey** -> int, i.e., stage_id to represent the stage;
    * **IterKey** -> pair\<int, int\>, i.e., stage_id & iter_id to represent a iterator;
    * **AttachMapNode**: stores the compute_at relation between stages
        ```c++
        /*! \brief A Map to store the mapping of stage to its attached iterator. */
        std::unordered_map<StageKey, IterKey> stage_to_attach_iter;
        /*! \brief A Map to store the mapping of iterator to the stages attached to it. */
        std::unordered_map<IterKey, std::vector<StageKey>, IterKeyHash> iter_to_attached_stages;
        ``` 
    * **StateNode** ---> the state in search process, consists of current loop structure and list of transformation steps, each state corresponds to a specific schedule for its ComputeDAG. The state similar to the schedule in tvm::te::Schedule.
        ```c++
        class StateNode : public Object {
        public:
        /*! \brief Current stages and loop structures. */
        Array<Stage> stages;
        /*! \brief History transformation steps. */
        Array<Step> transform_steps;
        /*!
        * \brief The attach relations of stages and iterators. This is used to track the compute at operation.
        */
        AttachMap attach_map;
        /*! \brief The up-to-date ComputeDAG of this state.
        * (e.g., CacheReadStep/CacheWriteStep) can modify the ComputeDAG.
        */
        Optional<ObjectRef> current_compute_dag;
        /*!
        * \brief Indicate whether this state has unfilled tile sizes. Only concrete state can be apply to TVM schedule.
        */
        bool concrete;
        ``` 
    * **State** ref support schedule primitives: bind, parallel, unroll, vectorize, fuse, pragma, reorder, split, storage_align. new two: follow_split, follow_fused_split, these two use split factors from previous steps; compute_at, compute_inline, compute_root; cache_read, cache_write, rfactor. Use stage_id to get the stage to be applied on.
    * **State** construct from the **Ops(tvm::te::Operation)**, the **stages** added to the StateNode is auto_schedule defined **Stage**.
    * The impl of the schedule primitives for the State:
        * **General Process**: most primitives share the same impl.
            * 1). get the Stage via stage_id and the related iter ids
            * 2). CopyOnWrite to add the step in transform_steps
            * 3). call the step->ApplyToState(this) to change the related stage in the State
        * **bind, parallel, unroll, vectorize** almost same impl ---> **AnnotationStep**
            * AnnotationStep with the right annotation_type
            * For 3). AnnotationStep change the Iterator annotation.
        * **fuse**: In ApplyToState, the FuseStep fuse the iter extents and add the new_iter to the stage, also update the attach_map since iters change.
        * **pragam**: currently only support the debug_skip_region and auto_unroll_max_step, simly use CopyOnWrite of stage and change the attrs.
        * **reorder**: change the order of Iterator, update the state stages.
        * **split**: split the iterator of the stage via iter_id and stage_id. iteratively calculate the range and add the new iterators. change the previous stage iters and update the attach_map.
        * **follow_split&follow_fused_split**: call ExtractSplitLengths to get the previous split params, and the do split.
        * **storage_align**: set the storage_align attr of the stage.
        * **compute_at**: remove bound info of the stage via change iterator bound to Range(). change the stage to **kIter** in state stages. update the attach map.
        * **compute_inline**: change the stage to **kInlined** in state stages.
        * **compute_root**: compare with compute_at, difference is change the stage to **kRoot**.
        * **cache_read&cache_write**: call the **ReplayAndGetDAG** of compute_dag to get the updated compute_dag. update the stages of stage via inserting the new stage(cache_read/cache_write). update the attach_map and the current_compute_dag of the state.
* **sketch_policy_rules.h/sketch_policy_rules.cc**
    * **SketchGenerationRule**: enum class ConditionKind, MeetCondition, Apply, GetRuleName.
    * **RuleSkipStage**: 
      * **MeetCondition** always ret the kApply. 
      * **Apply** decreases the stage_id and ret.
    * **RuleAlwaysInline**: 
      * **MeetCondtion** ret kApplyAndSkipRest if cond statisfied, else kSkip. call ShouldAlwaysBeInlined(placeholder, output, has_reduction -> false, then if gpu_task -> true, if cpu_task, IsStrictlyInlineable from compute_dag analysis). 
      * **Apply** can state compute_inline on stage_id, decreases the stage_id and ret.
    * **RuleMultiLevelTiling**: 
      * **MeetCondtion** ret kApplyAndSkipRest if cond statisfied, else kSkip. call **NeedMultiLevelTiling**, the cond is determined by compute_dag analysis. 
      * **Apply**: gpu_task, tile structure: SSSRRSRS, cpu_task, tile structure: SSRSRS. call **DoMultiLevelTiling**, based on IteratorKind and tile structure, split the iterator and add to space_level and reduce_level, finaly combine space_level and reduce_level and reorder. decreases the stage_id and ret.
    * **RuleMultiLevelTilingWithFusion**:
      * **MeetCondition**: call **NeedMultiLevelTiling** && **HasSingleElementwiseMathcedConsumer**, then for gpu_task, or for cache_write stage -> always fusion, ret kApplyAndSkipRest, else, ret kApply. for else case, kSkip.
      * **Apply**: for stage_id, DoMultiLevelTiling, the for the target stage_id, tile target_stage, finally call the compute_at to fuse current stage and the consumer stage.
    * **RuleAddCacheRead**;
      * **MeetCondition**: four rules: 1). Don't cache_read a stage if it has multiple consumers; 2). Don't cache_read a stage if its consumer does not need multi-level tiling; 3). Don't cache_read a stage if its consumer does cross-thread reduction; Only direct producers can be cache read; ret kSkip or kApplyAndSkipRest(if cond meets).
      * **Apply**: add cache_read stage, and compute_at cache_read stage to the target_stage(consumer). ret state, stage_id not changes, since add one new stage. 
    * **RuleAddCacheWrite**:
      * **MeetCondition**: Add cache write if a stage needs multi-level tiling, but does not have a element-wise matched consumer, i.e., not conflict with RuleMultiLevelTilingWithFusion. then for gpu_task, kApplyAndSkipRest, cpu_task, just kApply. 
      * **Apply**:  call cache_write via state, memory attr: local. ret state, stage_id not changes, since add one new stage. 
    * **RuleAddRfactor**: simply skip.
    * **RuleSimplifyComputeWithConstTensor**: 
      * **MeetCondition**: need to set the attrs in te.compute for the stage. ret kApplyAndSkipRest or kSkip.
      * **Apply**: get unrolled_inner_iters with configed names, unroll indices of const tensors, then tile the space indices, reorder the iters, and ret stage while decreases the stage_id.
    * **RuleCrossThreadReduction**: simply skip.
    * **RuleSpecialComputeLocationGPU**: simply skip now.
* **compute_dag.h/compute_dag.cc**
    * AccessAnalyzer -> the static analyzer for a ComputeDAG.
    * data members aka analysis infos:
      * OperationMap -> unordered_map\<te::Operation, T, ObjectPtrHash, ObjectPtrEqual\>
      * OperationMap **read_from** -> (Operation, (Operation, vector<vector<PrimExpr\>\>)), an operation to all operations it reads from, for each operation pair, use a two-dimensional array for multiple multi-dimensional accesses. the inner vector represents the indices of multi-dimensional access.
      * OperationMap **read_by** -> (Operation, (Operation, vector<vector<PrimExpr\>\>)), an operation to all operations it is read by.
      * OperationMap **num_common_outer_iterators** -> (Operation, (Operation, int)), store the number of common outer iterators for operation pairs that have read-write relations.
      * OperationMap **is_simple_access** -> (Operation, bool)
      * OperationMap **is_strictly_inlineable** -> (Operation, bool)
      * OperationMap **needs_multi_level_tiling** -> (Operation, bool)
      * OperationMap **is_output** -> (Operation, bool)
      * Array\<te\:\:Operation\> ops_topo_order -> topological order of operations.
    * provide for policys:
      * IsSimpleAccess, IsStrictlyInline, NeedsMultiLevelTiling, IsOutput, GetConsumers, GetProducers, GetDirectProducers, GetNumCommonOuterIterator, ElementWiseMatch.
    * Create the AccessAnalyzer from tensors (output_tensors):
      * 1). Topo sort the ops in graph, ops_topo_order ok.
      * 2). build the read & write access map
        * a). PlaceholderO, read_from is empty.
        * b). For ComputeOp, **ReadAccessExtractor** extract the ComputeOp body.
        * c). **ReadAccessExtractor**: data_mem: (Operation -> vector<vector<PrimExpr\>\>), for CallNode with builtin::if_then_else, SelectNode, IfThenElseNode(Stmt), has_branch -> true. If visit the ProducerLoadNode, add the PrimExpr indices to read_access map.
        * d). For **read_by** map, all loads in read_access map are read_by the current op, thus [iter.first, op\] to fill the primexpr indices. 
        * e). For **read_from** map, the read_access map is exactly the current op read_from, also set the op has_branch attr.
      * 3). compute number of common outer iterators, based on read_from map, first shape should match, then axis itervar should ConstShiftEqual to access primexpr. the **num_common_outer_iterators** of node is set.
      * 4). Do some static analysis on ComputeOps, is_simple_access, is_strictly_inlinable -> needs_multi_level_tiling.
      * 5). **is_simple_access**: get the op read_from map, iterate the access_list, call IsSimpleAccess to check the access to an operation -> based on all index is just a variable with an optional constant shift. also set the axis_missing, axis_duplicated, same_order(indices and op axes). for **is_simple_access**, determined by the func ret. 
      * 6). **is_strictly_inlineable**: is_simple_access && same_order && not axis_duplicated. also, if contain expensive op(currently only exp) and op has branch, set to false.
      * 7). **needs_multi_level_tiling**: get the op read_from map, iterate the access_list, for access expr get the indices expr. For reduce_axis itervar, or reduce_axis itervar primexpr, n_missing_cnt++. If n_missing_cnt >=2 or n_missing_cnt>=1 & has_reduce_axis. Thus, the op has reduction comp, need to do multi_level_tiling optim.
    * **GetConsumers:** Find all inlined_ops first, recursively collect all ops from read_by map, skip the inlined ops and add to consumers, i.e., all consumers along the whole compute_dag. 
    * **GetProducers & GetDirectProducers:** GetProducers work likes GetConsumers, but from the read_from map. GetDirectProducers just obtain the ops from read_from map, no recursive call. 
    * **ElementWiseMatch:** For op and target_op, along from the compute_dag, i.e., from read_by map. 1). The read_by map of op must only contain one consumer op; 2). Also, thus two ops should have the same output size; 3). Finally, the read(op->vector<vector<PrimExpr\>\> indices) is elmentwise(IsSimpleAccess succeed)
* **cost_model.h/cost_model.cc/cost_model.py/xgb_model.py**
    * Base CostModelNode, the virtual methods: Update/Predict/PredictStages.
        ```c++
        class CostModelNode : public Object {
        public:
        virtual void Update(const Array<MeasureInput>& inputs, 
                            const Array<MeasureResult>& results) = 0;
        virtual void Predict(const SearchTask& task, 
                             const Array<State>& states,
                             std::vector<float>* scores) = 0;
        virtual void PredictStages(const SearchTask& task, 
                                  const Array<State>& states, 
                                  std::vector<float>* state_scores,
                                  std::vector<std::vector<float>>* stage_scores);
        ```
    * Two cost models, **RandomModel**, **PythonBasedModel**
    * For RandomModel, inherits from CostModel
        ```c++
        class RandomModelNode : public CostModelNode {
        public:
        /*! \brief Pointer to a random number generator function */
        const TypedPackedFunc<void(size_t, void*)>* random_number_func;
        ...
        }; 
        ```
        * The key thing here is the random_number_func(TypePackedFunc), this will be passed from python side.
        * For override **Update** method, do nothing, since random model dose not need update params.
        * For override **Predict** method, call the random_number_func, this func is random_fill_float in cost_mode.py, just ret np.random.uniform(0, 1, (size,)), uniform distribution.
    * For **PythonBasedModel**, inherits from CostModel, currently support XGBModel(def in python side).
        ```c++
        class PythonBasedModelNode : public CostModelNode {
        public:
        /*! \brief Pointer to the update funcion in python */
        PackedFunc update_func;
        /*! \brief Pointer to the predict funcion in python */
        PackedFunc predict_func;
        /*! \brief Pointer to the predict funcion in python */
        PackedFunc predict_stage_func;
        };
        ```
    * 3 key things, update_func, predict_func, predict_stage_func, all PakcedFunc, thus call the python side corresponding func.
    * For both **Update** and **Predict**, call the corresponding member function.
    * Lets look at **XGBModel** defined in **xgb_model.py** and **PythonBasedModel** in python side.
    * in __init\_\_,  call self.\_\_init_handle_by_constructor\_\_(_ffi_api.PythonBasedModel, update_func, predict_func, predict_stage_func), also the udpate_func and predict_func will can self.update, self.predict, i.e., defined in python side.
    * In **XGBModel**, import the xgboost lib for train and predict.
        ```python
        global xgb
        try:
            if xgb is None:
                xgb = __import__("xgboost")
        ``` 
    * For **update**: call **get_per_store_features_from_measure_pairs** to extract features from inputs/results, this func call the cxx side. Then train xgboost model and save it. 
        ```python
        self.bst = xgb.train(
            self.xgb_params,
            dtrain,
            num_boost_round=10000,
            obj=pack_sum_square_error,
            callbacks=[
                custom_callback(
                    stopping_rounds=50,
                    metric="tr-p-rmse",
                    fevals=[
                        pack_sum_rmse,
                        pack_sum_average_peak_score(self.plan_size),
                    ],
                    evals=[(dtrain, "tr")],
                    maximize=False,
                    verbose_eval=self.verbose_eval,
                )
            ],
        )
        ``` 
    * For **predict**, same way to extract features, pre-process the features and call xgboost model, ret the predicted scores.
        ```python
        features = get_per_store_features_from_states(states, task)
        if self.bst is not None and len(self.inputs) > self.num_warmup_sample:
            dtest, pack_ids = feature_to_pack_sum_xgbmatrix(features)
            raw_preds = self.bst.predict(dtest)
            ret = predict_throughput_pack_sum(raw_preds, pack_ids)
        else:
            ret = np.random.uniform(0, 1, (len(states),))

        # Predict -inf for invalid states that failed to be lowered.
        for idx, feature in enumerate(features):
            if feature.min() == feature.max() == 0:
                ret[idx] = float("-inf")

        return ret
        ```  
    * Just skip the features packing, convert to xgbmatrix etc.
* **feature.h/feature.cc**
    * **feature.h**: defs **GetPerStoreFeaturesFromStates**, **GetPerStoreFeature**, **GetPerStoreFeaturesFromMeasurePairs**, the python side mainly calls these methods to obtain the features as describled above.
    * **feature.cc**: the features def shown below.
      * **BufferAccessFeature**:
        ```c++
        // Feature for an access of a buffer
        struct BufferAccessFeature {
        std::string buffer_name;        // The name of the buffer
        BufferAccessType acc_type;      // The type of the access
        float bytes;                    // The touched memory in bytes
        float unique_bytes;             // The touched unique memory in bytes
        float lines;                    // The number of touched cache lines
        float unique_lines;             // The number touched unique cache lines
        ReuseType reuse_type;           // Tye type of data reuse
        float reuse_dis_iter;           // The reuse distance in iterator number
        float reuse_dis_bytes;          // The reuse distance in total touched bytes
        float reuse_ct;                 // The reuse ratio
        float bytes_d_reuse_ct;         // bytes / reuse_ct
        float unique_bytes_d_reuse_ct;  // unique_bytes / reuse_ct
        float lines_d_reuse_ct;         // lines / reuse_ct
        float unique_lines_d_reuse_ct;  // unique_lines / reuse_ct
        float stride;                   // The stride in access
        };
        ```
      * **FeatureSet**:
        ```c++
        // Feature set of a BufferStore statement
        struct FeatureSet {
        // Group 1: Computation related features
        float float_mad;                  // The number of float MAD (Multiply–add) ops
        float float_addsub;               // The number of float add and sub ops
        float float_mul;                  // The number of float multiply ops
        float float_divmod;               // The number of float div and mod ops
        float float_cmp;                  // The number of float comparison ops
        float float_math_func;            // The number of float math func calls
        float float_other_func;           // The number of other float func calls
        float int_mad;                    // The number of integer MAD (Multiply–add) ops
        float int_addsub;                 // The number of integer add and sub ops
        float int_mul;                    // The number of float multiply ops
        float int_divmod;                 // The number of float div and mod ops
        float int_cmp;                    // The number of float comparison ops
        float int_math_func;              // The number of float math func calls
        float int_other_func;             // The number of other float func calls
        float bool_op;                    // The number of bool ops
        float select_op;                  // The number of select ops
        float vec_num;                    // The number of vectorized iterators
        float vec_prod;                   // The product of the lengths of vectorized iterators
        float vec_len;                    // The length of the innermost vectorized iterator
        AnnotationPosType vec_type;       // The type of vectorization position
        float unroll_num;                 // The number of unrolled iterators
        float unroll_prod;                // The product of the lengths of vectorized iterators
        float unroll_len;                 // The length of the innermost unrolled iterator
        AnnotationPosType unroll_type;    // The type of unroll position
        float parallel_num;               // The number of paralleled iterators
        float parallel_prod;              // The product of the lengths of paralleled iterators
        float parallel_len;               // The length of the innermost paralleled iterators
        AnnotationPosType parallel_type;  // The type of parallel position
        float is_gpu;                     // Whether it is a GPU task
        float blockIdx_x_len;             // The length of blockIdx.x
        float blockIdx_y_len;             // The length of blockIdx.y
        float blockIdx_z_len;             // The length of blockIdx.z
        float threadIdx_x_len;            // The length of threadIdx.x
        float threadIdx_y_len;            // The length of threadIdx.y
        float threadIdx_z_len;            // The length of threadIdx.z
        float vthread_len;                // The length of virtual thread

        // Group 2: Buffer access related features (per buffer)
        std::vector<BufferAccessFeature> access_feas;

        // Group 3: Arithmetic intensity related features
        float arith_intensity_curve[ARITH_INTENSITY_CURVE_SAMPLE_N];  // points sampled from the
                                                                        // arithmetic intensity curve

        // Group 4: Allocation related features
        float alloc_size;        // The size of allocated buffer in bytes
        float alloc_outer_prod;  // The product of lengths of loops outside the scope of the allocation
        float alloc_inner_prod;  // The product of lengths of loops inside the score of the allocation
        float alloc_prod;        // alloc_outer_prod * alloc_inner_prod

        // Group 5: Outer scope related features
        float outer_prod;            // The product of lengths of outer loops
        float num_loops;             // The number of outer loops
        float auto_unroll_max_step;  // The value of pragma "auto_unroll_max_step"
        };
        ``` 
      * To exact features, call **GetPerStoreFeaturesWorkerFunc**, in this func, first construct IRModule with PrimFunc, then call **GetPerStoreFeature** for the PrimFunc body, call **PerStoreFeatureExtractor**.
      * **PerStoreFeatureExtractor**:
       * **BufferStoreNode**: most features computed from visited BufferStore Stmt.   
            MathOpCounter will cacl the number of various ops for BufferStore value.(float_addsub, float_cmp, etc.) 
            ```c++
            // Group 1: Computation related features
            ExtractComputationFeature(node, math_op_counter);
            // Group 2: Buffer access related features (per buffer)
            ExtractBufferAccessFeature(node, math_op_counter, &cur_compute_ops, &compute_ops_list,
                                    &mem_bytes_list);
            // Group 3: Arithmetic intensity related features
            ExtractArithmeticIntensityFeature(node, cur_compute_ops, compute_ops_list, mem_bytes_list);
            // Group 4: Allocation related features
            ExtractOuterScopeFeature(node);
            ```
       * **BufferRealizeNode**:
            ```c++
            // Group 5: Outer scope related features
            ExtractAllocationFeature(node);
            ```
        * After extract features, do shift log for these values, push to ret features. When all feature extraction finish, **SerializeFeatures** and back to python side to train the model.
## IOS: INTER-OPERATOR SCHEDULER FOR CNN ACCELERATION (MLSys 2021)
### Motivation
* Existing intra-operator parallelism cannot saturate modern hardware’s high parallelism, especially for recent multi-branch CNN models. Current schedule method fails to find the best schedule.
<div align="center">
<img src="https://github.com/hehesnail/Boring_code/blob/main/imgs/ios_motivation_figure.png" width="70%" height="70%" /> 
</div>

### Contributions
* Propose dynamic programming algorithm to find a highly optimized schedule for inter-operator parallelization.
* Experiments compared with various framework validate the efficiency of the proposed algorithm.

### Algorithm
<div align="center">
<img src="https://github.com/hehesnail/Boring_code/blob/main/imgs/ios_dp_algo.png" width="70%" height="70%" /> 
</div>

### Limitations & Conclusions
* Limitations: 
  * The search overhead of the proposed method is high. Without pruning strategy, it takes up to 4 hours for IOS find the optimal schedule. With pruning strategy(r=3, s=8), inception v3(4+ mins), NasNet(18+ mins). 
  * The inter operator parallelism only occurs frequentlyi in multi-branch networks, thus the paper only take these networks to demonstrate the performance. This method may not generalize well on recurrent networks.
* Conclusions:
  * A good start point for our schedule work, all graph defs and optmization codes are on python side, the only thing is to bridge the python side and runtime(or our model converter tool). 
  * Generalize the parallelization strategy to match our heterogeneous situation.
  * Careful to choose the workload for comparision, single network may not eliminate the latency incured by the memory access. 
  

## RAMMER: Enabling Holistic Deep Learning Compiler Optimizations with rTasks (OSDI 2020)
### Motivation
* Hardware-managed intra-operator scheduling leads to low GPU utilization.
* High inter-operator scheduling overheads.
* Interplay between inter- and intra-operator scheduling, the two layer schedule architecture lacks the co-optimization.
* **Key idea**: manage the scheduling of inter-intra operator together.
<div align="center">
<img src="https://github.com/hehesnail/Boring_code/blob/main/imgs/rammer_two_layer.png" width="70%" height="70%" /> 
</div>

### Contributions
* a deep learning compiler that takes a holistic approach to manage the parallelism available in the DNN computation for scheduling.
* Rammer is compatible with optimizations developed in existing DNN compilers.
* Rammer is hardware neutral.

### Method
<div align="center">
<img src="https://github.com/hehesnail/Boring_code/blob/main/imgs/rammer_arch.png" width="70%" height="70%" /> 
</div>

* rTask: minimum computation unit in an operator to be executed on a processing element of the accelerator device.
* rOperator: defined as a group of independent, homogeneous rTasks.
* rProgram: contains a piece of computation to be carried out on the hardware, mapped to the corresponding vEUs at compile time.
* vDevice: abstracts a hardware accelerator as a software-managed virtual device.
* vEU: A vDevice further presents multiple parallel virtual execution units.

### Experiments
* Hardwares: Nvidia GPU, AMD GPU, GraphCore IPU.
* Networks: ResNeXt, NasNet, AlexNet, DeepSpeech2, LSTM, Seq2Seq.
* Frameworks: Tensorflow, Tensorflow-XLA, TensorRT, TVM.
* Rammer’s benefits are more significant when the intra-operator parallelism is insufficient to saturate hardware, this is the case when the input batch size is small.

### Conclusions
* The schedule policy proposed by Rammer is trivial.
* Lacks the support for dynamic model, i.e., control flow.
* The inter-job scheduling is not take into consideration.
* Rewriting or reimplementing the operators to rOperators needs huge effort. Also, one should be very familiar with hardware apis and features to abstract the hardware to vDevice.
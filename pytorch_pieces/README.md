# Pytorch Notes 

### *2021.5.10*
* torch.jit
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


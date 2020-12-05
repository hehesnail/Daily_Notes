# Boring LLVM Code

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
* LLVM 10.0同书中示例3.4的差异还是非常大的，基本都改了一套接口，很多类的定义也完全变掉...总而言之，进行修改还是非常蛋疼的事...
  

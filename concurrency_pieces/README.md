# Boring Concurrency Code

## *2020.11.14 --- Hello Concurrency World*
* C++ 11 ---> \<thread>  
* start thread = create thread object  
* joined vs detached  
* pass params to thread func
* 1. 默认参数会拷贝，即使参数为引用，避免参数为悬垂指针
* 2. 使用std::ref转换为引用形式
* 3. 可传递成员函数指针作为线程函数，并提供对象指针作为第一个参数
* 4. 使用std::move转移动态对象到线程中去  
* 新线程所有权都要转移, thread可移动，不可拷贝

## *2020.11.16*
* thread::hardware_concurrency() 返回能同时并发在一个程序中的线程数量
* 线程标识类型, thread::id, 可通过 thread object member function get_id()获取，或者在当前线程中调用this_thread::get_id(), thread::id 可以拷贝，比较(可用作容器的键值)

## *2020.11.17*
* <mutex>头文件中，std::mutex创建互斥量，使用lock()上锁，unlock()解锁。建议使用RAII语法模板类 std::lock_guard，构造时提供已锁互斥量，析构时进行解锁。
* 切勿将受保护数据的指针或引用传递到互斥锁作用域之外，无论是 函数返回值 or 存储在外部可见内存 or 以参数形式传递到用户提供的函数中。
* 接口设计存在内在的条件竞争


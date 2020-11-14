#include <iostream>
#include <thread>
#include <vector>
#include <algorithm>
#include <functional>

using namespace std;

void some_func();
void some_other_func();

struct func {
    int &i;
    func(int &i_) : i(i_) { }
    void operator() () {
        for (unsigned j = 0; j < 10; j++)  {
            cout << "func: " << i + j << endl;
        }
    }
};

void f1() {
    thread t1(some_func);
    thread t2 = move(t1); // t1->t2, some_func
    t1 = thread(some_other_func); // t1, some_other_func
    thread t3;
    t3 = move(t2); // t2->t3, some_func
}

// 同thread_guard不同在于，scoped_thread拥有线程的所有权
class scoped_thread {
    thread t;
    public:
    explicit scoped_thread(thread t_): t(move(t_)) {
        if (!t.joinable()) {
            throw logic_error("No thread");
        }
    }

    ~scoped_thread() {
        t.join();
    }

    scoped_thread(scoped_thread const &) = delete;
    scoped_thread& operator=(scoped_thread const &) = delete;
};

void f2() {
    int some_local_state = 1024;
    scoped_thread t(thread(func(some_local_state)));
    cout << "stupid f2" << endl;
}

void do_work(unsigned id) {
    cout << "do work: " <<  id << endl;
}

void f3() {
    vector<thread> threads; // vector支持移动操作
    for (unsigned i = 0; i < 20; i++) {
        threads.push_back(thread(do_work, i)); // move constructor
    }

    for_each(threads.begin(), threads.end(), mem_fn(&thread::join)); // call joins
}

int main() {

    f1();

    f2();

    f3();

    return 0;
}
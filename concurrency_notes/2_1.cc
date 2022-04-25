#include <iostream>
#include <thread>

using namespace std;

struct func {
    int &i;
    func(int &i_) : i(i_) { }
    void operator() () {
        for (unsigned j = 0; j < 100000; j++)  {
            cout << i << endl;
        }
    }
};

// RAII, Resouce acquistion is initialization
class thread_guard {
private:
    thread &t;
public:
    explicit thread_guard(thread& t_): t(t_) { }
    ~thread_guard() {
        if (t.joinable()) {
            t.join();
        }
    }
    thread_guard(thread_guard const&) = delete;
    thread_guard& operator= (thread_guard const&) = delete;
};

void f() {
    int some_local_state = 0;
    func my_func(some_local_state);
    thread t(my_func);
    thread_guard g(t);

    cout << "stupid f~" << endl;
}

int main() {

    f();

    return 0;
}
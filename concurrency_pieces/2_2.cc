#include <iostream>
#include <thread>
#include <memory>

using namespace std;

class X {
public:
    void foo(int i) {
        cout << i << endl;
    }
};

class big_object {
private:
    int a[10000];
public:
    big_object() {
    }
    void prepare_data() {
        for (int i = 0; i < 10000; i++) {
            a[i] = i;
        }
    }
    void print_data() {
        for (int i = 0; i < 10000; i++) {
            cout << a[i] << endl;
        }
    }
};

void f1() {
    X my_x;
    int num(0);
    thread t(&X::foo, &my_x, num);
    t.detach();
}

void process_big_object(unique_ptr<big_object> p) {
    p->print_data();
}

void f2() {
    unique_ptr<big_object> p (new big_object);
    p->prepare_data();
    thread t(process_big_object, move(p));

    t.join();
}

int main() {

    f1();

    f2();

    return 0;
}
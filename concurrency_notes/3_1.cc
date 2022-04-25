#include <list>
#include <iostream>
#include <mutex>
#include <algorithm>
#include <thread>

using namespace std;

list<int> some_list;
mutex some_mutex;

void add_to_list(int new_value) {
    lock_guard<mutex> guard(some_mutex);
    some_list.push_back(new_value);
}

bool list_contains(int value_to_find) {
    bool find_s = false;
    lock_guard<mutex> guard(some_mutex);
    find_s = find(some_list.begin(), some_list.end(), value_to_find) != some_list.end();
    cout << find_s << endl;

    return find_s;
}

void f() {
    thread t1(add_to_list, 1);
    thread t2(add_to_list, 2);
    thread t3(add_to_list, 3);
    thread t4(add_to_list, 4);

    for (auto i = some_list.begin(); i != some_list.end(); i++) {
        cout << *i << ' ';
    }
    cout << endl;

    thread t5(list_contains, 5);
    thread t6(list_contains, 2);

    t1.join();
    t2.join();
    t3.join();
    t4.join();

    t5.join();
    t6.join();
}

int main() {

    f();

    return 0;
}
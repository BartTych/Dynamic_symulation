#include <pybind11/pybind11.h>

long add(int a, int b) { return a + b; }

long subtract(int a, int b) { return a - b; }

int combine(long a, long b) { return a + b; }

int multiply(int a, int b) {
    
    for (int i = 0; i < 1000000; ++i) {
        a = a + b;   // Simulating some work
    }
    
    return a; }

PYBIND11_MODULE(basic, m) {
    m.def("add", &add);
    m.def("subtract", &subtract);
    m.def("combine", &combine);
    m.def("multiply", &multiply);
}
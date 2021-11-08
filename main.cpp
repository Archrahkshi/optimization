#include <iostream>
#include <functional>
#include <chrono>
#include <omp.h>

using namespace std;
using integrand_t = const function<double(double)> &;

/// Count CPU ticks using rdtsc
inline uint64_t count_ticks() {
    uint32_t lo, hi;
    asm volatile ("rdtsc\n" : "=a" (lo), "=d" (hi));
    return ((uint64_t) hi << 32) | lo;
}

double integrate_plain(double a, double b, unsigned n, integrand_t f) {
    double dx = (b - a) / n;
    double result = 0;
    for (unsigned i = 0; i < n; ++i)
        result += f(a + i * dx);
    return result * dx;
}

double integrate_omp_simd(double a, double b, unsigned n, integrand_t f) {
    double dx = (b - a) / n;
    double result = 0;
#pragma omp simd reduction(+ : result)
    for (unsigned i = 0; i < n; ++i)
        result += f(a + i * dx);
    return result * dx;
}

double integrate_omp_parallel(double a, double b, unsigned n, integrand_t f, int num_threads) {
    double dx = (b - a) / n;
    double result = 0;
    omp_set_num_threads(num_threads);
#pragma omp parallel for reduction(+ : result) default(none) shared(a, n, f, dx)
    for (unsigned i = 0; i < n; ++i)
        result += f(a + i * dx);
    return result * dx;
}

void print_results(double result1, double result2, long num_ns, double num_ticks) {
    // Printing both result1 and result2 so that they are not optimized by GCC
    cout << "Result: " << result1 << " / " << result2 << '\n'
         << "Time: " << num_ns << " ns (" << num_ticks << " ticks)\n";
}

void measure_execution_time(
        const function<double(double, double, unsigned, integrand_t)> &integral,
        double a,
        double b,
        unsigned n,
        integrand_t f
) {
    double result1, result2;
    chrono::time_point<chrono::system_clock, chrono::nanoseconds> ns_start, ns_end;
    uint64_t ticks_start, ticks_end;

    ns_start = chrono::high_resolution_clock::now();
    result1 = integral(a, b, n, f);
    ns_end = chrono::high_resolution_clock::now();

    ticks_start = count_ticks();
    result2 = integral(a, b, n, f);
    ticks_end = count_ticks();

    print_results(result1, result2, (ns_end - ns_start) / 1ns, (double) (ticks_start - ticks_end));
}

void measure_execution_time(
        const function<double(double, double, unsigned, integrand_t, int)> &integral,
        double a,
        double b,
        unsigned n,
        integrand_t f,
        int num_threads
) {
    double result1, result2;
    chrono::time_point<chrono::system_clock, chrono::nanoseconds> ns_start, ns_end;
    uint64_t ticks_start, ticks_end;

    ns_start = chrono::high_resolution_clock::now();
    result1 = integral(a, b, n, f, num_threads);
    ns_end = chrono::high_resolution_clock::now();

    ticks_start = count_ticks();
    result2 = integral(a, b, n, f, num_threads);
    ticks_end = count_ticks();

    print_results(result1, result2, (ns_end - ns_start) / 1ns, (double) (ticks_start - ticks_end));
}

int main(int argc, char **argv) {
    if (argc != 4) {
        cout << "Enter lower bound, upper bound and precision\n";
        return EXIT_FAILURE;
    }

    double a = stod(argv[1]);
    double b = stod(argv[2]);
    unsigned n = stoi(argv[3]);
    integrand_t f = [](double x) { return x * x; };

    for (function integral: {integrate_plain, integrate_omp_simd})
        measure_execution_time(integral, a, b, n, f);

    for (int num_threads = 1; num_threads <= 6; ++num_threads)
        measure_execution_time(integrate_omp_parallel, a, b, n, f, num_threads);

    return EXIT_SUCCESS;
}

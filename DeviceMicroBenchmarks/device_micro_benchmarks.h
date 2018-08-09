#ifndef DEVICE_MICRO_BENCHMARKS_H 
#define DEVICE_MICRO_BENCHMARKS_H
#include <iostream>
#include <chrono>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <vector>
#include <random>
#include <type_traits>

// HIP API
#include <hip/hip_runtime.h>

#define HIP_CHECK(condition)         \
  {                                  \
    hipError_t error = condition;    \
    if(error != hipSuccess){         \
        std::cout << "HIP error: " << error << " line: " << __LINE__ << std::endl; \
        exit(error); \
    } \
  }

std::string create_space(std::string input, int padding = 25)
{
    std::string init = " ";
    int count = std::abs((int)(padding - input.length()));
    for(int i = 0; i < count; i++)
    {
        init.append(" ");
    }
    return init;
}

template<class T>
constexpr unsigned int megabytes(unsigned int size)
{
    return(size * (1024 * 1024 / sizeof(T)));
}

const size_t max_random_size = 1024 * 1024;

template<class T>
inline auto get_random_data(size_t size, T min, T max)
    -> typename std::enable_if<std::is_integral<T>::value, std::vector<T>>::type
{
    std::random_device rd;
    std::default_random_engine gen(rd());
    std::uniform_int_distribution<T> distribution(min, max);
    std::vector<T> data(size);
    std::generate(
        data.begin(), data.begin() + std::min(size, max_random_size),
        [&]() { return distribution(gen); }
    );
    for(size_t i = max_random_size; i < size; i += max_random_size)
    {
        std::copy_n(data.begin(), std::min(size - i, max_random_size), data.begin() + i);
    }
    return data;
}

template<class T>
inline auto get_random_data(size_t size, T min, T max)
    -> typename std::enable_if<std::is_floating_point<T>::value, std::vector<T>>::type
{
    std::random_device rd;
    std::default_random_engine gen(rd());
    std::uniform_real_distribution<T> distribution(min, max);
    std::vector<T> data(size);
    std::generate(
        data.begin(), data.begin() + std::min(size, max_random_size),
        [&]() { return distribution(gen); }
    );
    for(size_t i = max_random_size; i < size; i += max_random_size)
    {
        std::copy_n(data.begin(), std::min(size - i, max_random_size), data.begin() + i);
    }
    return data;
}
#endif 


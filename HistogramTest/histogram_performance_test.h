#ifndef HISTOGRAM_PERFORMANCE_TEST_H
#define HISTOGRAM_PERFORMANCE_TEST_H
#include <stdio.h>
#include <chrono>
#include <vector>
#include <random>
#include <type_traits>
#include <iostream>
#include <limits>
#include <algorithm>

#include <hip/hip_runtime.h>

#define OUTPUT_VALIDATION_CHECK(validation_result)               \
  {                                                              \
    if ( validation_result == false )                            \
    {                                                            \
        std::cout << "Output validation failed!" << std::endl;   \
        return;                                                  \
    }                                                            \
  }

#define HIP_CHECK(condition)         \
{                                  \
    hipError_t error = condition;    \
    if(error != hipSuccess){         \
        std::cout << "HIP error: " << error << " line: " << __LINE__ << std::endl; \
        exit(error); \
    } \
}

template<class T>
inline auto get_random_data(size_t size, T min, T max)
    -> typename std::enable_if<std::is_integral<T>::value, std::vector<T>>::type
{
    std::random_device rd;
    std::default_random_engine gen(rd());
    std::uniform_int_distribution<T> distribution(min, max);
    std::vector<T> data(size);
    std::generate(data.begin(), data.end(), [&]() { return distribution(gen); });
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
    std::generate(data.begin(), data.end(), [&]() { return distribution(gen); });
    return data;
}

template<class T>
inline void hip_read_device_memory(std::vector<T> &host_destination, T *device_source)
{
    HIP_CHECK(
        hipMemcpy(
            host_destination.data(), device_source,
            host_destination.size() * sizeof(T),
            hipMemcpyDeviceToHost
        )
    );
}

template<class T>
inline void hip_write_device_memory(T *device_destination, std::vector<T>& host_source)
{
    HIP_CHECK(
        hipMemcpy(
            device_destination, host_source.data(),
            host_source.size() * sizeof(T),
            hipMemcpyHostToDevice
        )
    );
}

template<class T>
void print_array(const std::vector<T>& host_source)
{
    for(int index = 0; index < host_source.size(); index++)
    {
        std::cout << host_source[index] << " - ";
    }
    std::cout << std::endl;
}

template<class T>
constexpr unsigned int megabytes(unsigned int size)
{
    return(size * (1024 * 1024 / sizeof(T)));
}
#endif

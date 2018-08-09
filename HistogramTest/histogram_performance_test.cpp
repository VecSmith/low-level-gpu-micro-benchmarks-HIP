#include "histogram_performance_test.h"

enum histogram_modes
{
    noop,
    naive,
    striped_read,
    histogram_per_block,
    histogram_per_block_padding,
    output_cacheline_padding,
};

const std::string histogram_function_names[6] = 
{
    std::string("no_operation"),
    std::string("naive"),
    std::string("striped_read"),
    std::string("histogram_per_block"),
    std::string("histogram_per_block_padding"),
    std::string("output_cacheline_padding"),
};

template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    class T,
    unsigned int HistogramMode
>
struct histogram_function;

template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    class T
>
struct histogram_function<BlockSize, ItemsPerThread, T, noop>{};

template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    class T
>
struct histogram_function<BlockSize, ItemsPerThread, T, naive>
{
    __device__ inline 
    void operator()(T* device_input, unsigned int* device_output, 
                    unsigned int bin_count, T max_range)
    {
        const T range_identifier = (max_range / T(bin_count));
        const unsigned int offset = (hipBlockIdx_x * BlockSize + hipThreadIdx_x) * ItemsPerThread;
        T items[ItemsPerThread];
        #pragma unroll
        for(unsigned int index = 0; index < ItemsPerThread; index++)
        {
            items[index] = device_input[offset + index];
        }

        #pragma unroll
        for(unsigned int index = 0; index < ItemsPerThread; index++)
        {
            unsigned int bin_index = items[index] / range_identifier;
            atomicInc(&device_output[bin_index], UINT_MAX);
        }
    }
};

template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    class T
>
struct histogram_function<BlockSize, ItemsPerThread, T, striped_read>
{
    __device__ inline 
    void operator()(T* device_input, unsigned int* device_output, 
                    unsigned int bin_count, T max_range)
    {
        const T range_identifier = (max_range / T(bin_count));
        const unsigned int offset = hipBlockIdx_x * BlockSize * ItemsPerThread + hipThreadIdx_x;
        T items[ItemsPerThread];
        #pragma unroll
        for(unsigned int index = 0; index < ItemsPerThread; index++)
        {
            items[index] = device_input[offset + index * BlockSize];
        }

        #pragma unroll
        for(unsigned int index = 0; index < ItemsPerThread; index++)
        {
            unsigned int bin_index = items[index] / range_identifier;
            atomicInc(&device_output[bin_index], UINT_MAX);
        }
    }
};

constexpr unsigned int cacheline_size_byte = 128;
constexpr unsigned int cacheline_padding_size = cacheline_size_byte / 4;

template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    class T
>
struct histogram_function<BlockSize, ItemsPerThread, T, output_cacheline_padding>
{
    __device__ inline 
    void operator()(T* device_input, unsigned int* device_output, 
                    unsigned int bin_count, T max_range)
    {
        const T range_identifier = (max_range / T(bin_count));
        const unsigned int offset = hipBlockIdx_x * BlockSize * ItemsPerThread + hipThreadIdx_x;
        T items[ItemsPerThread];
        #pragma unroll
        for(unsigned int index = 0; index < ItemsPerThread; index++)
        {
            items[index] = device_input[offset + index * BlockSize];
        }

        #pragma unroll
        for(unsigned int index = 0; index < ItemsPerThread; index++)
        {
            unsigned int bin_index = items[index] / range_identifier;
            atomicInc(&device_output[bin_index * cacheline_padding_size], UINT_MAX);
        }
    }
};

template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    class T
>
struct histogram_function<BlockSize, ItemsPerThread, T, histogram_per_block>
{
    __device__ inline 
    void operator()(T* device_input, unsigned int* device_output, 
                    unsigned int bin_count, T max_range)
    {
        const T range = max_range / bin_count;
        const unsigned int offset = hipBlockIdx_x * BlockSize * ItemsPerThread + hipThreadIdx_x;
        HIP_DYNAMIC_SHARED(unsigned int, block_histogram);
        
        #pragma unroll
        for(unsigned int bin_index = hipThreadIdx_x; bin_index < bin_count; bin_index += BlockSize)
        {
            if(bin_index < bin_count)
            {
                block_histogram[bin_index] = 0;
            }
        }
        __syncthreads();

        T items[ItemsPerThread];
        #pragma unroll
        for(unsigned int index = 0; index < ItemsPerThread; index++)
        {
            items[index] = device_input[offset + index * BlockSize];
        }

        #pragma unroll
        for(unsigned int index = 0; index < ItemsPerThread; index++)
        {
            unsigned int bin_index = items[index] / range;
            atomicInc(&block_histogram[bin_index], UINT_MAX);

        }

        __syncthreads();
        #pragma unroll
        for(unsigned int bin_index = hipThreadIdx_x; bin_index < bin_count; bin_index += BlockSize)
        {
            if(bin_index < bin_count)
            {
                atomicAdd(&device_output[bin_index], block_histogram[bin_index]);
            }
        }
    }
};

template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    class T
>
struct histogram_function<BlockSize, ItemsPerThread, T, histogram_per_block_padding>
{
    __device__ inline 
    void operator()(T* device_input, unsigned int* device_output, 
                    unsigned int bin_count, T max_range)
    {
        const T range = max_range / bin_count;
        const unsigned int offset = hipBlockIdx_x * BlockSize * ItemsPerThread + hipThreadIdx_x;
        HIP_DYNAMIC_SHARED(unsigned int, block_histogram);
        
        #pragma unroll
        for(unsigned int bin_index = hipThreadIdx_x; bin_index < bin_count; bin_index += BlockSize)
        {
            if(bin_index < bin_count)
            {
                block_histogram[bin_index] = 0;
            }
        }
        __syncthreads();

        T items[ItemsPerThread];
        #pragma unroll
        for(unsigned int index = 0; index < ItemsPerThread; index++)
        {
            items[index] = device_input[offset + index * BlockSize];
        }

        #pragma unroll
        for(unsigned int index = 0; index < ItemsPerThread; index++)
        {
            unsigned int bin_index = items[index] / range;
            atomicInc(&block_histogram[bin_index], UINT_MAX);

        }

        __syncthreads();
        #pragma unroll
        for(unsigned int bin_index = hipThreadIdx_x; bin_index < bin_count; bin_index += BlockSize)
        {
            if(bin_index < bin_count)
            {
                atomicAdd(&device_output[bin_index * cacheline_padding_size], block_histogram[bin_index]);
            }
        }
    }
};

template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    class T,
    class HistogramFunction = typename histogram_function
        <BlockSize, ItemsPerThread, T, noop>::value_type
>
__global__ void
histogram_base_kernel(T* device_input, unsigned int* device_output, 
                      unsigned int bin_count, T max_range, HistogramFunction histogram)
{
    histogram(device_input, device_output, bin_count, max_range); 
}

template<class T>
void run_histogram_cpu(std::vector<T>& host_input, std::vector<unsigned int>& host_expected_output,
                       unsigned int bin_count, T max_range, unsigned int size)
{
    T range_identifier = (max_range/ bin_count);
    for(int i = 0; i < bin_count; i++)
    {
        host_expected_output[i] = 0;
    }
    for(int i = 0; i < size; i++)
    {
        T input = host_input[i]; 
        unsigned int bin_index = host_input[i] / range_identifier;
        host_expected_output[bin_index]++;
    }
}

template<
   unsigned int BlockSize,
   unsigned int ItemsPerThread,
   class T,
   unsigned int HistogramMode 
>
void run_histogram_gpu()
{
    constexpr unsigned int max_trials = 20;
    constexpr unsigned int size = megabytes<T>(128);
    constexpr unsigned int output_size = megabytes<T>(1);
    constexpr T max = std::numeric_limits<T>::max();
    constexpr unsigned int grid_size = size / (BlockSize * ItemsPerThread);

    std::vector<T> host_input = get_random_data<T>(size, T(0), max);
    // To test retarded cases 
    //std::vector<T> host_input(size, 0);
    
    std::vector<unsigned int> host_expected_output(output_size, 0);
    std::vector<unsigned int> host_output(output_size, 0);
    std::vector<unsigned int> host_output_clear(output_size, 0);

    // Device memory allocation
    T* device_input;
    unsigned int * device_output;
    HIP_CHECK(hipMalloc(&device_input, host_input.size() * sizeof(typename decltype(host_input)::value_type)));
    HIP_CHECK(hipMalloc(&device_output, host_output.size() * sizeof(typename decltype(host_output)::value_type)));

    // Writing input data to device memory
    hip_write_device_memory<T>(device_input, host_input);
    
    // Selecting histogram function
    histogram_function<BlockSize, ItemsPerThread, T, HistogramMode> histogram;

    // Dynamic shared memory allocation
    unsigned int shared_memory_allocation_size = 0;

    // For the benchmarks that use padding the index of output is different
    unsigned int index_offset = 1;
    if(HistogramMode == output_cacheline_padding || HistogramMode == histogram_per_block_padding)
    {
        index_offset = cacheline_padding_size;
    }

    double average_time = 0;

    std::vector<unsigned int> bins_ = { 
        1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 160, 192, 256, 288, 320, 
        500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 10000
    };

    for(auto bins : bins_)
    {
        // Calculate results on cpu for verifying output
        run_histogram_cpu<T>(host_input, host_expected_output, bins, max, size);
        if(HistogramMode == histogram_per_block || HistogramMode == histogram_per_block_padding)
        {
            shared_memory_allocation_size = bins * sizeof(unsigned int) * 2;
        }
        average_time = 0;
        for(int trials = 0; trials < max_trials; trials++)
        {
            hip_write_device_memory<unsigned int>(device_output, host_output_clear);
            auto start = std::chrono::high_resolution_clock::now();
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(histogram_base_kernel<BlockSize, ItemsPerThread, T>),
                dim3(grid_size), dim3(BlockSize),
                shared_memory_allocation_size, 0,
                device_input, device_output, bins, max,
                histogram
            );
            HIP_CHECK(hipPeekAtLastError());
            HIP_CHECK(hipDeviceSynchronize());
            auto end = std::chrono::high_resolution_clock::now();
            auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
            average_time = (average_time + elapsed_seconds.count()) / 2;

            // Reading output from device
            hip_read_device_memory<unsigned int>(host_output, device_output);
            for(unsigned int i = 0; i < bins; i++)
            {
                if(host_output[i * index_offset] != host_expected_output[i])
                {
                    std::cout << "Wrong result!" << std::endl;
                    break;
                }
            }
        }
        std::cout << histogram_function_names[HistogramMode] << " (" << bins << "-bins) " 
                  << " IPT: " << ItemsPerThread << " - "
                  << (size * sizeof(T)) / (average_time * 1024 * 1024 * 1024) << "GB/s" 
                  << std::endl;
    }

    HIP_CHECK(hipFree(device_input));
    HIP_CHECK(hipFree(device_output));
}

int main()
{
    // Initializing HIP device
    hipDeviceProp_t device_properties;
    HIP_CHECK(hipGetDeviceProperties(&device_properties, 0));
    std::cout << "Name: " << device_properties.name << std::endl;
    std::cout << "Global mem: " << device_properties.totalGlobalMem << std::endl;
    std::cout << "Shared mem per block: " << device_properties.sharedMemPerBlock << std::endl;
    std::cout << "Regs per block:  " << device_properties.regsPerBlock << std::endl;
    std::cout << "Warp size: " << device_properties.warpSize << std::endl;
    std::cout << "Max threads per block " << device_properties.maxThreadsPerBlock << std::endl;
    std::cout << "L2 cache size " << device_properties.l2CacheSize << std::endl;
    std::cout << "Multiprocessor count " << device_properties.multiProcessorCount << std::endl;
    std::cout << "Shared mem per multiprocessor " << device_properties.maxSharedMemoryPerMultiProcessor << std::endl << std::endl;

    run_histogram_gpu<256, 1,   float, naive>();
    run_histogram_gpu<256, 1,   float, striped_read>();
    run_histogram_gpu<256, 1,   float, output_cacheline_padding>();
    run_histogram_gpu<256, 1,   float, histogram_per_block>();
    run_histogram_gpu<256, 1,   float, histogram_per_block_padding>();

    run_histogram_gpu<256, 2,   float, naive>();
    run_histogram_gpu<256, 2,   float, striped_read>();
    run_histogram_gpu<256, 2,   float, output_cacheline_padding>();
    run_histogram_gpu<256, 2,   float, histogram_per_block>();
    run_histogram_gpu<256, 2,   float, histogram_per_block_padding>();

    run_histogram_gpu<256, 4,   float, naive>();
    run_histogram_gpu<256, 4,   float, striped_read>();
    run_histogram_gpu<256, 4,   float, output_cacheline_padding>();
    run_histogram_gpu<256, 4,   float, histogram_per_block>();
    run_histogram_gpu<256, 4,   float, histogram_per_block_padding>();

    run_histogram_gpu<256, 8,   float, naive>();
    run_histogram_gpu<256, 8,   float, striped_read>();
    run_histogram_gpu<256, 8,   float, output_cacheline_padding>();
    run_histogram_gpu<256, 8,   float, histogram_per_block>();
    run_histogram_gpu<256, 8,   float, histogram_per_block_padding>();
    
    run_histogram_gpu<256, 16,  float, naive>();
    run_histogram_gpu<256, 16,  float, striped_read>();
    run_histogram_gpu<256, 16,  float, output_cacheline_padding>();
    run_histogram_gpu<256, 16,  float, histogram_per_block>();
    run_histogram_gpu<256, 16,  float, histogram_per_block_padding>();

    run_histogram_gpu<256, 32,  float, naive>();
    run_histogram_gpu<256, 32,  float, striped_read>();
    run_histogram_gpu<256, 32,  float, output_cacheline_padding>();
    run_histogram_gpu<256, 32,  float, histogram_per_block>();
    run_histogram_gpu<256, 32,  float, histogram_per_block_padding>();

    run_histogram_gpu<256, 64,  float, naive>();
    run_histogram_gpu<256, 64,  float, striped_read>();
    run_histogram_gpu<256, 64,  float, output_cacheline_padding>();
    run_histogram_gpu<256, 64,  float, histogram_per_block>();
    run_histogram_gpu<256, 64,  float, histogram_per_block_padding>();
    
    run_histogram_gpu<256, 128, float, naive>();
    run_histogram_gpu<256, 128, float, striped_read>();
    run_histogram_gpu<256, 128, float, output_cacheline_padding>();
    run_histogram_gpu<256, 128, float, histogram_per_block>();
    run_histogram_gpu<256, 128, float, histogram_per_block_padding>();
}


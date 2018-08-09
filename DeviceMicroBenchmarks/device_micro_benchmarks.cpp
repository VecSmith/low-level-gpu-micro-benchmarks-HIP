#include "device_micro_benchmarks.h"

enum kernel_operation_mode
{
    no_operation,
    memory_load_store_striped,
    memory_load_store_direct,
    memory_striped_load_direct_store,
    memory_direct_load_striped_store,
    memory_load_store_vec4_32_direct,
    memory_load_store_vec4_32_striped,
    memory_load_store_vec4_32_srdw,
    memory_load_store_vec4_32_drsw,
    memory_load_store_vec2_32_direct,
    memory_load_store_vec2_32_striped,
    memory_load_store_vec2_32_srdw,
    memory_load_store_vec2_32_drsw,
    shared_mem_read,
    shared_mem_write,
    atomics_inter_block_conflict_padding,
    atomics_inter_block_conflict_no_padding,
    atomics_intra_block_conflict,
};

const std::string kernel_operation_mode_strings[20] =
{
    std::string("no_operation"),
    std::string("memory_load_store_striped"),
    std::string("memory_load_store_direct"),
    std::string("memory_striped_load_direct_store"),
    std::string("memory_direct_load_striped_store"),
    std::string("memory_load_store_vec4_32_direct"),
    std::string("memory_load_store_vec4_32_striped"),
    std::string("memory_load_store_vec4_32_srdw"),
    std::string("memory_load_store_vec4_32_drsw"),
    std::string("memory_load_store_vec2_32_direct"),
    std::string("memory_load_store_vec2_32_striped"),
    std::string("memory_load_store_vec2_32_srdw"),
    std::string("memory_load_store_vec2_32_drsw"),
    std::string("shared_mem_read"),
    std::string("shared_mem_write"),
    std::string("atomics_inter_block_conflict_padding"),
    std::string("atomics_inter_block_conflict_no_padding"),
    std::string("atomics_intra_block_conflict"),
};

// For better code readability when readong kernel operations
constexpr bool direct = false;
constexpr bool striped = true;

// AMD compiler does not treat uint4 as vector type
#ifdef __HIP_PLATFORM_HCC__
using vec4_32bit = ::hc::short_vector::uint4;
using vec2_32bit = ::hc::short_vector::uint2;
#else
using vec4_32bit = uint4;
using vec2_32bit = uint2;
#endif

// Helper device functions
#define MEMORY_LOAD_STORE_INIT() \
constexpr unsigned int offset_multiplier = (Striped) ? (1) : (ItemsPerThread); \
constexpr unsigned int index_multiplier = (Striped) ? (BlockSize) : (1); \
const unsigned int block_offset = hipBlockIdx_x * BlockSize * ItemsPerThread; \
const unsigned int offset = block_offset + hipThreadIdx_x * offset_multiplier;

template<
    class T,
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    bool Striped
>
__device__ inline
void scalar_load(T (&items)[ItemsPerThread], T* memory_pointer)
{
    MEMORY_LOAD_STORE_INIT();
    #pragma unroll
    for(unsigned int index = 0; index < ItemsPerThread; index++)
    {
        items[index] = memory_pointer[offset + index * index_multiplier];
    }
};

template<
    class T,
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    bool Striped
>
__device__ inline
void scalar_store(T (&items)[ItemsPerThread], T* memory_pointer)
{
    MEMORY_LOAD_STORE_INIT();
    #pragma unroll
    for(unsigned int index = 0; index < ItemsPerThread; index++)
    {
        memory_pointer[offset + index * index_multiplier] = items[index];
    }
};

#define MEMORY_LOAD_STORE_VECTOR_INIT() \
constexpr unsigned int offset_multiplier = (Striped) ? (1) : (VectorCount); \
constexpr unsigned int index_multiplier = (Striped) ? (BlockSize) : (1); \
const unsigned int block_offset = hipBlockIdx_x * BlockSize * VectorCount; \
const unsigned int offset = block_offset + hipThreadIdx_x * offset_multiplier; \

template<
    class T,
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    bool Striped,
    class VectorType,
    unsigned int VectorCount
>
__device__ inline
void vectorized_load(T* input, VectorType items[VectorCount])
{
    MEMORY_LOAD_STORE_VECTOR_INIT();
    #pragma unroll
    for(unsigned int index = 0; index < VectorCount; index++)
    {
        items[index] = (reinterpret_cast<VectorType*>(input))[offset + index * index_multiplier];
    }
}

template<
    class T,
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    bool Striped,
    class VectorType,
    unsigned int VectorCount
>
__device__ inline
void vectorized_store(T* output, VectorType items[VectorCount])
{
    MEMORY_LOAD_STORE_VECTOR_INIT();
    #pragma unroll
    for(unsigned int index = 0; index < VectorCount; index++)
    {
        (reinterpret_cast<VectorType*>(output))[offset + index * index_multiplier] = items[index];
    }
}

template<
    class T,
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    class VectorType,
    bool StripedLoad,
    bool StripedStore
>
__device__ inline
void vectorized_operation_base(T* input, T* output)
{
    constexpr unsigned int vector_count = ItemsPerThread / (sizeof(VectorType) / sizeof(T));
    VectorType items[vector_count];
    vectorized_load<T, BlockSize, ItemsPerThread, StripedLoad, VectorType, vector_count>(input, items);
    vectorized_store<T, BlockSize, ItemsPerThread, StripedStore, VectorType, vector_count>(output, items);
}

// kernel_operation base struct
template<
    class T,
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    kernel_operation_mode KernelJob
>
struct kernel_operation;

// Placeholder specialization
template<
    class T,
    unsigned int BlockSize,
    unsigned int ItemsPerThread
>
struct kernel_operation<T, BlockSize, ItemsPerThread, no_operation>
{
    __device__ inline
    void operator()(T* input, T* output)
    {}
};

#define CREATE_VECTOR_MEMORY_OPERATION(kernel_job, vector_type, read_pattern, write_pattern) \
template< \
    class T, \
    unsigned int BlockSize, \
    unsigned int ItemsPerThread \
> \
struct kernel_operation<T, BlockSize, ItemsPerThread, kernel_job> \
{ \
    __device__ inline \
    void operator()(T* input, T* output) \
    { \
        vectorized_operation_base \
            <T, BlockSize, ItemsPerThread, vector_type, read_pattern, write_pattern>(input, output); \
    } \
};

CREATE_VECTOR_MEMORY_OPERATION(memory_load_store_vec4_32_direct,  vec4_32bit, direct,  direct);
CREATE_VECTOR_MEMORY_OPERATION(memory_load_store_vec4_32_striped, vec4_32bit, striped, striped);
CREATE_VECTOR_MEMORY_OPERATION(memory_load_store_vec4_32_srdw,    vec4_32bit, striped, direct);
CREATE_VECTOR_MEMORY_OPERATION(memory_load_store_vec4_32_drsw,    vec4_32bit, direct,  striped);

CREATE_VECTOR_MEMORY_OPERATION(memory_load_store_vec2_32_direct,  vec2_32bit, direct,  direct);
CREATE_VECTOR_MEMORY_OPERATION(memory_load_store_vec2_32_striped, vec2_32bit, striped, striped);
CREATE_VECTOR_MEMORY_OPERATION(memory_load_store_vec2_32_srdw,    vec2_32bit, striped, direct);
CREATE_VECTOR_MEMORY_OPERATION(memory_load_store_vec2_32_drsw,    vec2_32bit, direct,  striped);

#define CREATE_MEMORY_OPERATION(kernel_job, read_pattern, write_pattern) \
template< \
    class T, \
    unsigned int BlockSize, \
    unsigned int ItemsPerThread \
> \
struct kernel_operation<T, BlockSize, ItemsPerThread, kernel_job> \
{ \
    __device__ inline \
    void operator()(T* input, T* output) \
    { \
        T items[ItemsPerThread]; \
        scalar_load<T, BlockSize, ItemsPerThread, read_pattern>(items, input); \
        scalar_store<T, BlockSize, ItemsPerThread, write_pattern>(items, output); \
    } \
};

CREATE_MEMORY_OPERATION(memory_load_store_striped,        striped, striped);
CREATE_MEMORY_OPERATION(memory_load_store_direct,         direct,  direct);
CREATE_MEMORY_OPERATION(memory_striped_load_direct_store, striped, direct);
CREATE_MEMORY_OPERATION(memory_direct_load_striped_store, direct,  striped);

#define SHARED_MEM_REPEATS 40
// shared memory helper function
template<
    class T,
    unsigned int BlockSize,
    unsigned int ActiveBanks
>
__device__ inline
void shared_mem_general_function(kernel_operation_mode shared_mem_operation,
                                 T* input, T* output)
{
    //TODO dynamic warp size for different devices using dynamic shared mem allocation
    //TODO dynamic bank count for devices other than GPUs
    #if defined __HIP_PLATFORM_HCC__
        constexpr unsigned int warp_size = 64;
        constexpr unsigned int bank_count = (32 * 4) / sizeof(T);
    #elif defined __HIP_PLATFORM_NVCC__
        constexpr unsigned int warp_size = 32;
        constexpr unsigned int bank_count = (32 * 4) / sizeof(T);
    #endif
    constexpr unsigned int shared_mem_read_size = bank_count * warp_size;

    // Shared mem has to be volatile so that the compiler doesnt optimize it
    __shared__ volatile T shared_items[shared_mem_read_size];

    const unsigned int thread_id_in_warp = hipThreadIdx_x % warp_size;
    const unsigned int thread_bank_id = thread_id_in_warp % ActiveBanks;
    const unsigned int thread_shared_mem_read_index =
        thread_bank_id + (thread_id_in_warp / ActiveBanks) * bank_count;

    #pragma unroll
    for(unsigned int index = 0; index < shared_mem_read_size / BlockSize; index++)
    {
        shared_items[hipThreadIdx_x + (index * BlockSize)] = index;
    }
    __syncthreads();

    T item[1];

    scalar_load<T, BlockSize, 1, striped>(item, input);

    #pragma unroll
    for(unsigned int repeat = 0; repeat < SHARED_MEM_REPEATS; repeat++)
    {
        if(shared_items[thread_shared_mem_read_index] != 50)
        {
            // Specializations defined here
            switch(shared_mem_operation)
            {
                case shared_mem_read:
                {
                    item[0] += shared_items[thread_shared_mem_read_index];
                } break;

                case shared_mem_write:
                {
                    shared_items[thread_shared_mem_read_index] += item[0] + repeat;
                } break;

                default:
                {} break;
            };
        }
    }

    item[0] += shared_items[hipThreadIdx_x];

    scalar_store<T, BlockSize, 1, striped>(item, output);
}

#define SHARED_MEM_OPERATION(shared_mem_operation) \
template< \
    class T, \
    unsigned int BlockSize, \
    unsigned int ActiveBanks \
> \
struct kernel_operation<T, BlockSize, ActiveBanks, shared_mem_operation> \
{ \
    __device__ inline \
    void operator()(T* input, T* output) \
    { \
        shared_mem_general_function<T, BlockSize, ActiveBanks>(shared_mem_operation, input, output); \
    } \
};

SHARED_MEM_OPERATION(shared_mem_read);
SHARED_MEM_OPERATION(shared_mem_write);

#define ATOMIC_REPEATS 4
// Atomics helper
template<
    class T,
    unsigned int BlockSize,
    unsigned int ActiveSegments
>
__device__ inline
void atomic_write(T* input, T* output, unsigned int write_index)
{
    T items[1];
    scalar_load<T, BlockSize, 1, striped>(items, input);
    #pragma nounroll
    for(unsigned int repeat = 0; repeat < ATOMIC_REPEATS; repeat++)
    {
        atomicExch(&output[write_index], items[0]);
    }
}

template<
    class T,
    unsigned int BlockSize,
    unsigned int ActiveSegments
>
struct kernel_operation<T, BlockSize, ActiveSegments, atomics_inter_block_conflict_no_padding>
{
    __device__ inline
    void operator()(T* input, T* output)
    {
        unsigned int write_index = (hipBlockIdx_x * BlockSize + hipThreadIdx_x) % ActiveSegments;
        atomic_write<T, BlockSize, ActiveSegments>(input, output, write_index);
    }
};

template<
    class T,
    unsigned int BlockSize,
    unsigned int ActiveSegments
>
struct kernel_operation<T, BlockSize, ActiveSegments, atomics_inter_block_conflict_padding>
{
    __device__ inline
    void operator()(T* input, T* output)
    {
        constexpr unsigned int segment_length_uint = 256;
        unsigned int write_index = (hipBlockIdx_x % ActiveSegments) * segment_length_uint + hipThreadIdx_x;
        atomic_write<T, BlockSize, ActiveSegments>(input, output, write_index);
    }
};

template<
    class T,
    unsigned int BlockSize,
    unsigned int ActiveSegments
>
struct kernel_operation<T, BlockSize, ActiveSegments, atomics_intra_block_conflict>
{
    __device__ inline
    void operator()(T* input, T* output)
    {
        constexpr unsigned int segment_length_uint = 256;
        unsigned int write_index = (hipBlockIdx_x * segment_length_uint) + (hipThreadIdx_x % ActiveSegments);
        atomic_write<T, BlockSize, ActiveSegments>(input, output, write_index);
    }
};

// Base kernel
template<
    class T,
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    class Operation = typename kernel_operation
        <T, BlockSize, ItemsPerThread, no_operation>::value_type
>
__global__
void kernel(T* input, T* output, Operation operation)
{
    operation(input, output);
}

template<
    class T,
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    kernel_operation_mode KernelOp = no_operation
>
void run_benchmark(size_t size, std::string benchmark_name)
{
    size_t grid_size;
    double result_multiplier;

    switch(KernelOp)
    {
        case shared_mem_read:
        case shared_mem_write:
        {
            grid_size = size / BlockSize;
            result_multiplier = SHARED_MEM_REPEATS;
        } break;

        case atomics_inter_block_conflict_padding:
        case atomics_inter_block_conflict_no_padding:
        case atomics_intra_block_conflict:
        {
            grid_size = size / BlockSize;
            result_multiplier = 1;
        } break;

        default:
        {
            grid_size = size / (BlockSize * ItemsPerThread);
            result_multiplier = 1;
        } break;
    };

    std::vector<T> input(size);
    if(std::is_floating_point<T>::value)
    {
        input = get_random_data<T>(size, (T)-1000, (T)+1000);
    }
    else
    {
        input = get_random_data<T>(
            size,
            std::numeric_limits<T>::min(),
            std::numeric_limits<T>::max()
        );
    }

    T * d_input;
    T * d_output;
    HIP_CHECK(hipMalloc(&d_input, size * sizeof(T)));
    HIP_CHECK(hipMalloc(&d_output, size * sizeof(T)));
    HIP_CHECK(
        hipMemcpy(
            d_input, input.data(),
            size * sizeof(T),
            hipMemcpyHostToDevice
        )
    );
    HIP_CHECK(hipDeviceSynchronize());

    kernel_operation<T, BlockSize, ItemsPerThread, KernelOp> operation;

    // Warm-up
    for(size_t i = 0; i < 10; i++)
    {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(kernel<T, BlockSize, ItemsPerThread>),
            dim3(grid_size), dim3(BlockSize), 0, 0,
            d_input, d_output, operation
        );
    }
    HIP_CHECK(hipDeviceSynchronize());

    const unsigned int batch_size = 10;
    double average_time = 0;
    for(unsigned int iteration = 0; iteration < 100; iteration++)
    {
        auto start = std::chrono::high_resolution_clock::now();
        for(size_t i = 0; i < batch_size; i++)
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(kernel<T, BlockSize, ItemsPerThread>),
                dim3(grid_size), dim3(BlockSize), 0, 0,
                d_input, d_output, operation
            );
        }
        HIP_CHECK(hipDeviceSynchronize());

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds =
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        average_time = (average_time + elapsed_seconds.count()) / 2;
    }
    std::cout
        << benchmark_name << create_space(benchmark_name, 40)
        << "<" <<BlockSize << "," << ItemsPerThread << ">"
        << create_space(std::to_string(ItemsPerThread) + std::to_string(BlockSize), 10)
        << ((size * sizeof(T)) * batch_size * result_multiplier
            / (1024.0 * 1024.0 * 1024.0))
            / average_time
        << " GB/s"
        << std::endl;

    HIP_CHECK(
        hipMemcpy(
            input.data(), d_output,
            size * sizeof(T),
            hipMemcpyDeviceToHost
        )
    );
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_output));
}

template<class T>
void run_benchmark_memcpy(size_t size, std::string benchmark_name = "Memory copy: ")
{
    std::vector<T> input;
    if(std::is_floating_point<T>::value)
    {
        input = get_random_data<T>(size, (T)-1000, (T)+1000);
    }
    else
    {
        input = get_random_data<T>(
            size,
            std::numeric_limits<T>::min(),
            std::numeric_limits<T>::max()
        );
    }
    T * d_input;
    T * d_output;
    HIP_CHECK(hipMalloc(&d_input, size * sizeof(T)));
    HIP_CHECK(hipMalloc(&d_output, size * sizeof(T)));
    // Warm-up
    for(size_t i = 0; i < 10; i++)
    {
        HIP_CHECK(hipMemcpy(d_output, d_input, size * sizeof(T), hipMemcpyDeviceToDevice));
    }
    HIP_CHECK(hipDeviceSynchronize());

    const unsigned int batch_size = 10;
    double average_time = 0;
    for(unsigned int iteration = 0; iteration < 100; iteration++)
    {
        auto start = std::chrono::high_resolution_clock::now();
        for(size_t i = 0; i < batch_size; i++)
        {
            HIP_CHECK(hipMemcpy(d_output, d_input, size * sizeof(T), hipMemcpyDeviceToDevice));
        }
        HIP_CHECK(hipDeviceSynchronize());

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds =
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        average_time = (average_time + elapsed_seconds.count()) / 2;
    }
    std::cout
        << benchmark_name << create_space(benchmark_name)
        << ((double)(size * sizeof(T)) * batch_size / (1024.0 * 1024.0 * 1024.0)) / average_time
        << " GB/s"
        << std::endl;

    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_output));
}

template<
    class T,
    unsigned int BlockSize,
    unsigned int SizeInBytes,
    kernel_operation_mode KernelOp
>
void run_fundamental_configs()
{
    run_benchmark<T, BlockSize, 4,   KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 8,   KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 12,  KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 16,  KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 20,  KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 24,  KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 28,  KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 32,  KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 36,  KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 40,  KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 44,  KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 48,  KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 52,  KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 56,  KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 60,  KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 64,  KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
}

template<
    class T,
    unsigned int BlockSize,
    unsigned int SizeInBytes,
    kernel_operation_mode KernelOp
>
void run_all_configs()
{
    run_benchmark<T, BlockSize, 1,   KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 2,   KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 3,   KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 4,   KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 5,   KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 6,   KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 7,   KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 8,   KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 9,   KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 10,  KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 11,  KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 12,  KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 13,  KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 14,  KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 15,  KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 16,  KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 17,  KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 18,  KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 19,  KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 20,  KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 21,  KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 22,  KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 23,  KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 24,  KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 25,  KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 26,  KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 27,  KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 28,  KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 29,  KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 30,  KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 31,  KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 32,  KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 33,  KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 34,  KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 35,  KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 36,  KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 37,  KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 38,  KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 39,  KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 40,  KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 41,  KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 42,  KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 43,  KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 60,  KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 61,  KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 62,  KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 63,  KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 64,  KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 65,  KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 66,  KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 67,  KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 68,  KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 69,  KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 70,  KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 71,  KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 72,  KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 73,  KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 74,  KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 75,  KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 76,  KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 77,  KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 78,  KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 79,  KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 80,  KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 81,  KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 82,  KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 83,  KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 84,  KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 85,  KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 86,  KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 87,  KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 88,  KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 89,  KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 90,  KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 91,  KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 92,  KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 93,  KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 94,  KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 95,  KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 96,  KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 97,  KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 98,  KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 99,  KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 100, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 101, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 102, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 103, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 104, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 105, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 106, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 107, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 108, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 109, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 110, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 111, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 112, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 113, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 114, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 115, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 116, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 117, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 118, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 119, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 120, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 121, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 122, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 123, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 124, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 125, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 126, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 127, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 128, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 129, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 130, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 131, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 132, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 133, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 134, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 135, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 136, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 137, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 138, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 139, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 140, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 141, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 142, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 143, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 144, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 145, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 146, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 147, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 148, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 149, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 150, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 151, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 152, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 153, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 154, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 155, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 156, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 157, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 158, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 159, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 160, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 161, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 162, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 163, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 164, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 165, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 166, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 167, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 168, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 169, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 170, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 171, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 172, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 173, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 174, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 175, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 176, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 177, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 178, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 179, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 180, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 181, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 182, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 183, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 184, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 185, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 186, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 187, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 188, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 189, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 190, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 191, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 192, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 193, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 194, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 195, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 196, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 197, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 198, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 199, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 200, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 201, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 202, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 203, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 204, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 205, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 206, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 207, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 208, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 209, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 210, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 211, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 212, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 213, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 214, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 215, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 216, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 217, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 218, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 219, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 220, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 221, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 222, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 223, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 224, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 225, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 226, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 227, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 228, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 229, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 230, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 231, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 232, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 233, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 234, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 235, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 236, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 237, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 238, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 239, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 240, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 241, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 242, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 243, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 244, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 245, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 246, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 247, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 248, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 249, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 250, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 251, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 252, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 253, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 254, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 255, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 256, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 257, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 258, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
    run_benchmark<T, BlockSize, 259, KernelOp>(SizeInBytes, kernel_operation_mode_strings[KernelOp]);
}

int main(int argc, char *argv[])
{
    // HIP init
    hipDeviceProp_t devProp;
    int device_id = 1;
    HIP_CHECK(hipGetDevice(&device_id));
    HIP_CHECK(hipGetDeviceProperties(&devProp, device_id));
    std::cout << "Device name: " << devProp.name << std::endl;
    std::cout << "L2 Cache size: " << devProp.l2CacheSize << std::endl;
    std::cout << "Warp size: " << devProp.warpSize << std::endl;
    std::cout << "Shared memory per block: " << devProp.sharedMemPerBlock << std::endl;

    // benchmarks configuration
    using data_type = unsigned int;
    constexpr unsigned int size_in_bytes = megabytes<data_type>(128);
    constexpr unsigned int block_size = 256;

    // running memcpy to use as base performance for memory operations
    run_benchmark_memcpy<data_type>(size_in_bytes);

#define ENABLE_MEMORY_BENCHMARKS
#define ENABLE_SHARED_MEMORY_BENCHMARKS
#define ENABLE_ATOMIC_BENCHMARKS

#ifdef ENABLE_MEMORY_BENCHMARKS
    // 32-bit
    run_all_configs<data_type, block_size, size_in_bytes, memory_load_store_direct>();
    run_all_configs<data_type, block_size, size_in_bytes, memory_load_store_striped>();
    run_all_configs<data_type, block_size, size_in_bytes, memory_striped_load_direct_store>();
    run_all_configs<data_type, block_size, size_in_bytes, memory_direct_load_striped_store>();

    // Vectorized
    // 128-bit
    run_fundamental_configs<data_type, block_size, size_in_bytes, memory_load_store_vec4_32_direct>();
    run_fundamental_configs<data_type, block_size, size_in_bytes, memory_load_store_vec4_32_striped>();
    run_fundamental_configs<data_type, block_size, size_in_bytes, memory_load_store_vec4_32_drsw>();
    run_fundamental_configs<data_type, block_size, size_in_bytes, memory_load_store_vec4_32_srdw>();
    // 64-bit
    run_fundamental_configs<data_type, block_size, size_in_bytes, memory_load_store_vec2_32_direct>();
    run_fundamental_configs<data_type, block_size, size_in_bytes, memory_load_store_vec2_32_striped>();
    run_fundamental_configs<data_type, block_size, size_in_bytes, memory_load_store_vec2_32_drsw>();
    run_fundamental_configs<data_type, block_size, size_in_bytes, memory_load_store_vec2_32_srdw>();
#endif

#ifdef ENABLE_SHARED_MEMORY_BENCHMARKS
    run_all_configs<data_type, block_size, size_in_bytes, shared_mem_read>();
    run_all_configs<data_type, block_size, size_in_bytes, shared_mem_write>();
#endif

#ifdef ENABLE_ATOMIC_BENCHMARKS
    run_all_configs<data_type, block_size, size_in_bytes, atomics_inter_block_conflict_padding>();
    run_all_configs<data_type, block_size, size_in_bytes, atomics_intra_block_conflict>();
    run_all_configs<data_type, block_size, size_in_bytes, atomics_inter_block_conflict_no_padding>();
#endif
}


#include <stdio.h>
#include <stdlib.h>
#include <cstring>
// CUDA headers
#include <cuda.h>
#include <cuda_runtime.h>
#include <cupti.h>
#include <stdio.h>
#include <iomanip>
#include <cassert>
#include "gmp/profile.h"
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/fill.h>
#include <thrust/transform.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cub/device/device_for.cuh>
#include <cub/iterator/cache_modified_input_iterator.cuh>
#include <cub/iterator/cache_modified_output_iterator.cuh>
#include <cuda/std/mdspan>
#include <cuda/atomic>

// #define N 13824*4*1024*16 // vector length, 3.456GB
#define N (4*128*768)
#define LARGE_N (4*128*768*32)

// #define CUPTI_CALL(call)                                                         \
//     do                                                                           \
//     {                                                                            \
//         CUptiResult _status = call;                                              \
//         if (_status != CUPTI_SUCCESS)                                            \
//         {                                                                        \
//             const char *errstr;                                                  \
//             cuptiGetResultString(_status, &errstr);                              \
//             fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n", \
//                     __FILE__, __LINE__, #call, errstr);                          \
//             exit(-1);                                                            \
//         }                                                                        \
//     } while (0)

// Simple CUDA kernel
__global__ void hello_kernel()
{
    printf("Hello World from GPU thread %d!\n", threadIdx.x);
}

__global__ void vecAdd(const float *A, const float *B, float *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements)
        C[i] = A[i] + B[i];
}

void vecAdd_thrust(float* out, float* inp1, float* inp2, int n) {
    cub::CacheModifiedInputIterator<cub::LOAD_CS, float> inp1cs(inp1);
    cub::CacheModifiedInputIterator<cub::LOAD_CS, float> inp2cs(inp2);
    thrust::transform(thrust::cuda::par_nosync, inp1cs, inp1cs + n, inp2cs, out, thrust::plus<float>());
}

__global__ void multiply(const float *A, const float *B, float *C, int numElements)
{
    for (int i =  blockDim.x * blockIdx.x + threadIdx.x; i < numElements; i += blockDim.x * gridDim.x)
        C[i] = A[i] * B[i];
}

__global__ void square(float *A, int n)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        A[i] = A[i] * A[i];
    }
}

__global__ void saxpy(int n, float a, float *x, float *y)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
         i < n; 
         i += blockDim.x * gridDim.x) 
      {
          y[i] = a * x[i] + y[i];
      }
}

__global__ void saxpy_more_compute(int n, float a, float *x, float *y)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
         i < n; 
         i += blockDim.x * gridDim.x) 
      {
        // 20 multiply-add operations
          y[i] = a * x[i] + y[i];
          y[i] = a * x[i] + y[i];
          y[i] = a * x[i] + y[i];
          y[i] = a * x[i] + y[i];
          y[i] = a * x[i] + y[i];
          y[i] = a * x[i] + y[i];
          y[i] = a * x[i] + y[i];
          y[i] = a * x[i] + y[i];
          y[i] = a * x[i] + y[i];
          y[i] = a * x[i] + y[i];
          y[i] = a * x[i] + y[i];
          y[i] = a * x[i] + y[i];
          y[i] = a * x[i] + y[i];
          y[i] = a * x[i] + y[i];
          y[i] = a * x[i] + y[i];
          y[i] = a * x[i] + y[i];
          y[i] = a * x[i] + y[i];
          y[i] = a * x[i] + y[i];
          y[i] = a * x[i] + y[i];
          y[i] = a * x[i] + y[i];
      }
}

__global__ void sumReduction(float *input, float *output, int n)
{
    __shared__ float sdata[256]; // shared memory for partial sums
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();

    // reduce within block
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0)
        output[blockIdx.x] = sdata[0];
}

// Test launch overhead by launching large number of kernels
// The total wall clock is compared with the total gpu time reported by GMP.
void testLaunchOverhead(float *d_A_1, float *d_B_1, float *d_A_2, float *d_B_2, float *d_C_2, float *d_A_3, float *d_B_3, float *d_C_3, float *d_A_4)
{
    GmpProfiler::getInstance()->pushRange("LaunchOverhead", GmpProfileType::CONCURRENT_KERNEL);
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 110; i++)
    {
        saxpy<<<4, 256>>>(N, 2.0f, d_A_1, d_B_1);
        multiply<<<4, 256>>>(d_A_3, d_B_3, d_C_3, N);
        multiply<<<4, 256>>>(d_A_2, d_B_2, d_C_2, N);
        square<<<4, 256>>>(d_A_4, N);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::nano> duration = end - start;
    printf("Wall clock time for all kernel launches: %f ns\n",
           duration.count());
    GmpProfiler::getInstance()->popRange("LaunchOverhead", GmpProfileType::CONCURRENT_KERNEL);

    GmpProfiler::getInstance()->pushRange("LaunchOverhead_SmallKernels", GmpProfileType::CONCURRENT_KERNEL);
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 110; i++)
    {
        saxpy<<<1, 16>>>(N, 2.0f, d_A_1, d_B_1);
        multiply<<<1, 16>>>(d_A_3, d_B_3, d_C_3, N);
        multiply<<<1, 16>>>(d_A_2, d_B_2, d_C_2, N);
        square<<<1, 16>>>(d_A_4, N);
    }
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    printf("Wall clock time for all small kernel launches: %f ns\n",
           duration.count());
    GmpProfiler::getInstance()->popRange("LaunchOverhead_SmallKernels", GmpProfileType::CONCURRENT_KERNEL);
}

// This function reveals the bandwidth distribution among multiple kernels launched within a logical range.
// The bandwidth reported by GMP has a high-low-high pattern.
void testBandwidthDist(float *d_A_5, float *d_B_5, float *d_C_5)
{
    GmpProfiler::getInstance()->pushRange("BandwidthDist", GmpProfileType::CONCURRENT_KERNEL);
    for(int i = 0; i<20;i++){
        // Typical workflow in a range
        // Each kernel accept the ouput of previous kernel as input.
        saxpy<<<108, 128>>>(N, 2.0f, d_A_5, d_B_5);
        multiply<<<108, 128>>>(d_A_5, d_B_5, d_C_5, N);
        multiply<<<108, 128>>>(d_A_5, d_B_5, d_C_5, N);
        square<<<108, 128>>>(d_C_5, N);
    }
    
    GmpProfiler::getInstance()->popRange("BandwidthDist", GmpProfileType::CONCURRENT_KERNEL);
}

void launch_kernels()
{
    size_t size = N * sizeof(float);
    size_t large_size = LARGE_N * sizeof(float);

    // No need to initialize the input data, as we are not copying from host to device
    // If N is too large, these host arrays lead to segment fault

    // Host vectors
    // float h_A[N], h_B[N], h_C[N];
    // for (int i = 0; i < N; i++)
    // {
    //     h_A[i] = i;
    //     h_B[i] = i * 10;
    // }

    // Device vectors
    float *d_A_1, *d_B_1, *d_C_1;
    float *d_A_2, *d_B_2, *d_C_2;
    float *d_A_3, *d_B_3, *d_C_3;
    float *d_A_4, *d_B_4, *d_C_4;
    float *d_A_5, *d_B_5, *d_C_5;   // Large arraies

    cudaMalloc((void **)&d_A_1, size);
    cudaMalloc((void **)&d_B_1, size);
    cudaMalloc((void **)&d_C_1, size);

    cudaMalloc((void **)&d_A_2, size);
    cudaMalloc((void **)&d_B_2, size);
    cudaMalloc((void **)&d_C_2, size);

    cudaMalloc((void **)&d_A_3, size);
    cudaMalloc((void **)&d_B_3, size);
    cudaMalloc((void **)&d_C_3, size);

    cudaMalloc((void **)&d_A_4, size);
    cudaMalloc((void **)&d_B_4, size);
    cudaMalloc((void **)&d_C_4, size);

    cudaMalloc((void **)&d_A_5, large_size);
    cudaMalloc((void **)&d_B_5, large_size);
    cudaMalloc((void **)&d_C_5, large_size);
    printf("Allocated device memory\n");

    // Copy from host to device
    // Note: We skip the memory copy time in profiling, as we want to focus on kernel execution
    // cudaMemcpy(d_A_1, h_A, size, cudaMemcpyHostToDevice);
    // cudaMemcpy(d_B_1, h_B, size, cudaMemcpyHostToDevice);
    // cudaMemcpy(d_A_2, h_A, size, cudaMemcpyHostToDevice);
    // cudaMemcpy(d_B_2, h_B, size, cudaMemcpyHostToDevice);
    // cudaMemcpy(d_A_3, h_A, size, cudaMemcpyHostToDevice);
    // cudaMemcpy(d_B_3, h_B, size, cudaMemcpyHostToDevice);
    // cudaMemcpy(d_A_4, h_A, size, cudaMemcpyHostToDevice);
    // cudaMemcpy(d_B_4, h_B, size, cudaMemcpyHostToDevice);
    // cudaMemcpy(d_C_4, h_C, size, cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();

    // GmpProfiler::getInstance()->pushRange("vecadd", GmpProfileType::CONCURRENT_KERNEL);
    // vecAdd<<<6144, 256>>>(d_A_1, d_B_1, d_C_1, N);
    // GmpProfiler::getInstance()->popRange("vecadd", GmpProfileType::CONCURRENT_KERNEL);
    // GmpProfiler::getInstance()->pushRange("vecadd_thrust", GmpProfileType::CONCURRENT_KERNEL);
    // vecAdd_thrust(d_A_2, d_B_2, d_C_2, N);
    // GmpProfiler::getInstance()->popRange("vecadd_thrust", GmpProfileType::CONCURRENT_KERNEL);


    testLaunchOverhead(d_A_1, d_B_1, d_A_2, d_B_2, d_C_2, d_A_3, d_B_3, d_C_3, d_A_4);

    testBandwidthDist(d_A_5, d_B_5, d_C_5);

    // More Computation
    // GmpProfiler::getInstance()->pushRange("saxpy1_more_compute", GmpProfileType::CONCURRENT_KERNEL);
    // saxpy_more_compute<<<108, 128>>>(N, 2.0f, d_A_1, d_B_1);
    // GmpProfiler::getInstance()->popRange("saxpy1_more_compute", GmpProfileType::CONCURRENT_KERNEL);
    // GmpProfiler::getInstance()->pushRange("saxpy4_more_compute", GmpProfileType::CONCURRENT_KERNEL);
    // saxpy_more_compute<<<54, 256>>>(N, 2.0f, d_A_4, d_B_4);
    // GmpProfiler::getInstance()->popRange("saxpy4_more_compute", GmpProfileType::CONCURRENT_KERNEL);

    // More kernels in the range
    // GmpProfiler::getInstance()->pushRange("1saxpy", GmpProfileType::CONCURRENT_KERNEL);
    // saxpy<<<108, 128>>>(N, 2.0f, d_A_1, d_B_1);
    // GmpProfiler::getInstance()->popRange("1saxpy", GmpProfileType::CONCURRENT_KERNEL);
    // GmpProfiler::getInstance()->pushRange("2saxpy", GmpProfileType::CONCURRENT_KERNEL);
    // saxpy<<<108, 128>>>(N, 2.0f, d_A_1, d_B_1);
    // saxpy<<<108, 128>>>(N, 2.0f, d_A_2, d_B_2);
    // GmpProfiler::getInstance()->popRange("2saxpy", GmpProfileType::CONCURRENT_KERNEL);
    // GmpProfiler::getInstance()->pushRange("3saxpy", GmpProfileType::CONCURRENT_KERNEL);
    // saxpy<<<108, 128>>>(N, 2.0f, d_A_1, d_B_1);
    // saxpy<<<108, 128>>>(N, 2.0f, d_A_2, d_B_2);
    // saxpy<<<108, 128>>>(N, 2.0f, d_A_3, d_B_3);
    // GmpProfiler::getInstance()->popRange("3saxpy", GmpProfileType::CONCURRENT_KERNEL);

    // Cleanup
    cudaFree(d_A_1);
    cudaFree(d_B_1);
    cudaFree(d_C_1);
    cudaFree(d_A_2);
    cudaFree(d_B_2);
    cudaFree(d_C_2);
    cudaFree(d_A_3);
    cudaFree(d_B_3);
    cudaFree(d_C_3);
}

int main(int argc, char **argv)
{
  std::string outputPath = "";
  GmpOutputKernelReduction outputOption = GmpOutputKernelReduction::SUM;
  assert(argc >= 2);
  for(int i = 2; i < argc; i++)
  {
      if(strcmp(argv[i], "-o") == 0){
          printf("Setting output path: %s\n", argv[i + 1]);
          assert(i + 1 < argc);
          outputPath = argv[i + 1];
          i++;
      }
      else if(strcmp(argv[i], "--max") == 0){
          outputOption = GmpOutputKernelReduction::MAX;
          printf("Setting output option to MAX\n");
      }
      else if(strcmp(argv[i], "--mean") == 0){
          outputOption = GmpOutputKernelReduction::MEAN;
          printf("Setting output option to MEAN\n");
      }
      else if(strcmp(argv[i], "--sum") == 0){
          outputOption = GmpOutputKernelReduction::SUM;
          printf("Setting output option to SUM\n");
      }
      else{
          printf("Adding metric: %s\n", argv[i]);
          GmpProfiler::getInstance()->addMetrics(argv[i]);
      }  
  }



    // hello_kernel<<<1, 4>>>();
    int curr_pass = 0;
    GmpProfiler::getInstance()->init();

    printf("Starting profiling runs...\n");
    printf("current pass: %zu\n", curr_pass++);
    GmpProfiler::getInstance()->startRangeProfiling();
    launch_kernels();
    GmpProfiler::getInstance()->stopRangeProfiling();

    cudaDeviceSynchronize();
    GmpProfiler::getInstance()->decodeCounterData();
    GmpProfiler::getInstance()->printProfilerRanges(outputOption);
    GmpProfiler::getInstance()->produceOutput(outputOption);

    CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));

    return 0;
}

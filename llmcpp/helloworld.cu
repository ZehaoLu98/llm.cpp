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

void vecAdd_thrust(float* out, float* inp1, float* inp2, int N) {
    cub::CacheModifiedInputIterator<cub::LOAD_CS, float> inp1cs(inp1);
    cub::CacheModifiedInputIterator<cub::LOAD_CS, float> inp2cs(inp2);
    thrust::transform(thrust::cuda::par_nosync, inp1cs, inp1cs + N, inp2cs, out, thrust::plus<float>());
}

__global__ void multiply(const float *A, const float *B, float *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements)
        C[i] = A[i] * B[i];
}

__global__ void multiply_complex(float *A, float *B, float *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements)
    {
        C[i] = A[i] * B[i];
        A[i] = B[i] + C[i];
        B[i] = A[i] * C[i];
    }
}

__global__ void square(float *A, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
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

__global__ void sumReduction(float *input, float *output, int N)
{
    __shared__ float sdata[256]; // shared memory for partial sums
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < N) ? input[i] : 0.0f;
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

// #define N 13824*4*1024*16 // vector length, 3.456GB
#define N (4*128*768)
void launch_add()
{
    size_t size = N * sizeof(float);

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
    printf("Allocated device memory\n");

    // Copy from host to device
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

    // 3 sets of tests for GMP

    // Normal
    GmpProfiler::getInstance()->pushRange("saxpy1", GmpProfileType::CONCURRENT_KERNEL);
    GMP_TIMED("saxpy1", saxpy<<<108, 128>>>(N, 2.0f, d_A_1, d_B_1););
    GmpProfiler::getInstance()->popRange("saxpy1", GmpProfileType::CONCURRENT_KERNEL);
    // GmpProfiler::getInstance()->pushRange("saxpy2", GmpProfileType::CONCURRENT_KERNEL);
    // saxpy<<<108, 128*2>>>(N, 2.0f, d_A_2, d_B_2);
    // GmpProfiler::getInstance()->popRange("saxpy2", GmpProfileType::CONCURRENT_KERNEL);
    // GmpProfiler::getInstance()->pushRange("saxpy3", GmpProfileType::CONCURRENT_KERNEL);
    // saxpy<<<108, 128*4>>>(N, 2.0f, d_A_3, d_B_3);
    // GmpProfiler::getInstance()->popRange("saxpy3", GmpProfileType::CONCURRENT_KERNEL);
    // GmpProfiler::getInstance()->pushRange("saxpy4", GmpProfileType::CONCURRENT_KERNEL);
    // saxpy<<<54, 256>>>(N, 2.0f, d_A_4, d_B_4);
    // GmpProfiler::getInstance()->popRange("saxpy4", GmpProfileType::CONCURRENT_KERNEL);

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

int main(int argc, char** argv)
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
    launch_add();
    GmpProfiler::getInstance()->stopRangeProfiling();

    cudaDeviceSynchronize();
    GmpProfiler::getInstance()->decodeCounterData();
    GmpProfiler::getInstance()->printProfilerRanges(outputOption);
    GmpProfiler::getInstance()->produceOutput(outputOption);

    CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));

    return 0;
}

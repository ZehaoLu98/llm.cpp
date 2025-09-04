#include <stdio.h>
#include <stdlib.h>
// CUDA headers
#include <cuda.h>
#include <cuda_runtime.h>
#include <cupti.h>
#include <stdio.h>
#include <iomanip>
#include "gmp/profile.h"

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

// // CUPTI buffer request callback
// void CUPTIAPI bufferRequested(uint8_t **buffer, size_t *size, size_t *maxNumRecords)
// {
//     *size = 16 * 1024;
//     *buffer = (uint8_t *)malloc(*size);
//     *maxNumRecords = 0; // unlimited
// }

// // CUPTI buffer complete callback
// void CUPTIAPI bufferCompleted(CUcontext ctx, uint32_t streamId,
//                               uint8_t *buffer, size_t size, size_t validSize)
// {
//     CUptiResult status;
//     CUpti_Activity *record = NULL;

//     // Iterate over records in the buffer
//     while (1)
//     {
//         status = cuptiActivityGetNextRecord(buffer, validSize, &record);
//         if (status == CUPTI_SUCCESS)
//         {
//             if (record->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL)
//             {
//                 CUpti_ActivityKernel8 *kernel = (CUpti_ActivityKernel8 *)record;
//                 printf("CUPTI: Kernel \"%s\" launched on stream %u, grid (%u,%u,%u), block (%u,%u,%u)\n",
//                        kernel->name, kernel->streamId,
//                        kernel->gridX, kernel->gridY, kernel->gridZ,
//                        kernel->blockX, kernel->blockY, kernel->blockZ);
//             }
//         }
//         else if (status == CUPTI_ERROR_MAX_LIMIT_REACHED)
//         {
//             printf("CUPTI: Maximum buffer limit reached\n");
//             break;
//         }
//         else
//         {
//             CUPTI_CALL(status);
//         }
//     }

//     // Report dropped records
//     size_t dropped;
//     cuptiActivityGetNumDroppedRecords(ctx, streamId, &dropped);
//     if (dropped != 0)
//     {
//         printf("CUPTI: Dropped %zu activity records\n", dropped);
//     }

//     free(buffer);
// }

__global__ void vecAdd(const float *A, const float *B, float *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements)
        C[i] = A[i] + B[i];
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

#define N 4096 // vector length

void launch_add()
{
    size_t size = N * sizeof(float);

    // Host vectors
    float h_A[N], h_B[N], h_C[N];
    for (int i = 0; i < N; i++)
    {
        h_A[i] = i;
        h_B[i] = i * 10;
    }

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
    // Launch kernel
    int threadsPerBlock = 512;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    // GmpProfiler::getInstance()->pushRange("launch_add", GmpProfileType::CONCURRENT_KERNEL);
    // vecAdd<<<blocksPerGrid / 2, threadsPerBlock>>>(d_A_1, d_B_1, d_C_1, N / 2);
    // GmpProfiler::getInstance()->popRange("launch_add", GmpProfileType::CONCURRENT_KERNEL);
    // GmpProfiler::getInstance()->pushRange("launch_multiplication", GmpProfileType::CONCURRENT_KERNEL);
    // vecAdd<<<blocksPerGrid / 4, threadsPerBlock>>>(d_A_2, d_B_2, d_C_2, N / 4);
    // GmpProfiler::getInstance()->popRange("launch_multiplication", GmpProfileType::CONCURRENT_KERNEL);
    // GmpProfiler::getInstance()->pushRange("launch_square", GmpProfileType::CONCURRENT_KERNEL);
    // multiply_complex<<<blocksPerGrid, threadsPerBlock>>>(d_A_3, d_B_3, d_C_3, N);
    // GmpProfiler::getInstance()->popRange("launch_square", GmpProfileType::CONCURRENT_KERNEL);
    // GmpProfiler::getInstance()->pushRange("launch_sumReduction", GmpProfileType::CONCURRENT_KERNEL);
    // vecAdd<<<blocksPerGrid / 8, threadsPerBlock>>>(d_A_4, d_B_4, d_C_4, N / 8);
    // GmpProfiler::getInstance()->popRange("launch_sumReduction", GmpProfileType::CONCURRENT_KERNEL);

    // GmpProfiler::getInstance()->pushRange("launch_add");
    // vecAdd<<<blocksPerGrid / 2, threadsPerBlock>>>(d_A_1, d_B_1, d_C_1, N / 2);
    // vecAdd<<<blocksPerGrid / 4, threadsPerBlock>>>(d_A_2, d_B_2, d_C_2, N / 4);
    // multiply_complex<<<blocksPerGrid, threadsPerBlock>>>(d_A_3, d_B_3, d_C_3, N);
    // GmpProfiler::getInstance()->popRange();
    cudaDeviceSynchronize();

    hello_kernel<<<1, 4>>>();
    GmpProfiler::getInstance()->pushRange("allallall");
    GmpProfiler::getInstance()->pushRange("all", GmpProfileType::CONCURRENT_KERNEL);
    vecAdd<<<blocksPerGrid / 2, threadsPerBlock>>>(d_A_1, d_B_1, d_C_1, N / 2);
    vecAdd<<<blocksPerGrid / 4, threadsPerBlock>>>(d_A_2, d_B_2, d_C_2, N / 4);
    multiply_complex<<<blocksPerGrid, threadsPerBlock>>>(d_A_3, d_B_3, d_C_3, N);
    vecAdd<<<blocksPerGrid / 8, threadsPerBlock>>>(d_A_1, d_B_1, d_C_1, N / 8);
    GmpProfiler::getInstance()->popRange("all", GmpProfileType::CONCURRENT_KERNEL);
    GmpProfiler::getInstance()->popRange();
    cudaDeviceSynchronize();
    // Copy result back

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

int main()
{
    // CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));

    // CUPTI_CALL(cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted));

    hello_kernel<<<1, 4>>>();
    int curr_pass = 0;
    GmpProfiler::getInstance()->init();
    printf("Starting profiling runs...\n");
#ifdef USE_CUPTI
// while (GmpProfiler::getInstance()->isAllPassSubmitted() == false)
// {
#endif
    printf("current pass: %zu\n", curr_pass++);
    GmpProfiler::getInstance()->startRangeProfiling();
    for (int i = 0; i < 1000; i++)
    {
        launch_add();
    }
    GmpProfiler::getInstance()->stopRangeProfiling();
#ifdef USE_CUPTI
// }
#endif
    cudaDeviceSynchronize();
    GmpProfiler::getInstance()->decodeCounterData();
    GmpProfiler::getInstance()->printProfilerRanges();

    CUPTI_CALL(cuptiActivityFlushAll(1));

    CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));

    return 0;
}

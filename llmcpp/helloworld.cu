#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cupti.h>
#include <stdio.h>
#include <iomanip>
#include <cupti_events.h>
#include "gmp/profile.h"

#define CUPTI_CALL(call)                                                         \
    do                                                                           \
    {                                                                            \
        CUptiResult _status = call;                                              \
        if (_status != CUPTI_SUCCESS)                                            \
        {                                                                        \
            const char *errstr;                                                  \
            cuptiGetResultString(_status, &errstr);                              \
            fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n", \
                    __FILE__, __LINE__, #call, errstr);                          \
            exit(-1);                                                            \
        }                                                                        \
    } while (0)

// Simple CUDA kernel
__global__ void hello_kernel()
{
    printf("Hello World from GPU thread %d!\n", threadIdx.x);
}

// CUPTI buffer request callback
void CUPTIAPI bufferRequested(uint8_t **buffer, size_t *size, size_t *maxNumRecords)
{
    *size = 16 * 1024;
    *buffer = (uint8_t *)malloc(*size);
    *maxNumRecords = 0; // unlimited
}

// CUPTI buffer complete callback
void CUPTIAPI bufferCompleted(CUcontext ctx, uint32_t streamId,
                              uint8_t *buffer, size_t size, size_t validSize)
{
    CUptiResult status;
    CUpti_Activity *record = NULL;

    // Iterate over records in the buffer
    while (1)
    {
        status = cuptiActivityGetNextRecord(buffer, validSize, &record);
        if (status == CUPTI_SUCCESS)
        {
            if (record->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL)
            {
                CUpti_ActivityKernel8 *kernel = (CUpti_ActivityKernel8 *)record;
                printf("CUPTI: Kernel \"%s\" launched on stream %u, grid (%u,%u,%u), block (%u,%u,%u)\n",
                       kernel->name, kernel->streamId,
                       kernel->gridX, kernel->gridY, kernel->gridZ,
                       kernel->blockX, kernel->blockY, kernel->blockZ);
            }
        }
        else if (status == CUPTI_ERROR_MAX_LIMIT_REACHED)
        {
            printf("CUPTI: Maximum buffer limit reached\n");
            break;
        }
        else
        {
            CUPTI_CALL(status);
        }
    }

    // Report dropped records
    size_t dropped;
    cuptiActivityGetNumDroppedRecords(ctx, streamId, &dropped);
    if (dropped != 0)
    {
        printf("CUPTI: Dropped %zu activity records\n", dropped);
    }

    free(buffer);
}

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

__global__ void multiplyABCDEFG(const float *A, const float *B, float *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements)
        C[i] = A[i] * B[i];
}

#define N 16 // vector length

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
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    // Copy from host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 8;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    GmpProfiler::getInstance()->pushRange("launch_add1");
    vecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    multiplyABCDEFG<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    vecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    GmpProfiler::getInstance()->popRange();
    GmpProfiler::getInstance()->pushRange("launch_add2");

    multiplyABCDEFG<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    vecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    multiply<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    GmpProfiler::getInstance()->popRange();

    // Copy result back
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Print result
    for (int i = 0; i < N; i++)
    {
        printf("%2d: %5.1f + %5.1f = %5.1f\n", i, h_A[i], h_B[i], h_C[i]);
    }

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main()
{
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));

    CUPTI_CALL(cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted));

    hello_kernel<<<1, 4>>>();
    int curr_pass = 0;
    while (GmpProfiler::getInstance()->isAllPassSubmitted() == false)
    {
        printf("current pass: %zu\n", curr_pass++);
        GmpProfiler::getInstance()
            ->startRangeProfiling();
        launch_add();
        GmpProfiler::getInstance()->stopRangeProfiling();
    }
    cudaDeviceSynchronize();
    GmpProfiler::getInstance()->printProfilerRanges();

    // CUPTI_CALL(cuptiActivityFlushAll(0));

    CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));

    return 0;
}

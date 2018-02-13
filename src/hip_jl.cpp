#include <hip/hip_runtime.h>
#include <hipblas.h>

#define THREADS_PER_BLOCK_X     32
#define THREADS_PER_BLOCK_Y     32
#define THREADS_PER_BLOCK_Z     1

__global__ void matrixTranspose(hipLaunchParm lp, float *out, float *in, const int nrow_in, const int ncol_in)
{
    uint32_t x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    uint32_t y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    out[y + x * ncol_in] = in[x + y * nrow_in];
}



__global__ void hip_sgemm_kernel_naive(hipLaunchParm lp, const int M,
                                            const int N, const int K,
                                            const float alpha,
                                            float *A, float *B,
                                            const float beta,
                                            float *C) //lda, ldb, and ldc are expected to equal stride.
{
        //column major NN
    int idx_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int idx_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
//    int dim_x = hipGridDim_x * hipBlockDim_x;//hip grid dim is 1, right?
//    int myIdx = idx_y * dim_x + idx_x;
    int myIdx = idx_y + idx_x * M;

    float local_c = beta * C[myIdx];
    int idx_A = idx_y;
    int idx_B = idx_x * K;

    for(int k = 0; k < K; k++) {
        local_c += alpha * A[idx_A] * B[idx_B + k];
        idx_A += M;
    }

    C[myIdx] = local_c;
}

__global__ void hip_sgemm_kernel_tiled(hipLaunchParm lp, const int M,
                                            const int N, const int K,
                                            const float alpha,
                                            float *A, float *B,
                                            const float beta,
                                            float *C) //lda, ldb, and ldc are expected to equal stride.
{
//Assumes that TILE_LENGTH == hipBlockDim_x == hipBlockDim_y

//column major NN
    int idx_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int idx_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
//    int dim_x = hipGridDim_x * hipBlockDim_x;//hip grid dim is 1, right?
//    int myIdx = idx_y * dim_x + idx_x;
    int id_local = hipThreadIdx_y + THREADS_PER_BLOCK_Y * hipThreadIdx_x;
    int id_localT = hipThreadIdx_x + THREADS_PER_BLOCK_X * hipThreadIdx_y;//tranposed id
    int myIdx = idx_y + idx_x * M;

    float local_c = 0.0;

    __shared__ float tileA[THREADS_PER_BLOCK_X * THREADS_PER_BLOCK_Y];
    __shared__ float tileB[THREADS_PER_BLOCK_X * THREADS_PER_BLOCK_Y];
    int num_tiles = K/THREADS_PER_BLOCK_X;
    // careful!!! How about, X and Y defined as is for A; tranpose for B?
    for (int t = 0; t < num_tiles; t++)
    {
        
        const int t_col = t*THREADS_PER_BLOCK_X;
        tileA[id_localT] = A[(t_col + hipThreadIdx_x)*M + idx_y];
        tileB[id_local ] = B[t_col + hipThreadIdx_y + K*idx_x];

        __syncthreads();

        int idxA = hipThreadIdx_y * THREADS_PER_BLOCK_X ;
        int idxB = hipThreadIdx_x * THREADS_PER_BLOCK_X ;

        #pragma unroll
        for (int k = 0; k < THREADS_PER_BLOCK_X; k++)
        {
            local_c += tileA[k + idxA] * tileB[k + idxB];
        }

        __syncthreads();

    }

    C[myIdx] = alpha * local_c + beta * C[myIdx];
}

extern "C" void hipSgemm( bool, bool, const int, const int, const int,
                            const float, float*, float*, const float, float* );
void hipSgemm( bool tA, bool tB, const int m, const int n, const int k,
                            const float alpha, float* A, float* B, const float beta, float* C )
{
    hipblasOperation_t trans_a, trans_b;
//    int size_a, size_b;
    int lda, ldb, ldc = m;
//    int a_stride_1, b_stride_1, a_stride_2, b_stride_2;
    if (tA == true)
    {
        trans_a = HIPBLAS_OP_T;
        lda = k;
//        size_a = m * lda;
//        a_stride_1 = lda;
//        a_stride_2 = 1;
    }
    else
    {
        trans_a = HIPBLAS_OP_N;
        lda = m;
//        size_a = k * lda;
//        a_stride_1 = 1;
//        a_stride_2 = lda;
    }
    
    if (tB == true)
    {
        trans_b = HIPBLAS_OP_T;
        ldb = k;
//        size_b = k * ldb;
//        b_stride_1 = ldb;
//        b_stride_2 = 1;
    }
    else
    {
        trans_b = HIPBLAS_OP_N;
        ldb = k;
//        size_a = n * ldb;
//        b_stride_1 = 1;
//        b_stride_2 = ldb;
    }
    

    hipblasHandle_t handle;
    hipblasCreate(&handle);
    hipblasSgemm(handle, trans_a, trans_b, m, n, k, &alpha,
        A, lda,
        B, ldb, &beta,
        C, ldc);

    hipblasDestroy(handle);

}

extern "C" void setDevice(const int);
void setDevice(const int i)
{
    hipSetDevice(i);
}
extern "C" void synchronize();
void synchronize()
{
    hipDeviceSynchronize();
}

extern "C" void naive_A_mul_B(float*, float*, float*, const int, const int, const int);
void naive_A_mul_B(float* C, float* A, float* B, const int M, const int N, const int K)
{
    const float alpha = 1.0;
    const float beta = 0.0;

    hipLaunchKernel(hip_sgemm_kernel_naive,
                  dim3(N/THREADS_PER_BLOCK_X, M/THREADS_PER_BLOCK_Y),
                  dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y),
                  0, 0,
                  M, N, K, alpha, A, B, beta, C);

}
extern "C" void naive_sgemm(const int, const int, const int, const float, float*, float*, const float, float*);
void naive_sgemm(const int M, const int N, const int K, const float alpha, float* A, float* B, const float beta, float* C)
{

    hipLaunchKernel(hip_sgemm_kernel_naive,
                  dim3(N/THREADS_PER_BLOCK_X, M/THREADS_PER_BLOCK_Y),
                  dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y),
                  0, 0,
                  M, N, K, alpha, A, B, beta, C);

}



extern "C" void tiled_sgemm(const int, const int, const int, const float, float*, float*, const float, float*);
void tiled_sgemm(const int M, const int N, const int K, const float alpha, float* A, float* B, const float beta, float* C)
{

    hipLaunchKernel(hip_sgemm_kernel_tiled,
                  dim3(N/THREADS_PER_BLOCK_X, M/THREADS_PER_BLOCK_Y),
                  dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y),
                  0, 0,
                  M, N, K, alpha, A, B, beta, C);

}





extern "C" void hipMemDevToHost(float*, float*, const int);
void hipMemDevToHost(float* A, float* B, const int n)
{
  hipMemcpy(A, B, n*sizeof(float), hipMemcpyDeviceToHost);
}

extern "C" float* hipArray(float*, const int);
float* hipArray(float* A, const int n)
{
    float* gpuArray;
    hipMalloc((void**)&gpuArray, n * sizeof(float));
    hipMemcpy(gpuArray, A, n*sizeof(float), hipMemcpyHostToDevice);
    return gpuArray;
}
extern "C" float* hipArrayUninit(const int);
float* hipArrayUninit(const int n)
{
    float* gpuArray;
    hipMalloc((void**)&gpuArray, n * sizeof(float));
    return gpuArray;
}
extern "C" void hipDelete(float*);
void hipDelete(float* A)
{
    hipFree(A);
}


extern "C" void tranpose(float*, float*, const int, const int);
void tranpose(float* gpuTransposeMatrix, float* gpuMatrix, const int nrow_in, const int ncol_in) {
  
    // Lauching kernel from host
    hipLaunchKernel(matrixTranspose,
                  dim3( ncol_in/THREADS_PER_BLOCK_X, nrow_in/THREADS_PER_BLOCK_Y),
                  dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y),
                  0, 0,
                  gpuTransposeMatrix , gpuMatrix, nrow_in, ncol_in);


}
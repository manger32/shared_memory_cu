/*В данном задании требуется представить 2 варианта программы для видеокарты: 1) максимально простой и короткий; и 2) быстрый, использующий разделяемую память.
Запрограммируйте генерацию случайных входных данных для алгоритма и автоматическую проверку корректности работы программы.
Выполните теоретическую оценку производительности обоих вариантов алгоритма. Укажите в отчете, насколько теоретическая оценка отличается от практической. */

/*Реализуйте умножение длинной матрицы, хранящейся по столбцам, на длинный вектор*/
#include <iostream>
#define N 16 //shortest dimension of A: 32
#define M 2*(102400*8) // 1

using namespace std;

#define CHECK(value) {                                          \
    cudaError_t _m_cudaStat = value;                                        \
    if (_m_cudaStat != cudaSuccess) {                                       \
        cout<< "Error:" << cudaGetErrorString(_m_cudaStat) \
            << " at line " << __LINE__ << " in file " << __FILE__ << "\n"; \
        exit(1);                                                            \
    } }

__global__ void Multiply(int *A, int *B, int *C){
// calculate the row & col index of the element
    int row = blockIdx.x*blockDim.x + threadIdx.x;
    if (row >= N)
      return;
    int result = 0;
// product between row of a and b
    for(int k = 0; k < M; ++k)
    {
        result += A[row + k*N] * B[k];
        //printf("%d ", result);
    }
    C[row] = result;
}

__global__ void Multiply_smart_string(int *A, int *B, int *C){
   int col = blockIdx.x*blockDim.x + threadIdx.x;
   if (col >= M)
    return;
   int dev_private = 0;
   __shared__ int dev_shared;
   for (int j = 0; j < M/blockDim.x; ++j)
   {
     int addition = A[(j*blockDim.x+threadIdx.x)*N+blockIdx.x] * B[j*blockDim.x+threadIdx.x];
     dev_private += addition;
   }
   if (threadIdx.x == 0)
     dev_shared = 0;
   __syncthreads();
   atomicAdd(&dev_shared, dev_private);
   __syncthreads();
   if (threadIdx.x == 0)
     C[blockIdx.x] = dev_shared;
}

__global__ void Multiply_smart_column(int *A, int *B, int *C){
   int global_id = blockIdx.x*blockDim.x + threadIdx.x;
   int global_trd_cnt = blockDim.x*gridDim.x;
   __shared__ int dev_shared_res[N];
   int addition = 0;
   if (threadIdx.x < N)
       dev_shared_res[threadIdx.x] = 0;

   for (int j = 0; j < M/(global_trd_cnt/N); ++j)
   {
     int super_global_id = global_id + j*global_trd_cnt;
     int row = super_global_id % N;
     int col = super_global_id / N;
     addition += A[col*N + row] * B[col];
   }
   __syncthreads();
   atomicAdd(&dev_shared_res[threadIdx.x % N], addition);
   __syncthreads();
   if (threadIdx.x < N)
     atomicAdd(&C[threadIdx.x], dev_shared_res[threadIdx.x]);
}

int main(int argc, char **argv)
{
  srand(time(NULL));
  int *A = new int [N*M];
  int *b = new int [M];
  int *res_CPU = new int[N];
  int *res_GPU = new int[N];
  int i, j;
  for(i = 0; i < N; ++i)
      res_CPU[i] = 0;
  for(i = 0; i < N; ++i)
  {
      for(j = 0; j < M; ++j)
      {
          A[i + j*N] = rand() % 10; // % 3 - 1; //1;
          //cout << A[i*N + j] << " ";
      }
      //cout << endl;
  }
  //cout << endl;
  for(i = 0; i < M; ++i)
  {
      b[i] = rand() % 10; // % 3 - 1; //1;
      //cout << b[i] << " ";
  }
  //cout << endl;
  // shared memory: t = 0..32 - warp
  clock_t startCPU = clock();
  for(i = 0; i < N; ++i)
  {
      for(j = 0; j < M; ++j)
          res_CPU[i] += A[i + j*N]*b[j];
      //cout << "Res_CPU[" << i << "] = " << res_CPU[i] << " " << endl;
  }
  double elapsedTimeCPU = (double)(clock()-startCPU)/CLOCKS_PER_SEC;
  cout << "CPU product time = " << elapsedTimeCPU*1000 << " ms\n";

  int (*aA), (*aB), (*aRes);
  cudaEvent_t startCUDA, stopCUDA;
  float elapsedTimeCUDA;
  cudaEventCreate(&startCUDA);
  cudaEventCreate(&stopCUDA);

  CHECK(cudaMalloc((void**)&aA, (N*M)*sizeof(int)));
  CHECK(cudaMalloc((void**)&aB, (M)*sizeof(int)));
  CHECK(cudaMalloc((void**)&aRes, (N)*sizeof(int)));

  CHECK(cudaMemcpy(aA, A, (N*M)*sizeof(int), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(aB, b, (M)*sizeof(int), cudaMemcpyHostToDevice));
  CHECK(cudaMemset(aRes, 0, (N)*sizeof(int)));

  //int numBlocks = 1;
  //dim3 threadsPerBlock(N,N);
  cudaEventRecord(startCUDA,0);
  //Multiply<<<(N+511)/512, 512>>>(aA,aB,aRes);
  //Multiply_smart_string<<<N, 512>>>(aA,aB,aRes);
  Multiply_smart_column<<<8, 1024>>>(aA,aB,aRes); //N*M/1024
  cudaEventRecord(stopCUDA,0);
  cudaEventSynchronize(stopCUDA);
  CHECK(cudaGetLastError());
  CHECK(cudaMemcpy(res_GPU, aRes, N*sizeof(int), cudaMemcpyDeviceToHost));

  cudaEventElapsedTime(&elapsedTimeCUDA, startCUDA, stopCUDA);

  cout << "CUDA product time = " << elapsedTimeCUDA << " ms\n";
  cout << "CUDA memory throughput = " << 3*N*sizeof(float)/elapsedTimeCUDA/1024/1024/1.024 << " Gb/s\n";
  for (i = 0; i < N; i++) {
    //cout << "Res_GPU[" << i << "] = " << res_GPU[i] << " " << endl;
  }
  for (i = 0; i < N; i++) {
    if (res_CPU[i] != res_GPU[i])
    {
      cout << "Not equal. Try again, again." << endl;
      break;
    }
  }
  CHECK(cudaFree(aA));
  CHECK(cudaFree(aB));
  CHECK(cudaFree(aRes));
  return 0;
}

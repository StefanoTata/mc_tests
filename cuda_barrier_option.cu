#include <curand.h>
#include <stdio.h>
#include <stdlib.h>

#define check_cuda(cuda_status){\
  if(cuda_status != cudaSuccess){\
    fprintf(stderr, "Error %s:%d ", __FILE__, __LINE__);\
    fprintf(stderr, "code: %d, reason: %s\n", cuda_status, cudaGetErrorString(cuda_status));\
    return -1;\
  }\
}

__global__ void mc_kernel(
  float* d_rng
  , float* d_s
  , float T
  , float K
  , float B
  , float S0
  , float sigma
  , float mu
  , float r
  , float dt
  , unsigned N_STEPS
  , unsigned N_PATHS
){
  unsigned s_idx = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned n_idx = threadIdx.x + blockIdx.x * blockDim.x;
  float s_curr = S0;
  if(s_idx  < N_PATHS){
    int n = 0;
    do{
      s_curr = s_curr + mu * s_curr * dt + sigma * s_curr * d_rng[n_idx];
      n_idx++;
      n++;
    } while (n < N_STEPS && s_curr > B);
    
    float payoff = (s_curr > K ? s_curr - K : 0.0f);
    __syncthreads();
    d_s[s_idx] = exp(-r * T) * payoff;
  }
}

int main(){
  const unsigned N_PATHS = 5e7;
  const unsigned N_STEPS = 365;
  const unsigned N_NORMALS = N_PATHS * N_STEPS;

  const float T = 1.0f;
  const float K = 100.0f;
  const float B = 95.0f;
  const float S0 = 100.0f;
  const float sigma = 0.2f;
  const float mu = 0.1f;
  const float r = 0.05f;

  const float dt = float(T)/float(N_STEPS);
  const float sqrdt = sqrt(dt);

  const unsigned BLOCK_SIZE = 1024;
  const unsigned GRID_SIZE = ceil(float(N_PATHS) / float(BLOCK_SIZE));

  float* s = (float*)malloc(N_PATHS * sizeof(float));

  float* d_rng;
  cudaError_t res = cudaMalloc((void**) &d_rng, N_NORMALS * sizeof(float));
  check_cuda(res);
  
  float* d_s;
  res = cudaMalloc((void**) &d_s, N_PATHS * sizeof(float));
  check_cuda(res);

  curandGenerator_t gen;
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MTGP32);
  curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
  curandGenerateNormal(gen, d_rng, N_NORMALS, 0.0f, sqrdt);
  
  mc_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_rng, d_s, T, K, B, S0, sigma, mu, r, dt, N_STEPS, N_PATHS);
  cudaDeviceSynchronize();

  res = cudaMemcpy(s, d_s, N_PATHS * sizeof(float), cudaMemcpyDeviceToHost);
  check_cuda(res);

  float tmp_sum = 0.0f;
  for(int i=0; i < N_PATHS; ++i) tmp_sum += s[i];
  tmp_sum /= N_PATHS;
  
  printf("Price: %f\n", tmp_sum);

  return 0;
}

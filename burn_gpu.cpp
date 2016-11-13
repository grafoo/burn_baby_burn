#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <iostream>

void print_matrix(std::string name, float *matrix, size_t len) {
  std::cout << name << std::endl;
  for (int i = 0; i < name.length(); i++) {
    std::cout << "-";
  }
  std::cout << std::endl;
  for (size_t i = 0; i < len * len; i++) {
    std::cout << matrix[i] << " ";
    if ((i + 1) % len == 0) {
      std::cout << std::endl;
    }
  }
  std::cout << std::endl;
}

int main(int argc, char **argv) {
  size_t matrix_size;
  size_t matrix_len;

  if (argc > 1) {
    matrix_len = std::atoll(argv[1]);
    matrix_size = matrix_len * matrix_len;
  } else {
    return 1;
  }

  float *host_a = new float[matrix_size]();
  float *host_b = new float[matrix_size]();
  float *host_result = new float[matrix_size]();
  float *device_a;
  float *device_b;
  float *device_result;
  cudaError_t err;
  curandGenerator_t rng;
  cublasHandle_t hndl;
  cublasCreate(&hndl);

  cudaMalloc((void **)&device_a, matrix_size * sizeof(float));
  cudaMalloc((void **)&device_b, matrix_size * sizeof(float));
  cudaMalloc((void **)&device_result, matrix_size * sizeof(float));

  int device_count = 0;
  err = cudaGetDeviceCount(&device_count);
  if (err != cudaSuccess) {
    std::cout << cudaGetErrorString(err) << std::endl;
    return 1;
  }

  for (int i = 0; i < device_count; i++) {
    std::cout << "using device: " << i << std::endl;
    std::cout << "=============" << std::endl << std::endl;
    cudaSetDevice(i);

    curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(rng, 42ULL);

    curandGenerateUniform(rng, device_a, matrix_size);
    curandGenerateUniform(rng, device_b, matrix_size);

    cudaMemcpy(host_a, device_a, matrix_size * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(host_b, device_b, matrix_size * sizeof(float),
               cudaMemcpyDeviceToHost);

    print_matrix("A", host_a, matrix_len);
    print_matrix("B", host_b, matrix_len);

    const float alpha = 1;
    const float beta = 0;

    cublasSgemm(hndl, CUBLAS_OP_N, CUBLAS_OP_N, matrix_len, matrix_len,
                matrix_len, &alpha, device_a, matrix_len, device_b, matrix_len,
                &beta, device_result, matrix_len);

    cudaMemcpy(host_result, device_result, matrix_size * sizeof(float),
               cudaMemcpyDeviceToHost);

    print_matrix("RESULT", host_result, matrix_len);
    curandDestroyGenerator(rng);
    cublasDestroy(hndl);
    delete[] host_a;
    delete[] host_b;
    delete[] host_result;
    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_result);
    cudaDeviceReset();
  }

  return 0;
}

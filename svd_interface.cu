#include "svd_interface.cuh"

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <iostream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <string.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

__global__ void uxs(float* u, float* s, float* out, int m, int n, int k)
{
	// s = m*n
	// u = m*m
	// v = n*n
	// u = m*k
	const int idx = threadIdx.x;
	for (size_t i = 0; i < k; i++)
	{
		const int ele_index = idx * n + i;
		out[idx * k + i] = u[ele_index] * s[i];
	}
}

void svd_decompose(float* A, float* left, float* right, int m, int n, int k)
{
	double accum; 
	cusolverDnHandle_t cusolverH; 
	cublasHandle_t cublasH; 
	cublasStatus_t cublas_status = CUBLAS_STATUS_SUCCESS;
	cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
	cudaError_t cudaStat = cudaSuccess;

	const int lda = m;
	float* U, * VT, * S; 
	U = (float*)malloc(lda * m * sizeof(float));
	VT = (float*)malloc(lda * n * sizeof(float));
	S = (float*)malloc(n * sizeof(float));

	float* d_A, * d_U, * d_VT, * d_S; 
	int* devInfo; 
	float* d_work, * d_rwork; 
	float* d_W; 
	int lwork = 0;
	int info_gpu = 0; 
	const float h_one = 1;
	const float h_minus_one = -1;

	// create cusolver and cublas handle
	cusolver_status = cusolverDnCreate(&cusolverH);
	cublas_status = cublasCreate(&cublasH);

	// prepare memory on the device
	cudaStat = cudaMalloc((void**)&d_A, sizeof(float) * lda * n);
	cudaStat = cudaMalloc((void**)&d_S, sizeof(float) * n);
	cudaStat = cudaMalloc((void**)&d_U, sizeof(float) * lda * m);
	cudaStat = cudaMalloc((void**)&d_VT, sizeof(float) * lda * n);
	cudaStat = cudaMalloc((void**)&devInfo, sizeof(int));
	cudaStat = cudaMalloc((void**)&d_W, sizeof(float) * lda * n);
	cudaStat = cudaMemcpy(d_A, A, sizeof(float) * lda * n,
		cudaMemcpyHostToDevice); // copy A- >d_A

			// compute buffer size and prepare workspace
	cusolver_status = cusolverDnSgesvd_bufferSize(cusolverH, m, n,
		&lwork);
	cudaStat = cudaMalloc((void**)&d_work, sizeof(float) * lwork);

	signed char jobu = 'A'; 
	signed char jobvt = 'A'; 



	auto st = std::chrono::system_clock::now();
	std::cout << "Start to time!" << std::endl;

	printf("SVD time :sec .\n"); 

	cusolver_status = cusolverDnSgesvd(cusolverH, jobu, jobvt,
		m, n, d_A, lda, d_S, d_U, lda, d_VT, lda, d_work, lwork,
		d_rwork, devInfo);
	cudaStat = cudaDeviceSynchronize();

	auto end = std::chrono::system_clock::now();

	auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(end - st);
	std::cout << "cusolverDnSgesvd time: " << dur.count() << " ms" << std::endl;

	cudaStat = cudaMemcpy(U, d_U, sizeof(float) * lda * m,
		cudaMemcpyDeviceToHost); 
	cudaStat = cudaMemcpy(VT, d_VT, sizeof(float) * lda * n,
		cudaMemcpyDeviceToHost); 
	cudaStat = cudaMemcpy(S, d_S, sizeof(float) * n,
		cudaMemcpyDeviceToHost); 
	cudaStat = cudaMemcpy(&info_gpu, devInfo, sizeof(int),
		cudaMemcpyDeviceToHost); 
	printf(" after gesvd : info_gpu = %d\n", info_gpu);
	
	cublas_status = cublasSdgmm(cublasH, CUBLAS_SIDE_LEFT, n, n,
		d_VT, lda, d_S, 1, d_W, lda); 
	cudaStat = cudaMemcpy(d_A, A, sizeof(float) * lda * n,
		cudaMemcpyHostToDevice); 
		
	cublas_status = cublasSgemm_v2(cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
		m, n, n, &h_minus_one, d_U, lda, d_W, lda, &h_one, d_A, lda);

	float dR_fro = 0.0; 
	// compute the norm of the difference d_A -d_U *d_S * d_VT
	cublas_status = cublasSnrm2_v2(cublasH, lda * n, d_A, 1, &dR_fro);
	printf("|A - U*S*VT| = %E \n", dR_fro); // print the norm

	float* d_left, * d_right;
	cudaStat = cudaMalloc((void**)&d_left, sizeof(float) * m * k);

	uxs << <1, k >> > (d_U, d_S, d_left, m, n, k);

	if (left) {
		//std::cout << "left" << std::endl;
		cudaStat = cudaMemcpy(left, d_left, sizeof(float) * m * k, cudaMemcpyDeviceToHost);
	}

	cudaStat = cudaMemcpy(right, d_VT, sizeof(float) * k * n, cudaMemcpyDeviceToHost);

	cudaFree(d_A);
	cudaFree(d_S);
	cudaFree(d_U);
	cudaFree(d_VT);
	cudaFree(devInfo);
	cudaFree(d_work);
	cudaFree(d_rwork);
	cudaFree(d_W);
	cudaFree(d_left);
	cublasDestroy(cublasH);
	cusolverDnDestroy(cusolverH);
	cudaDeviceReset();

}
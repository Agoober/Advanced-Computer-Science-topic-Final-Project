    /*
Matrix Factorization Simple
*/

#include "mm_helper.hpp"
#include "sparse_representation.hpp"
#include <iostream>

#include <getopt.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include <curand.h>
#include <curand_kernel.h>

#define threadNum 256

__global__ void generate_random_numbers(float* numbers, int Np) {

    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < Np) {

        curandState state;

        curand_init(clock64(), i, 0, &state);

        numbers[i] = curand_uniform(&state);
    }
}


__global__ void device_mf(unsigned int* mat_row_indx, unsigned int* mat_col_id, double* mat_values, unsigned int mat_nrows, 
     double * S_mat, double i, double j, double k, double *A, double *B, double *C){

    unsigned int threadId = blockIdx.x * threadNum + threadIdx.x;
    unsigned int tx = threadIdx.x;
    double S,G,a,b,c;

    for(int t = threadId; t<threadId + 5; ++t) {
        curandState state;
        curand_init(clock64(), 0, 0, &state);

        a = curand(&state)%100;
        b = curand(&state)%100;
        c = curand(&state)%100;

        for (unsigned int r = 0; r < mat_nrows; ++r) {
            unsigned int row_start = mat_row_indx[r];
            unsigned int row_end = mat_row_indx[r + 1];
            //dmat_out[r * K + threadId] = 0;
            for (unsigned int p = row_start; p < row_end; ++p) {
                //unsigned int col_id = mat_col_id[p];
                double val = mat_values[p];
                S += a*i + b*j + c*k - val;
            }
        }
        S_mat[t] = S;
        A[t] = a;
        B[t] = b;
        C[t] = c;
    }
   
}



int main(int argc, char *argv[]) {

    if(argc < 2) {
        std::cerr << "usage ./exec inputfile " << std::endl;
        exit(-1);
    }

    CSR mat = read_matrix_market_to_CSR(argv[1]);
    std::cout << mat.nrows << ' ' << mat.ncols << ' ' << mat.nnz << ' ' << '\n';

    

    unsigned int* row_indx_device;
    unsigned int* col_id_device;
    double* values_device;

    cudaMalloc((unsigned int**)&row_indx_device, (mat.nrows + 1) * sizeof(unsigned int));
    cudaMemcpy(row_indx_device, mat.row_indx, (mat.nrows + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice);

    cudaMalloc((unsigned int**)&col_id_device, mat.nnz * sizeof(unsigned int));
    cudaMemcpy(col_id_device, mat.col_id, mat.nnz * sizeof(unsigned int), cudaMemcpyHostToDevice);

    cudaMalloc((double**)&values_device, mat.nnz * sizeof(double));
    cudaMemcpy(values_device, mat.values, mat.nnz * sizeof(double), cudaMemcpyHostToDevice);

    double *dSmat, *dA, *dB, *dC;
    cudaMalloc((double**)&dSmat, 5*mat.nnz*sizeof(double) );
    cudaMalloc((double**)&dA, 5*mat.nnz*sizeof(double) );
    cudaMalloc((double**)&dB, 5*mat.nnz*sizeof(double) );
    cudaMalloc((double**)&dC, 5*mat.nnz*sizeof(double) );

    double di, dj,dk;
    di = rand()/100;
    dj = rand()/100;
    dk = rand()/100;

    dim3 threads = dim3(threadNum, 1,1);
    dim3 blocks = dim3(mat.nnz/threadNum + 1,1,1);

    device_mf<<<blocks, threads>>>(row_indx_device, col_id_device, values_device, mat.nrows,dSmat,di, dj, dk, dA, dB, dC);
    
    
    cudaDeviceSynchronize();
    double *hSmat, *hA, *hB, *hC;
  
    cudaMemcpy(hSmat, dSmat, mat.nnz * 5 * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(hA, dA, mat.nnz * 5 * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(hB, dB, mat.nnz * 5 * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(hC, dC, mat.nnz * 5 * sizeof(double), cudaMemcpyDeviceToHost);

    int tem = hSmat[0];
    double a_out, b_out, c_out;
    for (int n = 0; n<mat.nnz*5; n++) {
        if (tem>hSmat[n]) {
            tem = hSmat[n];
            a_out = hA[n];
            b_out = hB[n];
            c_out = hB[n];
        }
    }

    double result = a_out*di + b_out*dj + c_out*dk;

    std::cout<< result;


    free(mat.row_indx);
    free(mat.col_id);
    free(mat.values);
    free(hSmat);
    free(hA);
    free(hB);
    free(hC);
    cudaFree(dSmat);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    cudaFree(row_indx_device);
    cudaFree(col_id_device);
    cudaFree(values_device);




    return 0;
}
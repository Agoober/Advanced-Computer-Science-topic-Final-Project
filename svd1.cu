#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>

#include "svd_interface.cuh"

#include "mm_helper.hpp"

# define BILLION 1000000000L

void get_matrix_info(std::ifstream& fin, int& m, int& n, int& l)
{
	while (fin.peek() == '%') {
		fin.ignore(2048, '\n');
	}

	
	fin >> m >> n >> l;
	std::cout << "read : " << m << " " << n << " " << l << std::endl;
}

void get_matrix_data(std::ifstream& fin, float* A, int cols, int l)
{
	for (unsigned int i = 0; i < l; i++) {
		unsigned int m;
		unsigned int n;
		double val;
		fin >> m >> n >> val;

		A[(m - 1) * cols + n - 1] = val;
	}
}

int main(int argc, char* argv[])
{
	if (argc != 3)
	{
		std::cout << "Input number error." << std::endl;
		return 0;
	}

	int k = 20;

	std::string path;
	path = argv[1];
	k = std::atoi(argv[2]);

	struct timespec start, stop; 
	double accum; 

	int m = 100; 
	int n = 100; 
	int lda = m; 
	int L = 0;
	std::ifstream fin(path);
	if (!fin.is_open())
	{
		std::cout << path << " open matrix failed." << std::endl;
		return 0;
	}

	get_matrix_info(fin, m, n, L);

	// declare the factorized matrix A, orthogonal matrices U, VT
	float* A, * U, * VT, * S; 
	A = (float*)malloc(m * n * sizeof(float));

	get_matrix_data(fin, A, n, L);
	
	float* left = (float*)malloc(sizeof(float) * m * k);
	float* right = (float*)malloc(sizeof(float) * k * n);

	printf("m = %d n = %d k = %d\n", m, n, k);


	svd_decompose(A, left, right, m, n, k);

	std::cout << "Left: " << std::endl;
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < k; j++)
		{
			std::cout << left[i * k + j] << " ";
		}

		std::cout << std::endl;
	}

	std::cout << "Right: " << std::endl;
	for (int i = 0; i < k; i++)
	{
		for (int j = 0; j < n; j++)
		{
			std::cout << right[i * n + j] << " ";
		}

		std::cout << std::endl;
	}

	fin.close();
	return 0;
}
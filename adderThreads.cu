#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
//#include <helper_cuda.h>
//#include <helper_functions.h>

#define N 512
__global__ void add(int *a, int *b, int *c) {
	c[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];
	//putc(a[threadIdx.x], fp);
	//putc(' ', fp);
	//putc(b[threadIdx.x], fp);
	//putc(' ', fp);
	//putc(c[threadIdx.x], fp);
	//putc(',', fp);
	//putc(' ', fp);  
}

void random_ints(int *a, int size) {
	int i;
	time_t t;	
	srand((unsigned) time(&t));
	for (i=0; i<size; i++) {
		a[i] = 1; //rand() % 64;
		if(i == size-1) {
			a[i] = 2;
		}
		//printf("i: %d\na[i]: %d\n", i, a[i]);
	}
}

void zeroarr(int *a, int size) {
	for (int i=0; i < size; i++) {
		a[i] = 0;
	}
}

void printarr(int *a, int size) {
	for(int i=0 ; i < size; i++) {
		printf("%d", a[i]);
	}
	printf("\n\n");
}
 
int main(void) {
	int *a, *b, *c; // host copies of a, b, c
	//d*;
	int *d_a, *d_b, *d_c; // device copies of a, b, c
	//FILE *fp;
	//fp = fopen("dump.txt", "a");
	//int *d_d;
	int size = N * sizeof(int);

	// Allocate space for device copies of a, b, c
	checkCudaErrors(cudaMalloc((void **) &d_a, size));
	checkCudaErrors(cudaMalloc((void **) &d_b, size));
	checkCudaErrors(cudaMalloc((void **) &d_c, size));

	//cudaMalloc((void**) &d_d, size*6);

	// Alloc space for host copies of a, b, c and setup input values
	a = (int *) malloc(size);
	random_ints(a, N);
	printf("A before sendng to device\n");
	printarr(a, N);

	b = (int *) malloc(size);
	printf("B before sending to device\n");
	random_ints(b, N);
	printarr(b, N);

	c = (int *) malloc(size);
	printf("C before sending to device\n");
	zeroarr(c, N);
	printarr(c, N);

	//d = (int *) malloc(size*6);
	//printf("D before sending to device\n");
	//zeroarr(d, N*6);
	//printarr(d, N*6);

	//printf("a: %d\nb: %d\nc: %d\n", *a, *b, *c);

	// Copy inputs to device
	//printf("d_a: %d, a %d", *d_a, *a);
	checkCudaErrors(cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice));

	// Launch add() kernel on GPU
	add<<<1,N>>>(d_a, d_b, d_c);

	// copy result back to host
	checkCudaErrors(cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost));
	printf("D_C from device\n");
	printarr(c, N);

	//cudaMemcpy(d, d_d, size*6, cudaMemcpyDeviceToHost);
	//printf("D_D from device\n");
	//printarr(d, size*6);

	printf("TOTAL: %d\n", *c);

	// Cleanup
	free(a);
	cudaFree(d_a);
	free(b);
	cudaFree(d_b);
	free(c);
	cudaFree(d_c);
	return 0;
}


#import <stdio.h>

__global__ void add(int *a, int *b, int *c, int n) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < n) {
		c[index] = a[index] + b[index];
	}
}

void random_ints(int *a, int N) {
	int i;
	for (i=0; i<N; ++i) {
		a[i] = rand();
	}
}

#define N (2048*2048)
#define M 512
int main(void) {
	int *a, *b, *c; // host copies of a, b, c
	int *d_a, *d_b, *d_c; // devic copies of a, b, c
	int size = N * sizeof(int);

	// Allocate space for device copies of a, b, c
	cudaMalloc((void **) &d_a, size);
	cudaMalloc((void **) &d_b, size);
	cudaMalloc((void **) &d_c, size);

	// Alloc space for host copies of a, b, c and setup input values
	a = (int *) malloc(size); random_ints(a, N);
	b = (int *) malloc(size); random_ints(b, N);
	c = (int *) malloc(size);

	// copy inputs to device
	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

	// Launch add() kernel on GPU
	add<<<(N+M-1)/M,M>>>(d_a, d_b, d_c, N);

	// copy result back to host
	cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

	// Cleanup
	free(a);
	cudaFree(d_a);
	free(b);
	cudaFree(d_b);
	free(c);
	cudaFree(d_c);
	return 0;
}

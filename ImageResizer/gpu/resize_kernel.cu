// Includes that fix syntax highlighting
#ifdef IMG_RESIZER_DEBUG
#include "device_launch_parameters.h"
#include "stdio.h"
#endif

extern "C" {

	__constant__ int arrSize;

	__global__ void adder(int *arrA, int *arrB, int *result) {
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		idx = min(idx, arrSize - 1);
		result[idx] = arrA[idx] + arrB[idx];
	}

}

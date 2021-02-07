#include <cstdio>
#include <exception>

// EASTL
#include <EASTL/vector.h>

// User
#include <cuda_manager.h> 

int main(int argc, char **argv) {
	CUDAManager cuda;
	cuda.initialize();

	cuda.testSystem();

	return 0;
}
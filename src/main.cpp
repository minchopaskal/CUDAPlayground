#include <cstdio>
#include <exception>

// User
#include <cuda_manager.h> 

int main(int argc, char **argv) {
	initializeCUDAManager("data\\kernel.ptx");

	CUDAManager &cuda = getCUDAManager();

	CUDAError err = cuda.testSystem();
	if (err.hasError()) {
		LOG_CUDA_ERROR(err, LogLevel::Error);
	}

	deinitializeCUDAManager();

	return 0;
}
#include <cuda_manager.h>

int main(int argc, char **argv) {
	initializeCUDAManager("data\\resize_kernel.ptx");

	CUDAManager &cudaman = getCUDAManager();
	CUDAError err = cudaman.testSystem();
	if (err.hasError()) {
		LOG_CUDA_ERROR(err, LogLevel::Debug);
	}

	deinitializeCUDAManager();

	return 0;
}
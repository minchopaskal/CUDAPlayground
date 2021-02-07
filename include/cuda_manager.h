#pragma once

#include <cassert>

#include <EASTL/vector.h>

#include <cuda_error_handling.h>
#include <timer.h>

struct CUDAContext;
struct CUDADevice {
	CUDADevice();
	~CUDADevice();

	CUDAError destroy();

	CUDAError initialize(int deviceOridnal);
	CUDAError use() const;

	CUdevice getDevice() const;
	
private:
	CUcontext ctx;
	CUdevice dev;
	char name[128];
	size_t totalMem;
};

struct CUDAFunction {
	CUDAFunction(CUmodule module, const char *name);

	void initialize(CUmodule module, const char *name);

	template <class T, class ...Types>
	CUDAError addParams(T param, Types ... paramList) {
		if (!successfulLoading) {
			CUDAError err(CUDA_ERROR_UNKNOWN, "HOST Error", "Adding parameters to non-loaded funtion!");
			LOG_CUDA_ERROR(err, LogLevel::Warning);
			return err;
		}

		memcpy(currParam, (void *)&param, sizeof(T));
		kernelParams.push_back(static_cast<void *>(currParam));
		currParam += sizeof(T);
		if (currParam > params + paramsSize) {
			CUDAError err(CUDA_ERROR_UNKNOWN, "HOST Error", "Too many parameters!");
			LOG_CUDA_ERROR(err, LogLevel::Error);
			return err;
		}

		return addParams(paramList...);
	}

	CUDAError addParams() {
		return CUDAError();
	}

	CUfunction getFunction() const { return func; }
	void **getParams() { return kernelParams.data(); }

private:
	static const int paramsSize = 1024;

	CUfunction func;
	eastl::vector<void *> kernelParams;
	char params[paramsSize];
	char *currParam;
	int successfulLoading;
};

struct CUDAManager {
	CUDAManager();
	~CUDAManager();

	CUDAError initialize();

	/// Uploads host data to device constant memory.
	/// Note that it works only for variables and not arrays!
	/// Has the same behaviour as calling uploadConstantArray with arrSize==1.
	/// @param param_h Pointer to host memory.
	/// @param name Name of the global constant variable in device memory.
	template <class T>
	CUDAError uploadConstantParam(const T *param_h, const char *name) {
		return uploadConstantArray(param_h, 1, name);
	}

	/// Uploads host array to device constant memory array.
	/// @param array_h Pointer to host array.
	/// @param name Name of the global constant variable in device memory.
	template <class T>
	CUDAError uploadConstantArray(const T *array_h, int arrSize, const char *name) {
		CUdeviceptr array_d;
		size_t bytes;
		RETURN_ON_ERROR(cuModuleGetGlobal(&array_d, &bytes, module, name));

		assert(sizeof(T) * arrSize == bytes);

		RETURN_ON_ERROR(cuMemcpyHtoD(array_d, array_h, bytes));

		return CUDAError();
	}

	CUDAError testSystem() {
		const int arrSize = 2 << 19; // ~ 1 million
		const int arrSizeInBytes = arrSize * sizeof(int);

		// prepare the host arrays we want to add
		int *arrA_h = new int[arrSize];
		int *arrB_h = new int[arrSize];

		for (int i = 0; i < arrSize; ++i) {
			arrA_h[i] = 2 * i;
			arrB_h[i] = 2 * i + 1;
		}

		// Prepare the device memory that will be used for the computation
		Timer stopWatch;
		stopWatch.start();
		CUdeviceptr arrA_d;
		CUdeviceptr arrB_d;
		CUdeviceptr result_d;
		RETURN_ON_ERROR(cuMemAlloc(&arrA_d, arrSizeInBytes));
		RETURN_ON_ERROR(cuMemAlloc(&arrB_d, arrSizeInBytes));
		RETURN_ON_ERROR(cuMemAlloc(&result_d, arrSizeInBytes));
		
		// Create two streams - one will be used for upload/download and the other for execution launches
		CUstream transferStream, computeStream;
		RETURN_ON_ERROR(cuStreamCreate(&transferStream, 0));
		RETURN_ON_ERROR(cuStreamCreate(&computeStream, 0));

		// Simply copy the data. No fancy copy/execution overlap
		RETURN_ON_ERROR(cuMemcpyHtoDAsync(arrA_d, arrA_h, arrSizeInBytes, transferStream));
		RETURN_ON_ERROR(cuMemcpyHtoDAsync(arrB_d, arrB_h, arrSizeInBytes, transferStream));

		// load the adder function
		CUDAFunction adder(module, "adder");
		RETURN_ON_ERROR_HANDLED(adder.addParams(arrA_d, arrB_d, result_d));
		
		// Experiment with constant memory
		RETURN_ON_ERROR_HANDLED(uploadConstantParam(&arrSize, "arrSize"));

		// We should wait for the input arrays to be transfered before launching the kernel
		RETURN_ON_ERROR(cuStreamSynchronize(transferStream));

		// TODO: research to what exptend are the optimal values for
		// grid/block dims dependent on the number of cores in an SM.
		const int blockDim = 192;
		const int gridDim = arrSize / blockDim + (arrSize % blockDim != 0);
		RETURN_ON_ERROR(cuLaunchKernel(
			adder.getFunction(),
			gridDim, 1, 1,
			blockDim, 1, 1,
			0,
			computeStream,
			adder.getParams(),
			nullptr
		));

		RETURN_ON_ERROR(cuStreamSynchronize(computeStream));
		const float kernelTime = stopWatch.time();

		int *result_h = new int[arrSize];
		RETURN_ON_ERROR(cuMemcpyDtoH(result_h, result_d, arrSizeInBytes));
		
		Logger::log(LogLevel::Debug, "Kernel execution time: %.2f\n", kernelTime);

		for (int i = 0; i < arrSize; ++i) {
			assert(result_h[i] == 4 * i + 1);
		}

		stopWatch.restart();
		for (int i = 0; i < arrSize; ++i) {
			result_h[i] = arrA_h[i] + arrB_h[i];
		}
		const float cpuTime = stopWatch.time();
		Logger::log(LogLevel::Debug, "CPU execution time: %.2f\n", cpuTime);

		return CUDAError();
	}

private:
	CUDAError destroy();
	CUDAError initializeDevices();
	CUDAError loadModule();

private:
	eastl::vector<CUDADevice> devices;
	CUmodule module;
	int cudaVersion;
};
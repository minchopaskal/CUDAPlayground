#pragma once

#include <cassert>
#include <vector>

#include <cuda_memory.h>
#include <timer.h>

enum class CUDADefaultStreamsEnumeration : int {
	Execution = 0,
	Upload,
	Download,

	Count
};

struct CUDADevice {
	CUDADevice();
	~CUDADevice();

	CUDAError deinitialize();

	CUDAError initialize(int deviceOridnal, const std::vector<std::string> &ptxFiles, bool useDynamicParallelism);

	CUDAError use() const;

	CUdevice getDevice() const;
	CUmodule getModule() const;
	CUstream getDefaultStream(CUDADefaultStreamsEnumeration defStreamEnum) const;

	CUDAError getTotalMemory(SizeType &result) const;
	CUDAError getName(std::string &result) const;
	CUDAError getFreeMemory(SizeType &result) const;

	/// Uploads host data to device constant memory
	/// Has the same behaviour as calling uploadConstantArray with arrSize==1.
	/// @param param_h Pointer to host memory.
	/// @param name Name of the global constant variable in device memory.
	/// @param index The index at which to copy the host memory if the device constant is an array.
	template <class T>
	CUDAError uploadConstantParam(const T *param_h, const char *name, const SizeType index = SizeType(0)) const {
		CUdeviceptr array_d;
		size_t bytes;
		RETURN_ON_CUDA_ERROR(cuModuleGetGlobal(&array_d, &bytes, module, name));

		massert(sizeof(T) * (index + 1) <= bytes);

		RETURN_ON_CUDA_ERROR(cuMemcpyHtoD(array_d + index * sizeof(T), param_h, sizeof(T)));

		return CUDAError();
	}

	/// Uploads host array to device constant memory array.
	/// @param array_h Pointer to host array.
	/// @param name Name of the global constant variable in device memory.
	template <class T>
	CUDAError uploadConstantArray(const T *array_h, int arrSize, const char *name) const {
		CUdeviceptr array_d;
		size_t bytes;
		RETURN_ON_CUDA_ERROR(cuModuleGetGlobal(&array_d, &bytes, module, name));

		massert(sizeof(T) * arrSize == bytes);

		RETURN_ON_CUDA_ERROR(cuMemcpyHtoD(array_d, array_h, bytes));

		return CUDAError();
	}

private:
	CUDAError loadModule(const std::vector<std::string> &ptxFiles, bool useDynamicParallelism);
	
private:
	std::vector<CUstream> streams;
	CUcontext ctx;
	CUlinkState linkState;
	CUmodule module;
	CUdevice dev;
	char name[128];
	SizeType totalMem;
};

struct CUDAFunction {
	CUDAFunction();
	CUDAFunction(CUmodule module, const char *name);

	void initialize(CUmodule module, const char *name);

	/// Launch the currnet CUDA kernel with the specified thread count
	/// @param threadCount Number of CUDA threads we want to execute
	/// @param stream CUDA stream on which to launch the kernel
	/// @return CUDAError() on success
	CUDAError launch(unsigned int threadCount, CUstream stream);

	/// Launches the kernel and then synchronizes with the stream
	CUDAError launchSync(unsigned int threadCount, CUstream stream);

	template <class T, class ...Types>
	CUDAError addParams(T param, Types ... paramList) {
		if (!successfulLoading) {
			CUDAError err(CUDA_ERROR_UNKNOWN, "HOST Error", "Adding parameters to non-loaded funtion!");
			LOG_CUDA_ERROR(err, LogLevel::Warning);
			return err;
		}

		memcpy(currParam, (void*)&param, sizeof(T));
		kernelParams.push_back(static_cast<void*>(currParam));
		currParam += sizeof(T);
		if (currParam > params + paramsSize) {
			CUDAError err(CUDA_ERROR_UNKNOWN, "HOST Error", "Too many parameters!");
			LOG_CUDA_ERROR(err, LogLevel::Error);
			return err;
		}

		return addParams(paramList...);
	}

	/// Helper function for the variadic template function addParams.
	CUDAError addParams() {
		return CUDAError();
	}

	CUfunction getFunction() const { return func; }
	void** getParams() { return kernelParams.data(); }
	
	SizeType getNumParams() const { return kernelParams.size(); }
	
	template <class T>
	bool changeParam(T *newParam, int paramIndex) {
		if (!successfulLoading) {
			CUDAError err(CUDA_ERROR_UNKNOWN, "HOST Error", "Changing parameters of a non-loaded funtion!");
			LOG_CUDA_ERROR(err, LogLevel::Warning);
			return err;
		}

		if (paramIndex < 0 || paramIndex >= kernelParams.size()) {
			CUDAError err(CUDA_ERROR_UNKNOWN, "HOST Error", "Changing not yet set parameters!");
			LOG_CUDA_ERROR(err, LogLevel::Warning);
			return err;
		}

		void *oldParamPtr = kernelParams[paramIndex];
		memcpy(oldParamPtr, newParam, sizeof(T));

		return true;
	}

	void clearParams();

private:
	static const int paramsSize = 1024;

	CUfunction func;
	std::vector<void *> kernelParams;
	char params[paramsSize];
	char *currParam;
	int successfulLoading;

#ifdef TIME_KERNEL_EXECUTION
	std::string kernelName;
#endif
};

struct CUDAManager {
	template <class T>
	T& getAllocator();

	const std::vector<CUDADevice>& getDevices() const;

	/// Test everything is working correctly.
	/// Requires a kernel with the following definition
	/// __global__ void adder(int*, int*, int*)
	/// in the module and also the following constant:
	/// __constant__ int arrSize;
	CUDAError testSystem();

private:
	friend bool initializeCUDAManager(const std::vector<std::string> &ptxFiles, bool useDynamicParallelism);
	friend void deinitializeCUDAManager();
	
	CUDAManager(const std::vector<std::string> &ptxFiles, bool useDynamicParallelism);
	~CUDAManager();
	CUDAError initialize(const std::vector<std::string> &ptxFiles, bool useDynamicParallelism);
	CUDAError deinitialize();

	CUDAError initializeDevices(const std::vector<std::string> &ptxFiles, bool useDynamicParallelism);
	CUDAError initializeAllocators();

private:
	std::vector<CUDADevice> devices;
	CUDADefaultAllocator defaultAllocator;
	CUDAVirtualAllocator virtualAllocator;
	int cudaVersion;
	bool initialized;
};

bool initializeCUDAManager(const std::vector<std::string> &ptxFiles, bool useDynamicParallelism);
void deinitializeCUDAManager();
CUDAManager &getCUDAManager();

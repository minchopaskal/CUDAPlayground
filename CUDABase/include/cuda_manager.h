#pragma once

#include <cassert>
#include <vector>

#include <cuda_memory.h>
#include <timer.h>

struct CUDADevice {
	CUDADevice();
	~CUDADevice();

	CUDAError destroy();

	CUDAError initialize(int deviceOridnal, const char *modulePath);
	CUDAError use() const;

	CUdevice getDevice() const;
	CUmodule getModule() const;

	CUDAError getTotalMemory(SizeType &result) const;
	CUDAError getFreeMemory(SizeType &result) const;

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

		massert(sizeof(T) * arrSize == bytes);

		RETURN_ON_ERROR(cuMemcpyHtoD(array_d, array_h, bytes));

		return CUDAError();
	}

private:
	CUDAError loadModule(const char *modulePath);
	
private:
	CUcontext ctx;
	CUmodule module;
	CUdevice dev;
	char name[128];
	SizeType totalMem;
};

struct CUDAFunction {
	CUDAFunction(CUmodule module, const char *name);

	void initialize(CUmodule module, const char *name);

	template <class T, class ...Types>
	CUDAError addParams(T param, Types ... paramList);

	/// Helper function for the variadic template function addParams.
	CUDAError addParams() {
		return CUDAError();
	}

	CUfunction getFunction() const { return func; }
	void **getParams() { return kernelParams.data(); }

private:
	static const int paramsSize = 1024;

	CUfunction func;
	std::vector<void *> kernelParams;
	char params[paramsSize];
	char *currParam;
	int successfulLoading;
};

struct CUDAManager {
	template <class T>
	T& getAllocator();

	const std::vector<CUDADevice> &getDevices() const;

	CUDAError testSystem();

private:
	friend void initializeCUDAManager(const char*);
	friend void deinitializeCUDAManager();
	
	CUDAManager(const char *modulePath);
	~CUDAManager();
	CUDAError initialize(const char *modulePath);
	CUDAError destroy();

	CUDAError initializeDevices(const char *modulePath);
	CUDAError initializeAllocators();

private:
	std::vector<CUDADevice> devices;
	CUDADefaultAllocator defaultAllocator;
	CUDAVirtualAllocator virtualAllocator;
	int cudaVersion;
};

void initializeCUDAManager(const char *modulePath);
void deinitializeCUDAManager();
CUDAManager &getCUDAManager();

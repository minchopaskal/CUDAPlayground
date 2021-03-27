#include <cuda_manager.h>
#include <sstream>

#include <cuda_buffer.h>

#define GB_IN_BYTES 1e9f

/*
===============================================================
CUDADevice
===============================================================
*/
CUDADevice::CUDADevice() : ctx(NULL), dev(CU_DEVICE_INVALID), module(NULL), name("unknown device"), totalMem(0) { }

CUDADevice::~CUDADevice() {
	destroy();
}

CUDAError CUDADevice::destroy() {
	if (module != NULL) {
		RETURN_ON_ERROR(cuModuleUnload(module));
		module = NULL;
	}

	if (ctx != NULL) {
		RETURN_ON_ERROR(cuCtxDestroy(ctx));
		ctx = NULL;
	}

	if (dev != CU_DEVICE_INVALID) {
		dev = CU_DEVICE_INVALID;
	}

	memset(name, 0x0, sizeof(name));
	totalMem = 0;

	return CUDAError();
}


CUDAError CUDADevice::initialize(int deviceOridnal, const char *modulePath) {
	RETURN_ON_ERROR_HANDLED(destroy());

	RETURN_ON_ERROR(cuDeviceGet(&dev, deviceOridnal));
	RETURN_ON_ERROR(cuDeviceGetName(name, 128, dev));
	RETURN_ON_ERROR(cuDeviceTotalMem(&totalMem, dev));
	int supportUVA = false;
	RETURN_ON_ERROR(cuDeviceGetAttribute(&supportUVA, CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING, dev));

	if (!supportUVA) {
		char errorMsg[256];
		sprintf_s(errorMsg, "Device %s does not support unified virtual addresing! Exiting...", name);
		CUDAError err(CUDA_ERROR_INVALID_DEVICE, errorMsg, "");
		LOG_CUDA_ERROR(err, LogLevel::Debug);
		return err;
	}

	// Create a context for the device.
	// We create a contex for each device and associate it with it.
	// Since CUDA 4.0, multiple threads can have the same context as current,
	// so we don't need more contexts than that.
	RETURN_ON_ERROR(
		cuCtxCreate(&ctx, CU_CTX_SCHED_BLOCKING_SYNC | CU_CTX_MAP_HOST, dev)
	);

	Logger::log(LogLevel::Info, 
		"Device %s initialized! Total mem: %.2fGB", 
		name,
		totalMem / GB_IN_BYTES
	);

	// cuCtxCreate pushes the context onto the stack, so safe to load the module for this context
	loadModule(modulePath);

	return CUDAError();
}

CUdevice CUDADevice::getDevice() const {
	return dev;
}

CUmodule CUDADevice::getModule() const {
	return module;
}

CUDAError CUDADevice::getTotalMemory(SizeType &result) const {
	if (dev == CU_DEVICE_INVALID) {
		return CUDAError(CUDA_ERROR_NOT_INITIALIZED, "CUDADevice_ERROR_NOT_INITIALIZED", "");
	}

	result = totalMem;
	return CUDAError();
}

CUDAError CUDADevice::getFreeMemory(SizeType &result) const {
	if (dev == CU_DEVICE_INVALID) {
		return CUDAError(CUDA_ERROR_NOT_INITIALIZED, "CUDADevice_ERROR_NOT_INITIALIZED", "");
	}

	use();
	RETURN_ON_ERROR(cuMemGetInfo(&result, NULL));

	return CUDAError();
}

CUDAError CUDADevice::use() const {
	if (dev == CU_DEVICE_INVALID) {
		return CUDAError(CUDA_ERROR_NOT_INITIALIZED, "CUDADevice_ERROR_NOT_INITIALIZED", "");
	}

	RETURN_ON_ERROR(cuCtxSetCurrent(ctx));

	return CUDAError();
}

CUDAError CUDADevice::loadModule(const char *modulePath) {
	RETURN_ON_ERROR(cuModuleLoad(&module, modulePath));

	return CUDAError();
}

/*
===============================================================
CUDAFunction
===============================================================
*/
CUDAFunction::CUDAFunction(CUmodule module, const char *name) 
	: func(NULL), params(""), currParam(params), successfulLoading(false) {
	initialize(module, name);
}

void CUDAFunction::initialize(CUmodule module, const char *name) {
	if (func != NULL) {
		func = NULL;
	}

	CUDAError err = handleCUDAError(cuModuleGetFunction(&func, module, name));
	if (err.hasError()) {
		Logger::log(LogLevel::Error, "Failed to load function: %s!", name);
	}
	successfulLoading = !err.hasError();
}

template <class T, class ...Types>
CUDAError CUDAFunction::addParams(T param, Types ... paramList) {
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

/* 
===============================================================
CUDAManager
===============================================================
*/
CUDAManager::CUDAManager(const char *modulePath) : cudaVersion(0) {
	initialize(modulePath);
}

CUDAManager::~CUDAManager() {
	destroy();
}

CUDAError CUDAManager::initialize(const char *modulePath) {
	RETURN_ON_ERROR_HANDLED(destroy());

	RETURN_ON_ERROR(cuInit(0));
	RETURN_ON_ERROR(cuDriverGetVersion(&cudaVersion));

	Logger::log(LogLevel::Info, "CUDA version: %d.%d", cudaVersion / 1000, (cudaVersion % 100) / 10);

	RETURN_ON_ERROR_HANDLED(initializeDevices(modulePath));

	RETURN_ON_ERROR_HANDLED(initializeAllocators());

	return CUDAError();
}

const std::vector<CUDADevice> &CUDAManager::getDevices() const {
	return devices;
}

CUDAError CUDAManager::destroy() {
	for (int i = 0; i < devices.size(); ++i) {
		devices[i].destroy();
	}

	return CUDAError();
}

CUDAError CUDAManager::initializeDevices(const char *modulePath) {
	int deviceCount = 0;
	RETURN_ON_ERROR(cuDeviceGetCount(&deviceCount));

	if (deviceCount == 0) {
		Logger::log(LogLevel::Warning, "No CUDA devices found!");
		return CUDAError();
	}

	devices.resize(deviceCount);
	for (int i = 0; i < devices.size(); ++i) {
		RETURN_ON_ERROR_HANDLED(devices[i].initialize(i, modulePath));
	}

	return CUDAError();
}

CUDAError CUDAManager::initializeAllocators() {
	RETURN_ON_ERROR_HANDLED(defaultAllocator.initialize());
	RETURN_ON_ERROR_HANDLED(virtualAllocator.initialize());
	return CUDAError();
}

CUDAError CUDAManager::testSystem() {
	const int deviceToUseIdx = 0;
	auto &dev = devices[deviceToUseIdx];
	dev.use();

	int arrSize = 1 << 20; // ~ 1 million
	arrSize = arrSize + (arrSize % 100 ? (100 - arrSize % 100) : 0);

	Logger::log(
		LogLevel::Info,
		"Starting following test:\n"
		"\tTwo int arrays each with %ld elements will be added\n"
		"\telement by element into a third array both on GPU and CPU.\n"
		"\tTimes of both executions will be measured.\n",
		arrSize
	);

	const size_t arrSizeInBytes = arrSize * sizeof(int);

	// prepare the host arrays we want to add
	int *result_h = new int[arrSize];

	Timer gpuTimer;
	CUDADefaultPinnedBuffer arrA_d;
	CUDADefaultPinnedBuffer arrB_d;
	CUDADefaultBuffer result_d;
	RETURN_ON_ERROR_HANDLED(arrA_d.initialize(arrSizeInBytes));
	RETURN_ON_ERROR_HANDLED(arrB_d.initialize(arrSizeInBytes));
	RETURN_ON_ERROR_HANDLED(result_d.initialize(arrSizeInBytes));

	int *arrA_h = reinterpret_cast<int*>(arrA_d.hostHandle());
	int *arrB_h = reinterpret_cast<int*>(arrB_d.hostHandle());
	
	for (int i = 0; i < arrSize; ++i) {
		arrA_h[i] = 2 * i;
		arrB_h[i] = 2 * i + 1;
	}

	arrA_d.upload();
	arrB_d.upload();

	CUstream stream;
	RETURN_ON_ERROR(cuStreamCreate(&stream, 0));

	// load the adder function
	CUDAFunction adder(dev.getModule(), "adder");
	RETURN_ON_ERROR_HANDLED(adder.addParams(arrA_d.handle(), arrB_d.handle(), result_d.handle()));

	RETURN_ON_ERROR_HANDLED(dev.uploadConstantParam(&arrSize, "arrSize"));

	Timer kernelTimer;
	const int blockDim = 192;
	const int gridDim = arrSize / blockDim + (arrSize % blockDim != 0);
	RETURN_ON_ERROR(cuLaunchKernel(
		adder.getFunction(),
		gridDim, 1, 1,
		blockDim, 1, 1,
		0,
		stream,
		adder.getParams(),
		nullptr
	));

	// We only need to wait on the last stream as it's the last computation sent to the device
	RETURN_ON_ERROR(cuStreamSynchronize(stream));
	const float kernelTime = kernelTimer.time();

	RETURN_ON_ERROR_HANDLED(result_d.download(result_h));
	const float gpuTime = gpuTimer.time();

	Logger::log(LogLevel::InfoFancy, "GPUTime: %.2fms with kernel execution time: %.2fms\n", gpuTime, kernelTime);

	for (int i = 0; i < arrSize; ++i) {
		massert(result_h[i] == 4 * i + 1);
	}

	Timer cpuTimer;
	for (int i = 0; i < arrSize; ++i) {
		result_h[i] = arrA_h[i] + arrB_h[i];
	}
	const float cpuTime = cpuTimer.time();
	Logger::log(LogLevel::InfoFancy, "CPU execution time: %.2fms", cpuTime);

	return CUDAError();
}

template <>
CUDADefaultAllocator &CUDAManager::getAllocator<CUDADefaultAllocator>() { return defaultAllocator; }

template <>
CUDAVirtualAllocator &CUDAManager::getAllocator<CUDAVirtualAllocator>() { return virtualAllocator; }

static CUDAManager *_cudamanagerSingleton = nullptr;

void initializeCUDAManager(const char *modulePath) {
	if (_cudamanagerSingleton == nullptr) {
		_cudamanagerSingleton = new CUDAManager(modulePath);
	}
}

void deinitializeCUDAManager() {
	if (_cudamanagerSingleton == nullptr) {
		return;
	}

	delete _cudamanagerSingleton;
}

CUDAManager &getCUDAManager() {
	return *_cudamanagerSingleton;
}

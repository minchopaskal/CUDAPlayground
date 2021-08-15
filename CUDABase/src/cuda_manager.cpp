#include <cuda_manager.h>
#include <sstream>

#include <cuda_buffer.h>

#define GB_IN_BYTES 1e9f

/*
===============================================================
CUDADevice
===============================================================
*/
CUDADevice::CUDADevice() : ctx(NULL), linkState(NULL), dev(CU_DEVICE_INVALID), module(NULL), name("unknown device"), totalMem(0) { }

CUDADevice::~CUDADevice() {
	deinitialize();
}

CUDAError CUDADevice::deinitialize() {
	for (int i = 0; i < streams.size(); ++i) {
		RETURN_ON_CUDA_ERROR(cuStreamDestroy(streams[i]));
	}
	streams.clear();

	if (module != NULL) {
		RETURN_ON_CUDA_ERROR(cuModuleUnload(module));
		module = NULL;
	}

	if (linkState != NULL) {
		RETURN_ON_CUDA_ERROR(cuLinkDestroy(linkState));
	}

	if (ctx != NULL) {
		RETURN_ON_CUDA_ERROR(cuCtxDestroy(ctx));
		ctx = NULL;
	}

	if (dev != CU_DEVICE_INVALID) {
		dev = CU_DEVICE_INVALID;
	}

	memset(name, 0x0, sizeof(name));
	totalMem = 0;

	return CUDAError();
}


CUDAError CUDADevice::initialize(int deviceOridnal, const std::vector<std::string> &ptxFiles, bool useDynamicParallelism) {
	RETURN_ON_CUDA_ERROR_HANDLED(deinitialize());

	RETURN_ON_CUDA_ERROR(cuDeviceGet(&dev, deviceOridnal));
	RETURN_ON_CUDA_ERROR(cuDeviceGetName(name, 128, dev));
	RETURN_ON_CUDA_ERROR(cuDeviceTotalMem(&totalMem, dev));
	int supportUVA = false;
	RETURN_ON_CUDA_ERROR(cuDeviceGetAttribute(&supportUVA, CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING, dev));

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
	RETURN_ON_CUDA_ERROR(
		cuCtxCreate(&ctx, CU_CTX_SCHED_BLOCKING_SYNC | CU_CTX_MAP_HOST, dev)
	);

	Logger::log(LogLevel::Info, 
		"Device %s initialized! Total mem: %.2fGB", 
		name,
		totalMem / GB_IN_BYTES
	);

	// cuCtxCreate pushes the context onto the stack, so safe to load the module for this context
	loadModule(ptxFiles, useDynamicParallelism);

	const int numDefaultStreams = static_cast<int>(CUDADefaultStreamsEnumeration::Count);
	CUstream defaultStreams[numDefaultStreams];
	for (int i = 0; i < numDefaultStreams; ++i) {
		RETURN_ON_CUDA_ERROR(cuStreamCreate(&defaultStreams[i], 0));
		streams.push_back(defaultStreams[i]);
	}

	return CUDAError();
}

CUdevice CUDADevice::getDevice() const {
	return dev;
}

CUmodule CUDADevice::getModule() const {
	return module;
}

CUstream CUDADevice::getDefaultStream(CUDADefaultStreamsEnumeration defStreamEnum) const {
	if (dev == CU_DEVICE_INVALID) {
		return NULL;
	}

	massert(streams.size() >= 3);
	const int streamIdx = static_cast<int>(defStreamEnum);
	if (streamIdx < 0 || streamIdx >= 3) {
		return NULL;
	}

	return streams[streamIdx];
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
	RETURN_ON_CUDA_ERROR(cuMemGetInfo(&result, NULL));

	return CUDAError();
}

CUDAError CUDADevice::use() const {
	if (dev == CU_DEVICE_INVALID) {
		return CUDAError(CUDA_ERROR_NOT_INITIALIZED, "CUDADevice_ERROR_NOT_INITIALIZED", "");
	}

	RETURN_ON_CUDA_ERROR(cuCtxSetCurrent(ctx));

	return CUDAError();
}

CUDAError CUDADevice::loadModule(const std::vector<std::string> &ptxFiles, bool useDynamicParallelism) {
#ifdef CUDA_DEBUG
	int generateDebugInfo = 1;
#else // !CUDA_DEBUG
	int generateDebugInfo = 0;
#endif // CUDA_DEBUG

	static constexpr int NUM_LINK_OPTIONS = 1;
	CUjit_option options[NUM_LINK_OPTIONS] = { CU_JIT_GENERATE_DEBUG_INFO };
	void *optionValues[] = { (void*)&generateDebugInfo };

	CUlinkState linkState;
	RETURN_ON_CUDA_ERROR(cuLinkCreate(NUM_LINK_OPTIONS, options, optionValues, &linkState));
	CUjitInputType moduleType = CU_JIT_INPUT_PTX;
	for (int i = 0; i < ptxFiles.size(); ++i) {
		RETURN_ON_CUDA_ERROR(cuLinkAddFile(linkState, moduleType, ptxFiles[i].c_str(), 0, nullptr, nullptr));
	}

	if (useDynamicParallelism) {
		CUjitInputType libType = CU_JIT_INPUT_LIBRARY;
		RETURN_ON_CUDA_ERROR(cuLinkAddFile(
			linkState,
			libType,
			CUDA_LIB_PATH "/cudadevrt.lib",
			0,
			nullptr,
			nullptr
		));
	}

	void *outCubin = nullptr;
	size_t outSize = 0;
	RETURN_ON_CUDA_ERROR(cuLinkComplete(linkState, &outCubin, &outSize));

	RETURN_ON_CUDA_ERROR(cuModuleLoadData(&module, outCubin));

	RETURN_ON_CUDA_ERROR(cuLinkDestroy(linkState));

	return CUDAError();
}

/*
===============================================================
CUDAFunction
===============================================================
*/
CUDAFunction::CUDAFunction() : func(NULL), params(""), currParam(params), successfulLoading(false) { }

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
		LOG_CUDA_ERROR(err, LogLevel::Error);
		Logger::log(LogLevel::Error, "Failed to load function %s", name);
	}
	successfulLoading = !err.hasError();

#ifdef TIME_KERNEL_EXECUTION
	kernelName = name;
#endif
}

CUDAError CUDAFunction::launch(unsigned int threadCount, CUstream stream) {
#ifdef TIME_KERNEL_EXECUTION
	Timer kernelTimer;
#endif

	const int blockDim = 128;
	const int gridDim = threadCount / blockDim + (threadCount % blockDim != 0);
	RETURN_ON_CUDA_ERROR(cuLaunchKernel(
		getFunction(),
		gridDim, 1, 1,
		blockDim, 1, 1,
		0,
		stream,
		getParams(),
		nullptr
	));

#ifdef TIME_KERNEL_EXECUTION
	RETURN_ON_CUDA_ERROR(cuStreamSynchronize(stream));
	float kernelTimeMS = kernelTimer.time();
	Logger::log(LogLevel::InfoFancy, "Execution of CUDA kernel \"%s\" took %.2fms", kernelName.c_str(), kernelTimeMS);
#endif

	return CUDAError();
}

CUDAError CUDAFunction::launchSync(unsigned int threadCount, CUstream stream) {
	RETURN_ON_CUDA_ERROR_HANDLED(launch(threadCount, stream));

	RETURN_ON_CUDA_ERROR(cuStreamSynchronize(stream));
	
	return CUDAError();
}

void CUDAFunction::clearParams() {
	currParam = params;
	kernelParams.clear();
}

/* 
===============================================================
CUDAManager
===============================================================
*/
CUDAManager::CUDAManager(const std::vector<std::string> &ptxFiles, bool useDynamicParallelism) : cudaVersion(0) {
	initialize(ptxFiles, useDynamicParallelism);
}

CUDAManager::~CUDAManager() {
	deinitialize();
}

CUDAError CUDAManager::initialize(const std::vector<std::string> &ptxFiles, bool useDynamicParallelism) {
	RETURN_ON_CUDA_ERROR_HANDLED(deinitialize());

	RETURN_ON_CUDA_ERROR(cuInit(0));
	RETURN_ON_CUDA_ERROR(cuDriverGetVersion(&cudaVersion));

	Logger::log(LogLevel::Info, "CUDA version: %d.%d", cudaVersion / 1000, (cudaVersion % 100) / 10);

	RETURN_ON_CUDA_ERROR_HANDLED(initializeDevices(ptxFiles, useDynamicParallelism));

	RETURN_ON_CUDA_ERROR_HANDLED(initializeAllocators());

	return CUDAError();
}

const std::vector<CUDADevice> &CUDAManager::getDevices() const {
	return devices;
}

CUDAError CUDAManager::deinitialize() {
	defaultAllocator.deinitialize();
	virtualAllocator.deinitialize();

	// Destroy devices last as they hold the contexts
	for (int i = 0; i < devices.size(); ++i) {
		devices[i].deinitialize();
	}

	return CUDAError();
}

CUDAError CUDAManager::initializeDevices(const std::vector<std::string> &ptxFiles, bool useDynamicParallelism) {
	int deviceCount = 0;
	RETURN_ON_CUDA_ERROR(cuDeviceGetCount(&deviceCount));

	if (deviceCount == 0) {
		Logger::log(LogLevel::Warning, "No CUDA devices found!");
		return CUDAError();
	}

	devices.resize(deviceCount);
	for (int i = 0; i < devices.size(); ++i) {
		RETURN_ON_CUDA_ERROR_HANDLED(devices[i].initialize(i, ptxFiles, useDynamicParallelism));
	}

	return CUDAError();
}

CUDAError CUDAManager::initializeAllocators() {
	RETURN_ON_CUDA_ERROR_HANDLED(defaultAllocator.initialize());
	RETURN_ON_CUDA_ERROR_HANDLED(virtualAllocator.initialize());
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
	RETURN_ON_CUDA_ERROR_HANDLED(arrA_d.initialize(arrSizeInBytes));
	RETURN_ON_CUDA_ERROR_HANDLED(arrB_d.initialize(arrSizeInBytes));
	RETURN_ON_CUDA_ERROR_HANDLED(result_d.initialize(arrSizeInBytes));

	int *arrA_h = reinterpret_cast<int*>(arrA_d.hostHandle());
	int *arrB_h = reinterpret_cast<int*>(arrB_d.hostHandle());
	
	for (int i = 0; i < arrSize; ++i) {
		arrA_h[i] = 2 * i;
		arrB_h[i] = 2 * i + 1;
	}

	arrA_d.upload();
	arrB_d.upload();

	RETURN_ON_CUDA_ERROR_HANDLED(dev.uploadConstantParam(&arrSize, "arrSize"));

	// load the adder function
	CUDAFunction adder(dev.getModule(), "adder");
	RETURN_ON_CUDA_ERROR_HANDLED(adder.addParams(arrA_d.handle(), arrB_d.handle(), result_d.handle()));

	CUstream stream;
	RETURN_ON_CUDA_ERROR(cuStreamCreate(&stream, 0));

	Timer kernelTimer;
	adder.launch(arrSize, stream);

	// We only need to wait on the last stream as it's the last computation sent to the device
	RETURN_ON_CUDA_ERROR(cuStreamSynchronize(stream));
	const float kernelTime = kernelTimer.time();

	RETURN_ON_CUDA_ERROR_HANDLED(result_d.download(result_h));
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

void initializeCUDAManager(const std::vector<std::string> &ptxFiles, bool useDynamicParallelism) {
	if (_cudamanagerSingleton == nullptr) {
		_cudamanagerSingleton = new CUDAManager(ptxFiles, useDynamicParallelism);
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

#include <cuda_manager.h>
#include <sstream>

#define GB_IN_BYTES 1e9f

/*
===============================================================
CUDADevice
===============================================================
*/
CUDADevice::CUDADevice() : ctx(NULL), dev(NULL), name("unknown device"), totalMem(0) { }

CUDADevice::~CUDADevice() {
	destroy();
}

CUDAError CUDADevice::destroy() {
	if (ctx != NULL) {
		RETURN_ON_ERROR(cuCtxDestroy(ctx));
		ctx = NULL;
	}
	if (dev != NULL) {
		dev = NULL;
	}
	memset(name, 0x0, sizeof(name));
	totalMem = 0;

	return CUDAError();
}


CUDAError CUDADevice::initialize(int deviceOridnal) {
	RETURN_ON_ERROR_HANDLED(destroy());

	RETURN_ON_ERROR(cuDeviceGet(&dev, deviceOridnal));
	RETURN_ON_ERROR(cuDeviceGetName(name, 128, dev));
	RETURN_ON_ERROR(cuDeviceTotalMem(&totalMem, dev));

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

	return CUDAError();
}

CUdevice CUDADevice::getDevice() const {
	return dev;
}

CUDAError CUDADevice::use() const {
	RETURN_ON_ERROR(cuCtxSetCurrent(ctx));

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

/* 
===============================================================
CUDAManager
===============================================================
*/
CUDAManager::CUDAManager() : module(NULL), cudaVersion(0) { }

CUDAManager::~CUDAManager() {
	destroy();
}

CUDAError CUDAManager::initialize() {
	RETURN_ON_ERROR_HANDLED(destroy());

	RETURN_ON_ERROR(cuInit(0));
	RETURN_ON_ERROR(cuDriverGetVersion(&cudaVersion));

	Logger::log(LogLevel::Info, "CUDA version: %d.%d", cudaVersion / 1000, (cudaVersion % 100) / 10);

	RETURN_ON_ERROR_HANDLED(initializeDevices());

	RETURN_ON_ERROR_HANDLED(loadModule());

	return CUDAError();
}

CUDAError CUDAManager::destroy() {
	if (module != NULL) {
		RETURN_ON_ERROR(cuModuleUnload(module));
	}

	for (int i = 0; i < devices.size(); ++i) {
		devices[i].destroy();
	}

	return CUDAError();
}

CUDAError CUDAManager::initializeDevices() {
	int deviceCount = 0;
	RETURN_ON_ERROR(cuDeviceGetCount(&deviceCount));

	if (deviceCount == 0) {
		Logger::log(LogLevel::Warning, "No CUDA devices found!");
		return CUDAError();
	}

	devices.resize(deviceCount);
	for (int i = 0; i < devices.size(); ++i) {
		RETURN_ON_ERROR_HANDLED(devices[i].initialize(i));
	}

	return CUDAError();
}

CUDAError CUDAManager::loadModule() {
	RETURN_ON_ERROR(cuModuleLoad(&module, "data\\kernel.ptx"));

	return CUDAError();
}

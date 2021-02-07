#pragma once

// User
#include <logger.h>

// CUDA
#include <cuda.h>

#define LOG_CUDA_ERROR(err, logLevel) \
do { \
	Logger::log((logLevel), \
	"CUDA Error(%d) at %s in %s:%d:\n" \
	"\tError name: %s\n" \
	"\tError description : %s\n", \
	(err).getError(), \
	__FUNCTION__, __FILE__, __LINE__, \
	(err).getName(), \
	(err).getDesc()); \
} while (false)


#define RETURN_ON_ERROR(x) \
do { \
	CUDAError err_ = handleCUDAError((x)); \
	if (err_.hasError()) { \
		LOG_CUDA_ERROR(err_, LogLevel::Error); \
		return err_; \
	} \
} while (false)

#define RETURN_ON_ERROR_HANDLED(x) \
do { \
	CUDAError err_ = (x); \
	if (err_.hasError()) { \
		return err_; \
	} \
} while (false)

struct CUDAError {
	CUDAError() : error(CUDA_SUCCESS), name("CUDA_SUCCESS"), desc("") { }
	CUDAError(CUresult error, const char *name, const char *desc) : error(error), name(name), desc(desc) { }

	bool hasError() const { return error != CUDA_SUCCESS; }
	CUresult getError() const { return error; }
	const char *getName() const { return name; }
	const char *getDesc() const { return desc; }

private:
	CUresult error;
	const char *name;
	const char *desc;
};

static CUDAError handleCUDAError(CUresult err) {
	if (err != CUDA_SUCCESS) {
		const char *cudaErrorName = NULL;
		const char *cudaErrorDescription = "UNKNOWN CUDA ERROR DESCRIPTION";
		if (cuGetErrorName(err, &cudaErrorName) == CUDA_ERROR_INVALID_VALUE) {
			cudaErrorName = "UNKNOWN CUDA ERROR";
			cudaErrorDescription = "UNKNOWN CUDA ERROR DESCRIPTION";
		} else {
			cuGetErrorString(err, &cudaErrorDescription);
		}

#ifdef EXIT_ON_ERROR
		exit(err);
#else
		return CUDAError(err, cudaErrorName, cudaErrorDescription);
#endif // EXIT_ON_ERROR
	}

	return CUDAError();
}
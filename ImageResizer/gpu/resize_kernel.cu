// Includes that fix syntax highlighting
#ifdef IMG_RESIZER_DEBUG
#include "device_launch_parameters.h"
#include "stdio.h"
#include "math_functions.h"
#endif

#include "math_constants.h"

#define gvoid  __global__ void
#define gfloat __global__ float
#define gint   __global__ int

#define dvoid  __device__ void
#define dfloat __device__ float
#define dint   __device__ int

#define cvoid  __constant__ void
#define cfloat __constant__ float
#define cint   __constant__ int

typedef float (*samplingKernel)(float x, float y, int window);

extern "C" {

	cint arrSize;
	gvoid adder(int *arrA, int *arrB, int *result) {
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		idx = min(idx, arrSize - 1);
		result[idx] = arrA[idx] + arrB[idx];
	}

	dfloat sinc(float x) {
		float PI_x = CUDART_PI_F * x;
		return sin(PI_x) / (PI_x);
	}

	dfloat lanczos2(float x) {
		if (x > -1e-6f && x < 1e-6f) {
			return 1.f;
		}

		if (x < -2.f || x > 2.f) {
			return 0.f;
		}

		return sinc(x) * sinc(x / 2.f);
	}

	dfloat lanczos3(float x) {
		if (x > -1e-6f && x < 1e-6f) {
			return 1.f;
		}

		if (x < -3.f || x > 3.f) {
			return 0.f;
		}

		return sinc(x) * sinc(x / 3.f);
	}

	dfloat lanczos2D(float x, float y, int window) {
		if (window != 2 && window != 3) {
			return 0.f;
		}

		if (window == 2) {
			return lanczos2(x) * lanczos2(y);
		}
		
		return lanczos3(x) * lanczos3(y);
	}

	dfloat nearestNeighbour(float x, float y, int window) {
		return x >= -0.5f && x <= 0.5f && y >= -0.5f && y <= 0.5f;
	}

	dvoid convolve(
		const unsigned char *inImg,
		samplingKernel kernel,
		float2 sample,
		int2 rangeX,
		int2 rangeY,
		int inputWidth,
		int numComp,
		int window,
		unsigned char *result
	) {
		float result_[4];
		for (int i = 0; i < numComp; ++i) {
			result_[i] = 0;
		}

		for (int i = rangeY.x; i < rangeY.y; ++i) {
			for (int j = rangeX.x; j < rangeX.y; ++j) {
				int inputIdx = (i * inputWidth + j) * numComp;
				float kernelValue = kernel(sample.x - j, sample.y - i, window);

				for (int k = 0; k < numComp; ++k) {
					const float sampleWeighted = float(inImg[inputIdx + k]) * kernelValue;
					result_[k] += sampleWeighted;
				}
			}
		}

		for (int i = 0; i < numComp; ++i) {
			result[i] = (unsigned char)(min(max(0.f, result_[i]), 255.f));
		}
	}

	// TODO: put params in a struct and make it a constant variable
	gvoid resize(
		const unsigned char *inImg,
		const int inWidth,
		const int inHeight,
		const int numComp,
		const int outWidth,
		const int outHeight,
		unsigned char *outImg
	) {
		const int pixelCount = outWidth * outHeight;
		const int pixelIdx = min(blockIdx.x * blockDim.x + threadIdx.x, pixelCount - 1);

		const float ratioW = float(outWidth) / inWidth;
		const float ratioH = float(outHeight) / inHeight;

		const int outX = pixelIdx % outWidth;
		const int outY = pixelIdx / outWidth;
		
		float2 sample;
		sample.x = (float(outX) + 0.5f) / ratioW;
		sample.y = (float(outY) + 0.5f) / ratioH;

		int2 floorSample = { int(floor(sample.x)), int(floor(sample.y)) };

		// TODO: these may depend on the sampling algorithm chosen.
		const int lancsozWindow = 3;
		int2 rangeX = {
			min(max(0, floorSample.x - lancsozWindow - 1), inWidth),
			min(max(0, floorSample.x + lancsozWindow + 1), inWidth)
		};
		int2 rangeY = {
			min(max(0, floorSample.y - lancsozWindow - 1), inHeight),
			min(max(0, floorSample.y + lancsozWindow + 1), inHeight)
		};

		convolve(inImg, lanczos2D, sample, rangeX, rangeY, inWidth, numComp, lancsozWindow, &outImg[pixelIdx * numComp]);
	}

}

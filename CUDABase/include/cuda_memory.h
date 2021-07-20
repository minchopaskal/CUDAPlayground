#pragma once

#include <cuda_memory_defines.h>

#include <unordered_set>
#include <unordered_map>

struct CUDADefaultAllocator {
	static constexpr AllocatorType type = AllocatorType::Default;
	using CUDAMemBlock = CUDAMemoryBlock<type>;
public:
	CUDAError initialize();

	CUDAError allocate(CUDAMemBlock &memBlock);

	CUDAError upload(const CUDAMemBlock &memBlock, const void *hostPtr, CUstream stream);
	CUDAError download(const CUDAMemBlock &memBlock, void *hostPtr, CUstream stream);

	CUDAError free(const CUDAMemBlock &memBlock);
};

struct CUDAVirtualAllocator {
private:
	struct PhysicalMemAllocation {
		CUDAMemHandle virtualPtr;
		CUDAMemHandle physicalPtr;
		SizeType size;
	};

public:
	static constexpr AllocatorType type = AllocatorType::Virtual;
	using CUDAMemBlock = CUDAMemoryBlock<type>;

public:
	CUDAError initialize();

	CUDAError allocate(CUDAMemBlock &memBlock);

	CUDAError upload(const CUDAMemBlock &memBlock, const void *hostPtr, CUstream stream);
	CUDAError download(const CUDAMemBlock &memBlock, void *hostPtr, CUstream stream);

	CUDAError free(const CUDAMemBlock &memBlock);

private:
	std::unordered_map<CUDAMemBlock, std::vector<PhysicalMemAllocation>> virtualToPhysicalAllocations;
};

//template <class T = CUDADefaultAllocator, class U = CUDAVirtualAllocator>
//struct CUDAFallbackAllocator {
//private:
//	using BaseAllocator = T;
//	using FallbackAllocator = U;
//
//public:
//	CUDAFallbackAllocator() { }
//	~CUDAFallbackAllocator() { }
//
//	CUDAError initialize() {
//		RETURN_ON_CUDA_ERROR(baseAllocator.initialize());
//		RETURN_ON_CUDA_ERROR(fallbackAllocator.initialize());
//	}
//	
//	CUDAError allocate(CUDAMemBlock &memBlock) {
//		if (memBlock.ptr != NULL) {
//			return CUDAError(CUDA_ERROR_UNKNOWN, "FallbackAllocator_ERROR_BLOCK_INITIALIZED", "");
//		}
//
//		if (baseAllocator.allocate(memBlock).hasError()) {
//			RETURN_ON_CUDA_ERROR(fallbackAllocator.allocate(memBlock));
//			fallbackAllocatorBlocks.push_back(&memBlock);
//		} else {
//			baseAllocatorBlocks.push_back(memBlocks);
//		}
//
//		return CUDAError();
//	}
//
//	CUDAError upload(const CUDAMemBlock &memBlock, void *hostPtr) {
//		if (baseAllocator.upload(memBlock, hostPtr).hasError()) {
//			RETURN_ON_CUDA_ERROR(fallbackAllocator.upload(memBlock, hostPtr));
//		}
//
//		return CUDAError();
//	}
//
//	CUDAError free(const CUDAMemBlock &memBlock) {
//		if (baseAllocator.free(memBlock, hostPtr).hasError()) {
//			RETURN_ON_CUDA_ERROR(fallbackAllocator.upload(memBlock, hostPtr));
//		}
//
//		return CUDAError();
//	}
//
//private:
//	BaseAllocator baseAllocator;
//	FallbackAllocator fallbackAllocator;
//};
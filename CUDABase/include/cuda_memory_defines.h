#pragma once

#include <cuda_error_handling.h>
#include <cstdint>
#include <functional>

#define MEGABYTE_IN_BYTES 1'000'000

using CUDAMemHandle = CUmemGenericAllocationHandle;
using SizeType = unsigned long long;

enum class AllocatorType : uint8_t {
	Default,
	Pinned,
	Virtual
};

template <AllocatorType allocatorType>
struct CUDAMemoryBlock {
	CUDAMemHandle ptr;
	SizeType size;

	CUDAMemoryBlock() : ptr(NULL), size(0) { }
	CUDAMemoryBlock(CUDAMemHandle ptr, SizeType size) : ptr(ptr), size(size) { }

	bool operator==(const CUDAMemoryBlock &other) const {
		return ptr == other.ptr && size == other.size;
	}
};

namespace std {
template <AllocatorType allocatorType>
struct hash<CUDAMemoryBlock<allocatorType>> {
	std::size_t operator()(const CUDAMemoryBlock<allocatorType> &memBlock) const {
		return
			hash<CUDAMemHandle>()(memBlock.ptr) ^
			(hash<SizeType>()(memBlock.size) << 1);
	}
};
}
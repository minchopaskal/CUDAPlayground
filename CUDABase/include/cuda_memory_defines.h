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
	SizeType reserved;

	CUDAMemoryBlock() : ptr(NULL), size(0), reserved(0) { }
	CUDAMemoryBlock(CUDAMemHandle ptr, SizeType size) : ptr(ptr), size(size), reserved(size) { }

	bool operator==(const CUDAMemoryBlock &other) const {
		const bool result = ptr == other.ptr;
		if (result) {
			massert(size == other.size && reserved == other.reserved);
		}

		return result;
	}
};

namespace std {
template <AllocatorType allocatorType>
struct hash<CUDAMemoryBlock<allocatorType>> {
	std::size_t operator()(const CUDAMemoryBlock<allocatorType> &memBlock) const {
		return
			// TODO: better hash
			hash<CUDAMemHandle>()(memBlock.ptr) ^
			(hash<SizeType>()(memBlock.size) << 8) ^
			(hash<SizeType>()(memBlock.reserved) << 16);
	}
};
}
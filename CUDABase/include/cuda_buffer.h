#pragma once

#include <cuda_manager.h>

template <class T>
struct CUDABuffer {
private:
	using Allocator = T;

public:
	using CUDAMemBlock = CUDAMemoryBlock<Allocator::type>;

public:
	CUDABuffer() { }

	~CUDABuffer() {
		deinitialize();
	}

	CUDABuffer(const CUDABuffer&) = delete;
	CUDABuffer &operator=(const CUDABuffer&) = delete;
	CUDABuffer(CUDABuffer&&) = delete;
	CUDABuffer &operator=(CUDABuffer&&) = delete;

	CUDAError initialize(SizeType size) {
		if (size == 0) {
			return CUDAError();
		}

		if (memBlock.ptr != NULL && memBlock.reserved >= size) {
			memBlock.size = size;
			return CUDAError();
		}
		
		deinitialize();

		memBlock.reserved = size;
		memBlock.size = size;

		CUDAManager &cudaman = getCUDAManager();
		Allocator &allocator = cudaman.getAllocator<Allocator>();
		RETURN_ON_CUDA_ERROR_HANDLED(allocator.allocate(memBlock));

		return CUDAError();
	}

	CUDAError deinitialize() {
		if (memBlock.ptr == NULL) {
			massert(memBlock.size == 0);
			memBlock.size = 0;
			memBlock.reserved = 0;
			return CUDAError();
		}

		CUDAManager &cudaman = getCUDAManager();
		Allocator &allocator = cudaman.getAllocator<Allocator>();
		RETURN_ON_CUDA_ERROR_HANDLED(allocator.free(memBlock));
		memBlock.ptr = NULL;
		memBlock.size = 0;
		memBlock.reserved = 0;

		return CUDAError();
	}

	CUDAError upload(const void *hostPtr) {
		return uploadAsync(hostPtr, NULL);
	}

	CUDAError uploadAsync(const void *hostPtr, CUstream stream) {
		if (memBlock.ptr == NULL) {
			massert(memBlock.size == 0);
			return CUDAError(CUDA_ERROR_NOT_INITIALIZED, "CUDABuffer_ERROR_NOT_INITIALIZED", "Attempt to upload uninitalized CUDABuffer!");
		}

		if (hostPtr == nullptr) {
			return CUDAError(CUDA_ERROR_UNKNOWN, "CUDABuffer_ERROR_IVALID_HOST_HANDLE", "");
		}

		CUDAManager &cudaman = getCUDAManager();
		Allocator &allocator = cudaman.getAllocator<Allocator>();
		RETURN_ON_CUDA_ERROR_HANDLED(allocator.upload(memBlock, hostPtr, stream));

		return CUDAError();
	}

	CUDAError download(void *hostPtr) {
		return downloadAsync(hostPtr, NULL);
	}

	CUDAError downloadAsync(void *hostPtr, CUstream stream) {
		if (memBlock.ptr == NULL) {
			massert(memBlock.size == 0);
			return CUDAError(CUDA_ERROR_NOT_INITIALIZED, "CUDABuffer_ERROR_NOT_INITIALIZED", "Attempt to upload uninitalized CUDABuffer!");
		}

		if (hostPtr == nullptr) {
			return CUDAError(CUDA_ERROR_UNKNOWN, "CUDABuffer_ERROR_IVALID_HOST_HANDLE", "");
		}

		CUDAManager &cudaman = getCUDAManager();
		Allocator &allocator = cudaman.getAllocator<Allocator>();
		RETURN_ON_CUDA_ERROR_HANDLED(allocator.download(memBlock, hostPtr, stream));

		return CUDAError();
	}

	CUDAMemHandle handle() {
		return memBlock.ptr;
	}

	SizeType getSize() const {
		return memBlock.size;
	}

private:
	CUDAMemBlock memBlock;
};

using CUDADefaultBuffer = CUDABuffer<CUDADefaultAllocator>;
using CUDAVirtualBuffer = CUDABuffer<CUDAVirtualAllocator>;

template <typename T>
struct CUDAPinnedMemoryBuffer {
private:
	using Allocator = T;

public:
	using CUDAMemBlock = CUDAMemoryBlock<T::type>;
	using HostMemHandle = void*;

public:
	CUDAPinnedMemoryBuffer() : hostPtr(nullptr) { }

	~CUDAPinnedMemoryBuffer() {
		deinitialize();
	}

	// TODO: try zero copy. Somehow we need to know which device allocates the pinned memory so we can ask
	// if it has the CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM so it can just use the hostPtr
	CUDAError initialize(SizeType size) {
		if (size <= 0) {
			return CUDAError(CUDA_ERROR_UNKNOWN, "CUDAPinnedMemoryBuffer_INVALID_INIT_ARGUMENTS", "");
		}

		if (memBlock.ptr != NULL && memBlock.reserved >= size) {
			memBlock.size = size;
			return CUDAError();
		}

		if (memBlock.ptr != NULL && memBlock.reserved < size) {
			deinitialize();
		}

		if (hostPtr != nullptr) {
			RETURN_ON_CUDA_ERROR(cuMemHostAlloc(&hostPtr, size, CU_MEMHOSTALLOC_PORTABLE));
		}

		memBlock.size = size;
		memBlock.reserved = size;
		
		CUDAManager &cudaman = getCUDAManager();
		Allocator allocator = cudaman.getAllocator<Allocator>();
		RETURN_ON_CUDA_ERROR_HANDLED(allocator.allocate(memBlock));

		return CUDAError();
	}

	CUDAError deinitialize() {
		if (hostPtr != nullptr) {
			RETURN_ON_CUDA_ERROR(cuMemFreeHost(hostPtr));
			hostPtr = nullptr;
		}

		if (memBlock.ptr == NULL) {
			massert(hostPtr == nullptr);
			massert(memBlock.size == 0);
			memBlock.size = 0;
			memBlock.reserved = 0;
			return CUDAError();
		}

		CUDAManager &cudaman = getCUDAManager();
		Allocator allocator = cudaman.getAllocator<Allocator>();
		RETURN_ON_CUDA_ERROR_HANDLED(allocator.free(memBlock));
		memBlock.ptr = NULL;
		memBlock.size = 0;
		memBlock.reserved = 0;

		return CUDAError();
	}

	CUDAError upload() {
		return uploadAsync(NULL);
	}

	CUDAError uploadAsync(CUstream stream) {
		if (memBlock.ptr == NULL) {
			return CUDAError(CUDA_ERROR_NOT_INITIALIZED, "CUDAPinnedMemoryBuffer_NOT_INITIALIZED", "");
		}

		if (hostPtr == nullptr) {
			return CUDAError(CUDA_ERROR_UNKNOWN, "CUDABuffer_ERROR_IVALID_HOST_HANDLE", "");
		}

		massert(memBlock.size > 0);

		CUDAManager &cudaman = getCUDAManager();
		Allocator &allocator = cudaman.getAllocator<Allocator>();
		RETURN_ON_CUDA_ERROR_HANDLED(allocator.upload(memBlock, hostPtr, stream));

		return CUDAError();
	}

	CUDAError download() {
		return downloadAsync(NULL);
	}

	CUDAError downloadAsync(CUstream stream) {
		if (memBlock.ptr == NULL) {
			return CUDAError(CUDA_ERROR_NOT_INITIALIZED, "CUDAPinnedMemoryBuffer_NOT_INITIALIZED", "");
		}

		if (hostPtr == nullptr) {
			return CUDAError(CUDA_ERROR_UNKNOWN, "CUDABuffer_ERROR_IVALID_HOST_HANDLE", "");
		}

		massert(memBlock.size > 0);

		CUDAManager &cudaman = getCUDAManager();
		Allocator &allocator = cudaman.getAllocator<Allocator>();
		RETURN_ON_CUDA_ERROR_HANDLED(allocator.download(memBlock, hostPtr, stream));

		return CUDAError();
	}

	HostMemHandle hostHandle() const {
		return hostPtr;
	}

	CUDAMemHandle handle() const {
		return memBlock.ptr;
	}

	SizeType getSize() const {
		return memBlock.size;
	}

private:
	CUDAMemBlock memBlock;
	HostMemHandle hostPtr;
};

using CUDADefaultPinnedBuffer = CUDAPinnedMemoryBuffer<CUDADefaultAllocator>;
using CUDAVirtualPinnedBuffer = CUDAPinnedMemoryBuffer<CUDAVirtualAllocator>;

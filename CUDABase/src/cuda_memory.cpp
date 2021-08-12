#include <cuda_memory.h>
#include <cuda_manager.h>

/*
===============================================================
CUDADefaultAllocator
===============================================================
*/
CUDAError CUDADefaultAllocator::initialize() {
	return CUDAError();
}

CUDAError CUDADefaultAllocator::deinitialize() {
	// TODO: we need the context also
	for (auto it = allocations.begin(); it != allocations.end(); ++it) {
		RETURN_ON_CUDA_ERROR_HANDLED(internalFree(**it));
	}

	allocations.clear();

	return CUDAError();
}

CUDAError CUDADefaultAllocator::allocate(CUDAMemBlock &memBlock) {
	if (memBlock.size <= 0) {
		return CUDAError(CUDA_ERROR_UNKNOWN, "CUDADefaultAllocator_ERROR_INVALID_SIZE", "");
	}

	RETURN_ON_CUDA_ERROR(cuMemAlloc(reinterpret_cast<CUdeviceptr*>(&memBlock.ptr), size_t(memBlock.size)));

	allocations.insert(&memBlock);

	return CUDAError();
}

CUDAError CUDADefaultAllocator::upload(const CUDAMemBlock &memBlock, const void *hostPtr, CUstream stream) {
	massert(memBlock.size > 0);

	if (stream != NULL) {
		RETURN_ON_CUDA_ERROR(cuMemcpyHtoDAsync(memBlock.ptr, hostPtr, memBlock.size, stream));
	} else {
		RETURN_ON_CUDA_ERROR(cuMemcpyHtoD(memBlock.ptr, hostPtr, memBlock.size));
	}
	return CUDAError();
}

CUDAError CUDADefaultAllocator::download(const CUDAMemBlock &memBlock, void *hostPtr, CUstream stream) {
	massert(memBlock.size > 0);

	if (stream != NULL) {
		RETURN_ON_CUDA_ERROR(cuMemcpyDtoHAsync(hostPtr, memBlock.ptr, memBlock.size, stream));
	} else {
		RETURN_ON_CUDA_ERROR(cuMemcpyDtoH(hostPtr, memBlock.ptr, memBlock.size));
	}
	return CUDAError();
}

CUDAError CUDADefaultAllocator::free(CUDAMemBlock &memBlock) {
	RETURN_ON_CUDA_ERROR_HANDLED(internalFree(memBlock));

	auto it = allocations.find(&memBlock);
	massert(it != allocations.end());
	allocations.erase(it);

	return CUDAError();
}

CUDAError CUDADefaultAllocator::internalFree(CUDAMemBlock &memBlock) {
	RETURN_ON_CUDA_ERROR(cuMemFree(static_cast<CUdeviceptr>(memBlock.ptr)));
	memBlock.ptr = NULL;
	memBlock.size = 0;
	memBlock.reserved = 0;

	return CUDAError();
}

/*
===============================================================
CUDAVirtualAllocator
===============================================================
*/
SizeType getPaddedSize(SizeType size, SizeType granularity) {
	return ((size + granularity - 1) / granularity) * granularity;
}

CUDAError CUDAVirtualAllocator::initialize() {
	return CUDAError();
}

CUDAError CUDAVirtualAllocator::deinitialize() {
	return CUDAError();
}

CUDAError CUDAVirtualAllocator::allocate(CUDAMemBlock &memBlock) {
	if (memBlock.size <= 0) {
		return CUDAError(CUDA_ERROR_UNKNOWN, "CUDAVirtualAllocator_ERROR_INVALID_SIZE", "");
	}

	CUDAManager &cudaManager = getCUDAManager();
	const std::vector<CUDADevice> &devices = cudaManager.getDevices();

	// Check if we have the required memory (plus some just in case bytes over it)
	// on any one of the devices. 
	const SizeType jicBytes = 64 * MEGABYTE_IN_BYTES;
	const SizeType requiredMemory = memBlock.size + jicBytes;
	int devIdx = -1;
	for (int i = 0; i < devices.size(); ++i) {
		const CUDADevice &dev = devices[i];
		SizeType currDeviceFreeMem;
		RETURN_ON_CUDA_ERROR_HANDLED(dev.getFreeMemory(currDeviceFreeMem));
		if (currDeviceFreeMem >= requiredMemory) {
			devIdx = i;
			break;
		}
	}

	if (devIdx < 0 || devIdx >= devices.size()) {
		return CUDAError(CUDA_ERROR_OUT_OF_MEMORY, "CUDAVirtualAllocator_ERROR_OUT_OF_MEM", "");
	}

	CUmemAllocationProp allocationProperties = {};
	allocationProperties.type = CU_MEM_ALLOCATION_TYPE_PINNED;
	allocationProperties.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
	allocationProperties.location.id = devIdx;
	SizeType granularity = 0;
	cuMemGetAllocationGranularity(&granularity, &allocationProperties, CU_MEM_ALLOC_GRANULARITY_MINIMUM);

	memBlock.size = getPaddedSize(memBlock.size, granularity);
	RETURN_ON_CUDA_ERROR(cuMemAddressReserve(&memBlock.ptr, memBlock.size, 0, 0, 0));

	// Try to create physical blocks which will be mapped to the virtual adress range we just reserved.
	// If an allocation fails, ask for two times less memory. If we start asking for less memory than is
	// the padding size(`granulariry`) then the memory is just too fragmeneted so we fail.
	// Each time a physical block is allocated - it is mapped to a sub-region of the virtual range and
	// is saved in virtualToPhysicalAllocations, so we can later unmap and release it.
	CUDAMemHandle currPtr = memBlock.ptr;
	SizeType requiredMemorySize = memBlock.size;
	SizeType physicalAllocationSize = requiredMemorySize;
	while (requiredMemorySize > 0) {
		CUDAMemHandle physicalMemHandle;
		CUresult res = cuMemCreate(&physicalMemHandle, physicalAllocationSize, &allocationProperties, 0);
		if (res != CUDA_SUCCESS) {
			// Memory is too defragmented. Free all allocations and fail.
			if (physicalAllocationSize == granularity) {
				RETURN_ON_CUDA_ERROR_HANDLED(free(memBlock));
				return CUDAError(CUDA_ERROR_OUT_OF_MEMORY, "CUDAVirtualAllocator_ERROR_OUT_OF_MEM", "");
			}

			physicalAllocationSize = getPaddedSize(physicalAllocationSize / 2, granularity);
			continue;
		}

		RETURN_ON_CUDA_ERROR(cuMemMap(currPtr, physicalAllocationSize, 0, physicalMemHandle, 0));

		requiredMemorySize -= physicalAllocationSize;
		currPtr += physicalAllocationSize;

		PhysicalMemAllocation physicalMemAlloc = { currPtr, physicalMemHandle, physicalAllocationSize };
		auto &block = virtualToPhysicalAllocations[memBlock];
		block.push_back(physicalMemAlloc);
	}
	massert(requiredMemorySize == 0);

	CUmemAccessDesc accessDesc = {};
	accessDesc.location = allocationProperties.location;
	accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
	RETURN_ON_CUDA_ERROR(cuMemSetAccess(memBlock.ptr, memBlock.size, &accessDesc, 1));

	return CUDAError();
}

CUDAError CUDAVirtualAllocator::upload(const CUDAMemBlock &memBlock, const void *hostPtr, CUstream stream) {
	massert(memBlock.size > 0);

	std::vector<PhysicalMemAllocation> &blocks = virtualToPhysicalAllocations[memBlock];
	CUDAMemHandle hostPtrUVA = reinterpret_cast<CUDAMemHandle>(hostPtr);
	SizeType offset = 0;
	for (int i = 0; blocks.size(); ++i) {
		auto memAlloc = blocks[i];
		CUDAMemHandle dstDevice = memAlloc.virtualPtr;
		CUDAMemHandle srcHost = hostPtrUVA + offset;

		/*if (stream != NULL) {
			RETURN_ON_CUDA_ERROR(cuMemcpyAsync(dstDevice, srcHost, memAlloc.size, stream));
		} else {
			RETURN_ON_CUDA_ERROR(cuMemcpy(dstDevice, srcHost, memAlloc.size));
		}*/
		offset += memAlloc.size;
	}

	return CUDAError();
}

CUDAError CUDAVirtualAllocator::download(const CUDAMemBlock &memBlock, void *hostPtr, CUstream stream) {
	massert(memBlock.size > 0);

	std::vector<PhysicalMemAllocation> blocks = virtualToPhysicalAllocations[memBlock];
	CUDAMemHandle hostPtrUVA = reinterpret_cast<CUDAMemHandle>(hostPtr);
	SizeType offset = 0;
	for (int i = 0; blocks.size(); ++i) {
		auto memAlloc = blocks[i];
		CUDAMemHandle srcDevice = memAlloc.virtualPtr;
		CUDAMemHandle dstHost = hostPtrUVA + offset;
		if (stream != NULL) {
			RETURN_ON_CUDA_ERROR(cuMemcpyAsync(dstHost, srcDevice, memAlloc.size, stream));
		} else {
			RETURN_ON_CUDA_ERROR(cuMemcpy(dstHost, srcDevice, memAlloc.size));
		}
		offset += memAlloc.size;
	}

	return CUDAError();
}

CUDAError CUDAVirtualAllocator::free(CUDAMemBlock &memBlock) {
	const std::vector<PhysicalMemAllocation> &allocs = virtualToPhysicalAllocations[memBlock];
	for (PhysicalMemAllocation memAlloc : allocs) {
		RETURN_ON_CUDA_ERROR(cuMemUnmap(memAlloc.virtualPtr, memAlloc.size));
		RETURN_ON_CUDA_ERROR(cuMemRelease(memAlloc.physicalPtr));
	}

	RETURN_ON_CUDA_ERROR(cuMemAddressFree(memBlock.ptr, memBlock.size));

	return CUDAError();
}

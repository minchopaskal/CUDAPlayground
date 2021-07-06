set(LIB_SOURCE_DIR ${PROJECT_SOURCE_DIR}/CUDABase)

set(INCLUDE_DIR ${LIB_SOURCE_DIR}/include)
set(SRC_DIR ${LIB_SOURCE_DIR}/src)
set(RESOURCES_DIR ${LIB_SOURCE_DIR}/gpu)

set(HEADERS
	${INCLUDE_DIR}/cuda_buffer.h
	${INCLUDE_DIR}/cuda_error_handling.h
	${INCLUDE_DIR}/cuda_manager.h
	${INCLUDE_DIR}/cuda_memory.h
	${INCLUDE_DIR}/cuda_memory_defines.h
	${INCLUDE_DIR}/logger.h
	${INCLUDE_DIR}/timer.h
)

set(SOURCES
	${SRC_DIR}/cuda_manager.cpp
	${SRC_DIR}/cuda_memory.cpp
	${SRC_DIR}/logger.cpp
)

source_group("src"           FILES ${SOURCES})
source_group("include"       FILES ${HEADERS})
source_group("gpu"           FILES ${RESOURCES_DIR}/kernel.cu)

add_library(CUDABaseLib ${HEADERS} ${SOURCES})

set_property(TARGET CUDABaseLib PROPERTY VS_DEBUGGER_WORKING_DIRECTORY $<TARGET_FILE_DIR:CUDABaseLib>)

target_compile_definitions(
	CUDABaseLib
	PRIVATE
	$<$<CONFIG:Debug>:CUDA_DEBUG>
	$<$<CONFIG:Release>:CUDA_NDEBUG>
)

target_compile_definitions(
	CUDABaseLib
	PRIVATE
	_CRT_SECURE_NO_WARNINGS
)

target_link_directories(CUDABaseLib PRIVATE ${LIB_DIR})

target_link_libraries(CUDABaseLib CUDA::cuda_driver)

target_include_directories(
	CUDABaseLib
	PRIVATE
	${INCLUDE_DIR}
)

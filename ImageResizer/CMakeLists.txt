set(IR_SOURCE_DIR ${PROJECT_SOURCE_DIR}/ImageResizer)

set(INCLUDE_DIR ${IR_SOURCE_DIR}/include)
set(SRC_DIR ${IR_SOURCE_DIR}/src)
set(RESOURCES_DIR ${IR_SOURCE_DIR}/gpu)

set(HEADERS
	${INCLUDE_DIR}/image_resizer.h
)

set(SOURCES
	${SRC_DIR}/image_resizer.cpp
	${SRC_DIR}/main.cpp
)

set(GPU
	${RESOURCES_DIR}/resize_kernel.cu
)

source_group("src"           FILES ${SOURCES})
source_group("include"       FILES ${HEADERS})
source_group("gpu"           FILES ${GPU})

add_executable(ImageResizer ${HEADERS} ${SOURCES})

set_property(TARGET ImageResizer PROPERTY VS_DEBUGGER_WORKING_DIRECTORY $<TARGET_FILE_DIR:ImageResizer>)

target_compile_definitions(
	ImageResizer
	PRIVATE
	$<$<CONFIG:Debug>:CUDA_DEBUG>
	$<$<CONFIG:Release>:CUDA_NDEBUG>
	_CRT_SECURE_NO_WARNINGS
)

target_link_directories(ImageResizer PRIVATE ${LIB_DIR})

target_link_libraries(ImageResizer CUDABaseLib)

target_include_directories(
	ImageResizer
	PRIVATE
	${INCLUDE_DIR}
	${PROJECT_SOURCE_DIR}/CUDABase/include
	${IR_SOURCE_DIR}
)

compilePtx(ImageResizer ${GPU})

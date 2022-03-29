cmake_minimum_required(VERSION 3.17)

project(CUDABase VERSION 0.1 LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
	set(CMAKE_CUDA_STANDARD 11)
	set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "Configs" FORCE)

option(COMPILE_PROJECTS "Compile non base projects" FALSE)

find_package(CUDAToolkit)

macro(compilePtx TRG GPU_FILES ADDITIONAL_ARGS RDC)
	if (NOT CUDAToolkit_FOUND)
		find_package(CUDAToolkit)
	endif()

	add_custom_command(
		TARGET ${TRG}
		COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${TRG}>/data/
	)

	foreach(GPU_FILE ${GPU_FILES})
		get_filename_component(FDIR ${GPU_FILE} DIRECTORY)
		get_filename_component(FNAME ${GPU_FILE} NAME)
		get_filename_component(FNAME_NOEXT ${GPU_FILE} NAME_WE)
		get_filename_component(FILE_EXT ${GPU_FILE} LAST_EXT)

		if (${FILE_EXT} STREQUAL ".cuh")
			continue()
		endif()
		
		set(COMPILE_ARGS
			-ptx
			-m 64
			-t 0
			-arch=sm_52
			$<$<CONFIG:Debug>:-G>
			${ADDITIONAL_ARGS}
			-o $<TARGET_FILE_DIR:${TRG}>/data/${FNAME_NOEXT}.ptx
		)
		if (${RDC})
			list(APPEND COMPILE_ARGS -rdc=true) # tells nvcc that we will do separate compile and linking steps
		endif()
		list(APPEND COMPILE_ARGS 
			${GPU_FILE}
		)

		add_custom_command(
			TARGET ${TRG}
			COMMAND ${CUDAToolkit_NVCC_EXECUTABLE} ${COMPILE_ARGS}
		)
	endforeach()
endmacro()

add_subdirectory(CUDABase)
if (COMPILE_PROJECTS)
add_subdirectory(ImageResizer)
endif()

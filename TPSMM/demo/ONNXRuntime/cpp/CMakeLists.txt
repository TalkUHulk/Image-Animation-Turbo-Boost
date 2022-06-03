cmake_minimum_required(VERSION 3.19)
project(onnxruntime_demo)

set(CMAKE_CXX_STANDARD 11)
set(ORT_PATH ${PROJECT_SOURCE_DIR}/ort)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)

file(GLOB_RECURSE SOURCE_FILES ${PROJECT_SOURCE_DIR}/src/*.cpp ${PROJECT_SOURCE_DIR}/src/*.c)


IF (WIN32)
    MESSAGE(STATUS "Now is windows")
ELSEIF (APPLE)
    MESSAGE(STATUS "Now is Apple systems.")
    set(ORT_INCLUDE_DIRS ${ORT_PATH}/mac/include)
    find_library(LIBORT onnxruntime ${ORT_PATH}/mac/lib)
    include_directories(${ORT_PATH}/mac/include/ort)
ELSEIF (UNIX)
    MESSAGE(STATUS "Now is UNIX-like OS's. Including aPPLE os x  and CygWin")
    set(ORT_INCLUDE_DIRS ${ORT_PATH}/x64/include)
    find_library(LIBORT onnxruntime ${ORT_PATH}/x64/lib)
    find_library(LIBORT_CUDA onnxruntime_providers_cuda ${ORT_PATH}/x64/lib)
    find_library(LIBORT_SHARED onnxruntime_providers_shared ${ORT_PATH}/x64/lib)
    include_directories(${ORT_PATH}/x64/include/ort)
ENDIF ()


if (LIBORT AND  LIBORT_CUDA AND LIBORT_SHARED)
    add_definitions(-DUSE_ORT)
    include_directories(${ORT_INCLUDE_DIRS})
endif ()


add_executable(onnx_demo demo.cpp ${SOURCE_FILES})
target_link_libraries(onnx_demo  ${OpenCV_LIBS} ${LIBORT}  ${LIBORT_CUDA} ${LIBORT_SHARED} )

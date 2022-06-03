cmake_minimum_required(VERSION 3.19)
project(openvino)
set(CMAKE_CXX_STANDARD 11)


# OpenVINO
set(ngraph_DIR "/opt/intel/openvino_2021/deployment_tools/ngraph/cmake")
find_package(ngraph REQUIRED)
set(InferenceEngine_DIR "/opt/intel/openvino_2021/deployment_tools/inference_engine/share")
find_package(InferenceEngine REQUIRED)
set(OpenCV_DIR "/opt/intel/openvino_2021/opencv/cmake")
find_package(OpenCV REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include)
file(GLOB_RECURSE SOURCE_FILES ${PROJECT_SOURCE_DIR}/src/*.cpp ${PROJECT_SOURCE_DIR}/src/*.c)

add_executable(openvino_demo demo.cpp ${SOURCE_FILES})
target_link_libraries(
        ${PROJECT_NAME}
        PRIVATE ${InferenceEngine_LIBRARIES}
        ${OpenCV_LIBS}
        ${NGRAPH_LIBRARIES})

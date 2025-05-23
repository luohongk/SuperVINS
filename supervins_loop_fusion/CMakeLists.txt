cmake_minimum_required(VERSION 2.8.3)
project(supervins_loop_fusion)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++14")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

find_package(catkin REQUIRED COMPONENTS
    roscpp
    std_msgs
    nav_msgs
    camera_models
    cv_bridge
    roslib
    )
find_package(Ceres REQUIRED)
find_package(Ceres REQUIRED PATHS "/home/lhk/Thirdparty/Ceres")
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(Eigen3)

set(CMAKE_CXX_STANDARD 14)
option(onnxruntime_USE_CUDA "Build with CUDA support" OFF)
option(onnxruntime_USE_TENSORRT "Build with TensorRT support" OFF)
option(LIBPNG_ROOTDIR "libpng root dir")
option(ONNXRUNTIME_ROOTDIR "onnxruntime root dir")

set(ONNXRUNTIME_ROOTDIR "/home/lhk/Thirdparty/onnxruntime")
include_directories("${ONNXRUNTIME_ROOTDIR}/include")
link_directories("${ONNXRUNTIME_ROOTDIR}/lib")

find_package(JPEG)
find_package(PNG)
find_package(OpenCV REQUIRED)
INCLUDE_DIRECTORIES(
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ThirdParty/DBoW3/src/
)

file(GLOB DBOW3_SOURCE "src/ThirdParty/DBoW3/src/*.cpp" "src/ThirdParty/DBoW3/src/*.c")

link_libraries(${OpenCV_LIBRARIES})


include_directories(${catkin_INCLUDE_DIRS} ${CERES_INCLUDE_DIRS}  ${EIGEN3_INCLUDE_DIR})

catkin_package()

add_executable(supervins_loop_fusion_node
    src/pose_graph_node.cpp
    src/pose_graph.cpp
    src/keyframe.cpp
    src/utility/CameraPoseVisualization.cpp
    src/ThirdParty/DBoW3/src/BowVector.cpp
    src/ThirdParty/DBoW3/src/Database.cpp
    src/ThirdParty/DBoW3/src/DescManip.cpp
    src/ThirdParty/DBoW3/src/FeatureVector.cpp
    src/ThirdParty/DBoW3/src/QueryResults.cpp
    src/ThirdParty/DBoW3/src/ScoringObject.cpp
    src/ThirdParty/DBoW3/src/Vocabulary.cpp
    src/ThirdParty/DUtils/Random.cpp
    src/ThirdParty/DUtils/Timestamp.cpp
    src/ThirdParty/DVision/BRIEF.cpp
    src/ThirdParty/VocabularyBinary.cpp
    src/ThirdParty/DBoW3/src/DBoW3.h
    src/ThirdParty/DBoW3/src/Vocabulary.h
    src/ThirdParty/DBoW3/src/BowVector.h
    src/ThirdParty/DBoW3/src/Database.h
    src/ThirdParty/DBoW3/src/DescManip.h
    src/ThirdParty/DBoW3/src/exports.h
    src/ThirdParty/DBoW3/src/FeatureVector.h
    src/ThirdParty/DBoW3/src/quicklz.c
    src/ThirdParty/DBoW3/src/quicklz.h
    src/ThirdParty/DBoW3/src/QueryResults.h
    src/ThirdParty/DBoW3/src/ScoringObject.h
    src/ThirdParty/DBoW3/src/timers.h
    ${DBOW3_SOURCE}
    )
get_filename_component(PROJECT_SOURCE_DIR_ABS "${CMAKE_SOURCE_DIR}" ABSOLUTE)
target_compile_definitions(supervins_loop_fusion_node PRIVATE PROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR_ABS}")
target_link_libraries(supervins_loop_fusion_node ${catkin_LIBRARIES}  ${OpenCV_LIBS} ${CERES_LIBRARIES}  ${OpenCV_LIBRARIES} onnxruntime) 



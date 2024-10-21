cmake_minimum_required(VERSION 2.8.3)
project(supervins)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++14")
set(CMAKE_CXX_STANDARD 14)
#-DEIGEN_USE_MKL_ALL")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

find_package(catkin REQUIRED COMPONENTS
    roscpp
    std_msgs
    geometry_msgs
    nav_msgs
    tf
    cv_bridge
    camera_models
    image_transport)

find_package(OpenCV REQUIRED)

# message(WARNING "OpenCV_VERSION: ${OpenCV_VERSION}")
# set(Ceres_DIR "/home/lhk/Thirtyparty/Ceres")
find_package(Ceres REQUIRED PATHS "/home/lhk/Thirdparty/Ceres")

#lightglue
set(ONNXRUNTIME_ROOTDIR "/home/lhk/Thirdparty/onnxruntime")
# set(ONNXRUNTIME_LIB ${ONNXRUNTIME_ROOTDIR}/lib)
include_directories("${ONNXRUNTIME_ROOTDIR}/include")
link_directories("${ONNXRUNTIME_ROOTDIR}/lib")
include_directories(${PROJECT_SOURCE_DIR}/featureTracker/ort_include/)
#########

include_directories(${PROJECT_SOURCE_DIR}/src/estimator/)
include_directories(${catkin_INCLUDE_DIRS} ${CERES_INCLUDE_DIRS})

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(Eigen3)
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)

catkin_package()

add_library(supervins_lib
    src/estimator/parameters.cpp
    src/estimator/estimator.cpp
    src/estimator/feature_manager.cpp
    src/factor/pose_local_parameterization.cpp
    src/factor/projectionTwoFrameOneCamFactor.cpp
    src/factor/projectionTwoFrameTwoCamFactor.cpp
    src/factor/projectionOneFrameTwoCamFactor.cpp
    src/factor/marginalization_factor.cpp
    src/utility/utility.cpp
    src/utility/visualization.cpp
    src/utility/CameraPoseVisualization.cpp
    src/initial/solve_5pts.cpp
    src/initial/initial_aligment.cpp
    src/initial/initial_sfm.cpp
    src/initial/initial_ex_rotation.cpp
    src/featureTracker/feature_tracker.cpp
    src/featureTracker/feature_tracker_dpl.cpp
    src/featureTracker/extractor_matcher_dpl.cpp
    src/featureTracker/transform_dpl.cpp)
target_link_libraries(supervins_lib ${catkin_LIBRARIES} ${OpenCV_LIBS} ${CERES_LIBRARIES} onnxruntime)


add_executable(supervins_node src/rosNodeTest.cpp)
target_link_libraries(supervins_node supervins_lib) 

# add_executable(kitti_odom_test src/KITTIOdomTest.cpp )
# target_link_libraries(kitti_odom_test vins_lib -pthread) 

# add_executable(kitti_gps_test src/KITTIGPSTest.cpp)
# target_link_libraries(kitti_gps_test vins_lib -pthread) 


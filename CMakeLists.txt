cmake_minimum_required(VERSION 3.13)

project(control-by-face CXX)

find_package(dlib REQUIRED)
find_package(OpenCV REQUIRED)
find_package (Python3 COMPONENTS Interpreter Development NumPy REQUIRED)

set(CEREAL_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/cereal/include")
set(MATPLOTLIBCPP_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/matplotlib-cpp")

message(STATUS "CEREAL_INCLUDE_DIR: ${CEREAL_INCLUDE_DIR}")
message(STATUS "MATPLOTLIBCPP_INCLUDE_DIR: ${MATPLOTLIBCPP_INCLUDE_DIR}")

add_executable(collector collector.cpp)
target_link_libraries(collector dlib::dlib ${OpenCV_LIBS})
target_include_directories(collector PUBLIC 
    ${MATPLOTLIBCPP_INCLUDE_DIR}
    ${CEREAL_INCLUDE_DIR})

add_executable(trainer trainer.cpp)
target_link_libraries(trainer dlib::dlib ${OpenCV_LIBS} Python3::NumPy Python3::Python)
target_include_directories(trainer PUBLIC 
    ${MATPLOTLIBCPP_INCLUDE_DIR}
    ${CEREAL_INCLUDE_DIR})

add_executable(predictor predictor.cpp)
target_link_libraries(predictor dlib::dlib ${OpenCV_LIBS} Python3::NumPy Python3::Python)
target_include_directories(predictor PUBLIC 
    ${MATPLOTLIBCPP_INCLUDE_DIR}
    ${CEREAL_INCLUDE_DIR})
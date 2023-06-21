cmake_minimum_required(VERSION 3.13)

project(control-by-face CXX)

find_package(dlib REQUIRED)
find_package(OpenCV REQUIRED)

set(CEREAL_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/cereal/include")

message(STATUS "CEREAL_INCLUDE_DIR: ${CEREAL_INCLUDE_DIR}")

add_executable(collector collector.cpp)
target_link_libraries(collector dlib::dlib ${OpenCV_LIBS})
target_include_directories(collector PUBLIC 
    ${CEREAL_INCLUDE_DIR})

add_executable(trainer trainer.cpp)
target_link_libraries(trainer dlib::dlib ${OpenCV_LIBS})
target_include_directories(trainer PUBLIC 
    ${CEREAL_INCLUDE_DIR})

add_executable(predictor predictor.cpp)
target_link_libraries(predictor dlib::dlib ${OpenCV_LIBS})
target_include_directories(predictor PUBLIC 
    ${CEREAL_INCLUDE_DIR})
cmake_minimum_required(VERSION 3.13)

project(control-by-face CXX)

find_package(dlib REQUIRED)
find_package(OpenCV REQUIRED)
find_package(unofficial-nana CONFIG REQUIRED)

set(CEREAL_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/cereal/include")

# 日本語文字列を正しく表示するための設定
if (WIN32)
    add_definitions(-D_UNICODE -DUNICODE)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /utf-8")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /utf-8")
endif()

add_executable(01collector 01collector.cpp)
target_link_libraries(01collector dlib::dlib ${OpenCV_LIBS}  unofficial::nana::nana)
target_include_directories(01collector PUBLIC 
    ${CEREAL_INCLUDE_DIR})

add_executable(02trainer 02trainer.cpp)
target_link_libraries(02trainer dlib::dlib ${OpenCV_LIBS})
target_include_directories(02trainer PUBLIC 
    ${CEREAL_INCLUDE_DIR})

add_executable(03predictor 03predictor.cpp)
target_link_libraries(03predictor dlib::dlib ${OpenCV_LIBS})
target_include_directories(03predictor PUBLIC 
    ${CEREAL_INCLUDE_DIR})
cmake_minimum_required(VERSION 3.13)

project(control-by-face CXX)

find_package(dlib REQUIRED)
find_package(OpenCV REQUIRED)
find_package (Python3 COMPONENTS Interpreter Development NumPy REQUIRED)
find_package(cereal CONFIG REQUIRED)


add_executable(collector collector.cpp)
target_link_libraries(collector dlib::dlib ${OpenCV_LIBS} cereal::cereal)

add_executable(trainer trainer.cpp)
target_link_libraries(trainer dlib::dlib ${OpenCV_LIBS} Python3::NumPy Python3::Python cereal::cereal)
target_include_directories(trainer PUBLIC "${PROJECT_SOURCE_DIR}/matplotlib-cpp")

add_executable(otameshi otameshi.cpp)
target_link_libraries(otameshi dlib::dlib ${OpenCV_LIBS} cereal::cereal)
target_include_directories(otameshi PUBLIC "${PROJECT_SOURCE_DIR}/matplotlib-cpp")
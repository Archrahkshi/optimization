cmake_minimum_required(VERSION 3.20)
project(optimization)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_CXX_FLAGS "-Ofast")
add_executable(${PROJECT_NAME} main.cpp)

find_package(OpenMP REQUIRED)
target_link_libraries(${PROJECT_NAME} OpenMP::OpenMP_CXX)

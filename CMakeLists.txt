cmake_minimum_required(VERSION 3.10)

# set the project name and version
project(Tutorial VERSION 1.0)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# add matplot
find_package(Matplot++ REQUIRED)
find_package(GSL REQUIRED)

 
add_executable(out.o naive_integration.cpp)

target_link_libraries(out.o Matplot++::matplot GSL::gsl)

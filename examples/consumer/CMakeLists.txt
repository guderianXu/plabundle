cmake_minimum_required(VERSION 3.25)
project(plabundle_consumer LANGUAGES CXX)

if(NOT TARGET plabundle::plabundle)
    find_package(plabundle CONFIG REQUIRED)
endif()

add_executable(plabundle_consumer main.cpp)
target_compile_features(plabundle_consumer PRIVATE cxx_std_20)
target_link_libraries(plabundle_consumer PRIVATE plabundle::plabundle)

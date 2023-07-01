cmake_minimum_required(VERSION 3.0.0)
project(ENGN8535Assignment VERSION 0.1.0)

find_package(OpenCV REQUIRED)
find_package(Doxygen)

set(SOURCES 
    main.cpp
    kmeans.hpp
    Ncut.hpp
    io.hpp)
add_executable(ENGN8535Assignment ${SOURCES})
target_link_libraries(ENGN8535Assignment ${OpenCV_LIBS})


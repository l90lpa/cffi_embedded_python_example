cmake_minimum_required(VERSION 3.22)
project(fort_python LANGUAGES C Fortran)

option(LINK_PYTHON_DYNAMICALLY "Choose to embed Python dynamically or statically." TRUE)
message(STATUS "LINK_PYTHON_DYNAMICALLY=${LINK_PYTHON_DYNAMICALLY}")

add_subdirectory(src)
#cmake -G "MinGW Makefiles" -S . - B build
# if needed use gcc-8
#cmake -G "" -S . - B build -D CMAKE_C_COMPILER=gcc-8 -D CMAKE_CXX_COMPILER=g++-8

cmake_minimum_required(VERSION 3.5)
project(heisenberg)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR})
set(CMAKE_VERBOSE_MAKEFILE on )

set( OpenGL_GL_PREFERENCE GLVND)
find_package(OpenGL REQUIRED)
find_package(CUDA REQUIRED)
find_package(OpenMP)
find_package(GLEW)

find_package(PkgConfig REQUIRED)
pkg_search_module(GLFW REQUIRED glfw3)

set(GL_TOOLS_PATH "${PROJECT_SOURCE_DIR}/include/gltools")
set(GLM_PATH "${PROJECT_SOURCE_DIR}/include/glm")

set(SOURCES main.cu update.cu energy.cu funs.cu main_kernels.cu)

#set(CMAKE_C_COMPILER "gcc-8")
#set(CMAKE_CXX_COMPILER "g++-8")

set(CUDA_NVCC_FLAGS "-O3 -use_fast_math -keep -arch=compute_75 -code=sm_75")

set(COMPILER_FLAGS "-Ofast -march=native -ffast-math -funroll-loops -fopenmp")
SET(CMAKE_CXX_FLAGS ${COMPILER_FLAGS})

set(MYHEADS "${PROJECT_SOURCE_DIR}/include")

include_directories(${MYHEADS} ${GLFW_INCLUDE_DIR} ${GLEW_INCLUDE_DIR} ${GL_TOOLS_PATH} ${GLM_PATH})

cuda_add_executable(heisenberg ${SOURCES})

target_link_libraries(heisenberg glfw ${GLFW_LIBRARY_DIR} ${OPENGL_LIBRARY} ${GLEW_LIBRARIES} )

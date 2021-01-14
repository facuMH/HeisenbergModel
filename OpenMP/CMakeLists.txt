#cmake -G "MinGW Makefiles" -S . - B build
# if needed use gcc-8
#cmake -G "" -S . - B build -D CMAKE_C_COMPILER=gcc-8 -D CMAKE_CXX_COMPILER=g++-8

cmake_minimum_required(VERSION 3.5)
project(heisenberg)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR})
set(CMAKE_VERBOSE_MAKEFILE on )

set( OpenGL_GL_PREFERENCE GLVND)
find_package(OpenGL REQUIRED)
find_package(OpenMP)
find_package(GLEW)

find_package(PkgConfig REQUIRED)
pkg_search_module(GLFW REQUIRED glfw3)

find_package(PkgConfig REQUIRED)
pkg_search_module(GLFW REQUIRED glfw3)

set(GL_TOOLS_PATH "${PROJECT_SOURCE_DIR}/include/gltools")
set(GLM_PATH "${PROJECT_SOURCE_DIR}/include/glm")

set(GL_TOOLS_PATH ${PROJECT_SOURCE_DIR}/include/gltools)
set(GLM_PATH ${PROJECT_SOURCE_DIR}/include/glm)
set(LCG64_LIB ${PROJECT_SOURCE_DIR}/lib/libtrng4.a)

set(SOURCES main.cpp update.cpp energy.cpp funs.cpp main_kernels.cpp)

#set(CMAKE_C_COMPILER "gcc-8")
#set(CMAKE_CXX_COMPILER "g++-8")

set(GCC_COMPILE_FLAGS "-g -Ofast -march=native -ffast-math -funroll-loops -fopenmp ")
SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${GCC_COMPILE_FLAGS}")

set(MYHEADS "${PROJECT_SOURCE_DIR}/include")

include_directories(${MYHEADS} ${GLFW_INCLUDE_DIR} ${GLEW_INCLUDE_DIR} ${GL_TOOLS_PATH} ${GLM_PATH})

link_directories(${PROJECT_SOURCE_DIR}/lib)

add_executable(heisenberg ${SOURCES} )

target_link_libraries(heisenberg glfw ${LCG64_LIB} ${GLFW_LIBRARY}
                        ${OPENGL_LIBRARY} ${GLEW_LIBRARIES})

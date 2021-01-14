# HeisenbergModel
Optimization of a Heisenberg Model for GPU using CUDA and CPU using openMP


This is an optimization of the model developed by Orlando Billoni in https://iopscience.iop.org/article/10.1088/0953-8984/28/47/476003/data.

Dependencies
=====
requirements:
- CUDA >= 10.1
- CMAKE >= 3.5
- OpenMP

Depending on OS you might need to get the following graphics libs and their headers:
- OpenGL
- GLFW
- GLEW

*Note:*

CUDA 11 has CUB integrated so this might require you to do a few changes.


To build and compile:
========
- cd to the folder corresponding to the version you want to run (CUDA or OpenMP)
- cmake -S . -B /build
- cd build
- make

If you are compiling CUDA keep in mind the *-arch=compute_75 -code=sm_75* flags in the CMakeLists.txt file and change them if you have a newer or older GPU archetecture.

To Run
====
./heisenbergmodel -L L

Where L is half a size. So N = (2\*L)\*(2\*L)\*(2\*L)

That's the only mandatory parameter. All of the following parameters will take a default values specified in the code. Those parameters are:

- -g for visual representation.
- -o for file outputs
- -t t0 t1 - to assign specific tmax and t_mic, which represent how many updates per Temperature and the measuring interval.
- -f f0 f1 - to specify frac and flucfrac, which assign percentage for the gris distribution.

Also if you are running the CUDA you can specify block sizes with -b BX BY BZ
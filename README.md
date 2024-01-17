# CFFI Embedded Python Example

This project contains an example of using `python cffi` to embedded the Python interpreter into a Fortran program, and call python functons from Fortran.

## Set-up Dev Environment

### MPI
To satisfy the MPI dependency one could install OpenMPI as follows:
- `sudo apt install openmpi-bin libopenmpi-dev`

### Set-up Python dependencies (ensure that the MPI is installed first)
The Python dependenices are listed in `requirements.txt` (this project has been tested with Python 3.10.2). To use `venv` to set-up the dependencies:
- Ensure that you have Python Virtual Environment installed: `sudo apt install python3.10-venv`
- Set-up virtual environment: `python3 -m venv .venv`
- Activiate the virtual environment: `source ./.venv/bin/activate`
- Install Python dependencies: `pip install -r requirements.txt`

## Build

Python CFFI is used to generate a shared library that contains the Python interpreter, and has a API that matches that specification dictated by `py_swe_plugin_module.py` through the use of the decarators `@ffi.def_extern()`. To work, the generation process also requires a C header to be provided that matches the specificed interface. As a result of the process, C source code for the plugin is generated. Therefore, we can either choose to link against that shared library, thus embedding the interpreter dynamically, or build the source code of the library into our app and hence embed the interpreter statically. This choice is expressed through the CMake option `LINK_PYTHON_DYNAMICALLY` (which defaults to true). 

### CMake configure
- `mkdir build`
- `cd build`
- `cmake .. -DLINK_PYTHON_DYNAMICALLY=[true|false]`

### Cmake build
- `cmake --build .`

## Run
- `mpirun -np 4 ./build/src/fort_python`

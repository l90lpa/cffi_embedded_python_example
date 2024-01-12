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

### CMake configure
- `mkdir build`
- `cd build`
- `cmake ../src`

### Cmake build
- `cmake --build .`

## Run
- `mpirun -np 4 ./build/src/fort_python`

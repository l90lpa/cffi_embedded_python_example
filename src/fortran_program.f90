program py_swe
    use mpi
    use, intrinsic :: iso_c_binding
    use fortran_functions_mod
    use fortran_types_mod
  
    implicit none
  
    integer :: ierr, rank, size
    character(len=MPI_MAX_PROCESSOR_NAME) :: processor_name
    integer :: name_len
    type(ParGeometry_f) geom
    type(State_f) state
    integer(c_int) :: nx, ny
    real(c_double) :: xmax, ymax
  
    call MPI_INIT(ierr)
    
    nx = 100
    ny = 100
    xmax = 100000.0
    ymax = 100000.0
    call init_geometry(nx, ny, xmax, ymax, geom)

    call init_tsunami_pulse_initial_condition(geom, state)

    call step_model(geom, state)
  
    call MPI_FINALIZE(ierr)
  
  end program py_swe
  
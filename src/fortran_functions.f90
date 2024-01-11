module fortran_functions_mod
    use, intrinsic :: iso_c_binding
    
    implicit none

    public

    interface
        subroutine init_geometry(nx, ny, xmax, ymax, geom) bind(c, name='init_geometry')
            use iso_c_binding
            use fortran_types_mod, only: ParGeometry_f

            integer(c_int), value, intent(in):: nx, ny
            real(c_double), value, intent(in):: xmax, ymax
            type(ParGeometry_f), intent(inout) :: geom
        end subroutine init_geometry

        subroutine init_tsunami_pulse_initial_condition_impl(geom, nmx, nmy, u, v, h) &
            bind(c, name='init_tsunami_pulse_initial_condition_impl')
            use iso_c_binding
            use fortran_types_mod, only: ParGeometry_f

            type(ParGeometry_f), intent(in) :: geom
            integer(c_int), value, intent(in) :: nmx, nmy
            real(c_double), intent(inout) :: u(nmx, nmy), v(nmx, nmy), h(nmx, nmy)
        end subroutine init_tsunami_pulse_initial_condition_impl

        subroutine step_model_impl(geom, nmx, nmy, u, v, h) &
            bind(c, name='step_model_impl')
            use iso_c_binding
            use fortran_types_mod, only: ParGeometry_f

            type(ParGeometry_f), intent(in) :: geom
            integer(c_int), value, intent(in) :: nmx, nmy
            real(c_double), intent(in) :: u(nmx, nmy), v(nmx, nmy), h(nmx, nmy)
        end subroutine step_model_impl
    end interface

    contains

    subroutine init_tsunami_pulse_initial_condition(geom, state)
        use iso_c_binding
        use fortran_types_mod, only: ParGeometry_f, State_f
        type(ParGeometry_f), intent(in) :: geom
        type(State_f), intent(inout) :: state
        integer :: nmx, nmy

        nmx = (geom%local_domain%halo_depth%west + geom%local_domain%ghost_depth%west + &
               geom%local_domain%grid_extent%x + &
               geom%local_domain%halo_depth%east + geom%local_domain%ghost_depth%east)
    
        nmy = (geom%local_domain%halo_depth%south + geom%local_domain%ghost_depth%south + &
               geom%local_domain%grid_extent%y + &
               geom%local_domain%halo_depth%north + geom%local_domain%ghost_depth%north)

        print *, "nmx=", nmx, "nmy=", nmy

        allocate(state%u(nmx, nmy))
        allocate(state%v(nmx, nmy))
        allocate(state%h(nmx, nmy))

        call init_tsunami_pulse_initial_condition_impl(geom, nmx, nmx, state%u, state%v, state%h)
    end subroutine init_tsunami_pulse_initial_condition

    subroutine step_model(geom, state)
        use iso_c_binding
        use fortran_types_mod, only: ParGeometry_f, State_f
        type(ParGeometry_f), intent(in) :: geom
        type(State_f), intent(in) :: state
        integer :: nmx, nmy

        nmx = (geom%local_domain%halo_depth%west + geom%local_domain%ghost_depth%west + &
               geom%local_domain%grid_extent%x + &
               geom%local_domain%halo_depth%east + geom%local_domain%ghost_depth%east)
    
        nmy = (geom%local_domain%halo_depth%south + geom%local_domain%ghost_depth%south + &
               geom%local_domain%grid_extent%y + &
               geom%local_domain%halo_depth%north + geom%local_domain%ghost_depth%north)

        print *, "nmx=", nmx, "nmy=", nmy

        call step_model_impl(geom, nmx, nmx, state%u, state%v, state%h)
    end subroutine step_model

end module fortran_functions_mod
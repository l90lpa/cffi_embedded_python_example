module fortran_types_mod
    use, intrinsic :: iso_c_binding
    implicit none

    public

    type, bind(c) :: Vec2i_f
        integer(c_int) :: x, y
    end type

    type, bind(c) :: Vec2d_f
        real(c_double) :: x, y
    end type

    type, bind(c) :: ProcessGridGlobalInfo_f
        integer(c_int) :: size, nxprocs, nyprocs
    end type

    type, bind(c) :: ProcessGridLocalTopology_f
        integer(c_int) :: north, south, east, west
    end type

    type, bind(c) :: ProcessGridLocalInfo_f
        integer(c_int) :: rank
        type(ProcessGridLocalTopology_f) :: topology
    end type

    type, bind(c) :: DomainGlobalInfo_f
        type(Vec2d_f) :: origin
        type(Vec2d_f) :: extent
        type(Vec2i_f) :: grid_extent
    end type

    type, bind(c) :: HaloDepth_f
        integer(c_int) :: north, south, east, west
    end type

    type, bind(c) :: GhostDepth_f
        integer(c_int) :: north, south, east, west
    end type

    type, bind(c) :: DomainLocalInfo_f
        type(Vec2i_f) :: grid_origin
        type(Vec2i_f) :: grid_extent
        type(HaloDepth_f) :: halo_depth
        type(GhostDepth_f) :: ghost_depth
    end type

    type, bind(c) :: ParGeometry_f
        type(ProcessGridGlobalInfo_f) :: global_pg
        type(ProcessGridLocalInfo_f)  :: local_pg
        type(DomainGlobalInfo_f)      :: global_domain
        type(DomainLocalInfo_f)       :: local_domain
    end type


    type :: State_f
        real(c_double), allocatable, dimension(:,:) :: u, v, h
    end type

end module fortran_types_mod
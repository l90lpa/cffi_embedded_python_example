from py_swe_plugin import ffi

import mpi4py
# Set initialize to False, to stop mpi4py calling MPI_Init when `MPI` is imported
mpi4py.rc.initialize=False 
from mpi4py import MPI

from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from py_swe.geometry import RectangularGrid, Vec2, ParGeometry, create_domain_par_geometry, add_halo_geometry, add_ghost_geometry
from py_swe.state import State, create_local_field_zeros, create_local_field_tsunami_height

# ======== Geometry =========

def to_py_geometry(geom_ptr):
    from py_swe.geometry import (ProcessGridGlobalInfo, ProcessGridLocalInfo, 
                                 ProcessGridLocalTopology,
                                 DomainGlobalInfo, DomainLocalInfo,
                                 HaloDepth, GhostDepth, Vec2)
                                
    global_pg = ProcessGridGlobalInfo(geom_ptr.global_pg.size, 
                                      geom_ptr.global_pg.nxprocs,
                                      geom_ptr.global_pg.nyprocs)


    local_pg = ProcessGridLocalInfo(geom_ptr.local_pg.rank,
                                    ProcessGridLocalTopology(*[
                                        geom_ptr.local_pg.topology.north,
                                        geom_ptr.local_pg.topology.south,
                                        geom_ptr.local_pg.topology.east,
                                        geom_ptr.local_pg.topology.west,
                                    ]))

    global_domain = DomainGlobalInfo(Vec2(geom_ptr.global_domain.origin.x, geom_ptr.global_domain.origin.y),
                                     Vec2(geom_ptr.global_domain.extent.x, geom_ptr.global_domain.extent.y),
                                     Vec2(geom_ptr.global_domain.grid_extent.x, geom_ptr.global_domain.grid_extent.y))

    local_domain = DomainLocalInfo(Vec2(geom_ptr.local_domain.grid_origin.x, geom_ptr.local_domain.grid_origin.y),
                                   Vec2(geom_ptr.local_domain.grid_extent.x, geom_ptr.local_domain.grid_extent.y),
                                   HaloDepth(
                                        geom_ptr.local_domain.halo_depth.north,
                                        geom_ptr.local_domain.halo_depth.south,
                                        geom_ptr.local_domain.halo_depth.east,
                                        geom_ptr.local_domain.halo_depth.west,
                                   ),
                                   GhostDepth(
                                        geom_ptr.local_domain.ghost_depth.north,
                                        geom_ptr.local_domain.ghost_depth.south,
                                        geom_ptr.local_domain.ghost_depth.east,
                                        geom_ptr.local_domain.ghost_depth.west,
                                   ))
    
    return ParGeometry(global_pg, local_pg, global_domain, local_domain)

def from_py_geometry(py_geom, geom_ptr):
    geom_ptr.global_pg.size         = py_geom.global_pg.size        
    geom_ptr.global_pg.nxprocs      = py_geom.global_pg.nxprocs     
    geom_ptr.global_pg.nyprocs      = py_geom.global_pg.nyprocs     

    geom_ptr.local_pg.rank           = py_geom.local_pg.rank     
    geom_ptr.local_pg.topology.north = py_geom.local_pg.topology.north     
    geom_ptr.local_pg.topology.south = py_geom.local_pg.topology.south     
    geom_ptr.local_pg.topology.east  = py_geom.local_pg.topology.east      
    geom_ptr.local_pg.topology.west  = py_geom.local_pg.topology.west      
    
    geom_ptr.global_domain.origin.x      = py_geom.global_domain.origin.x
    geom_ptr.global_domain.origin.y      = py_geom.global_domain.origin.y
    geom_ptr.global_domain.extent.x      = py_geom.global_domain.extent.x
    geom_ptr.global_domain.extent.y      = py_geom.global_domain.extent.y
    geom_ptr.global_domain.grid_extent.x = py_geom.global_domain.grid_extent.x
    geom_ptr.global_domain.grid_extent.y = py_geom.global_domain.grid_extent.y

    geom_ptr.local_domain.grid_origin.x     = py_geom.local_domain.grid_origin.x
    geom_ptr.local_domain.grid_origin.y     = py_geom.local_domain.grid_origin.y
    geom_ptr.local_domain.grid_extent.x     = py_geom.local_domain.grid_extent.x
    geom_ptr.local_domain.grid_extent.y     = py_geom.local_domain.grid_extent.y
    geom_ptr.local_domain.halo_depth.north  = py_geom.local_domain.halo_depth.north     
    geom_ptr.local_domain.halo_depth.south  = py_geom.local_domain.halo_depth.south     
    geom_ptr.local_domain.halo_depth.east   = py_geom.local_domain.halo_depth.east      
    geom_ptr.local_domain.halo_depth.west   = py_geom.local_domain.halo_depth.west
    geom_ptr.local_domain.ghost_depth.north = py_geom.local_domain.ghost_depth.north     
    geom_ptr.local_domain.ghost_depth.south = py_geom.local_domain.ghost_depth.south     
    geom_ptr.local_domain.ghost_depth.east  = py_geom.local_domain.ghost_depth.east      
    geom_ptr.local_domain.ghost_depth.west  = py_geom.local_domain.ghost_depth.west

@ffi.def_extern()
def init_geometry(nx, ny, xmax, ymax, geom_ptr):
    py_geom = create_geometry(MPI.Comm.py2f(MPI.COMM_WORLD), nx, ny, xmax, ymax)
    from_py_geometry(py_geom, geom_ptr)

def create_geometry(comm_int, nx, ny, xmax, ymax):
    comm = MPI.Comm.f2py(comm_int)
    rank = comm.Get_rank()
    size = comm.Get_size()

    grid = RectangularGrid(nx, ny)
    geometry = create_domain_par_geometry(rank, size, grid, Vec2(0.0, 0.0), Vec2(xmax, ymax))
    geometry = add_ghost_geometry(geometry, 1)
    geometry = add_halo_geometry(geometry, 1)
    return geometry

# ======== Initial Condition =========

def as_ndarray(ptr, shape) -> np.ndarray:
    length = np.prod(shape)
    c_type = ffi.getctype(ffi.typeof(ptr).item)
    array = np.frombuffer(
        ffi.buffer(ptr, length * ffi.sizeof(c_type)),
        dtype=np.dtype(c_type),
        count=-1,
        offset=0,
    ).reshape(shape)
    return array

def copy_to_buffer(ptr, array: np.ndarray):
    length = np.prod(array.shape)
    c_type = ffi.getctype(ffi.typeof(ptr).item)
    ffi.memmove(ptr, np.ravel(array), length * ffi.sizeof(c_type))

@ffi.def_extern()
def init_tsunami_pulse_initial_condition_impl(geom_ptr, nmx, nmy, u_ptr, v_ptr, h_ptr):
    py_geom = to_py_geometry(geom_ptr)
    new_state = create_tsunami_pulse_initial_condition(py_geom)
    u = as_ndarray(u_ptr, (nmx, nmy))
    v = as_ndarray(v_ptr, (nmx, nmy))
    h = as_ndarray(h_ptr, (nmx, nmy))
    u[...] = new_state.u[...]
    v[...] = new_state.v[...]
    h[...] = new_state.h[...]

def create_tsunami_pulse_initial_condition(geometry: ParGeometry):
    zero_field = create_local_field_zeros(geometry, jnp.float64)

    u = jnp.copy(zero_field)
    v = jnp.copy(zero_field)
    h = create_local_field_tsunami_height(geometry, jnp.float64)
    u_ = np.array(u)
    v_ = np.array(v)
    h_ = np.array(h)

    return State(u_, v_, h_)

# ======== Model =========
from math import sqrt, ceil
import time

import matplotlib as mpl
import matplotlib.pyplot as plt

from jax.tree_util import tree_map
import numpy as np

from mpi4jax._src.utils import HashableMPIType

from py_swe.geometry import at_local_domain
from py_swe.state import gather_global_field
from py_swe.model import shallow_water_model_w_padding


def gather_global_state_domain(s, geometry, mpi4py_comm, root):

    s_local_domain = tree_map(lambda x: np.array(x[at_local_domain(geometry)]), s)
    s_global = tree_map(lambda x: gather_global_field(x, geometry.global_pg.nxprocs, geometry.global_pg.nyprocs, root, geometry.local_pg.rank, mpi4py_comm), s_local_domain)
    return s_global


def save_state_figure(state, filename):

    def reorientate(x):
        return np.fliplr(np.rot90(x, k=3))
    
    def downsample(x, n):
        nx = np.size(x, axis=0)
        ns = nx // n
        return x[::ns,::ns]

    # make a color map of fixed colors
    cmap = mpl.colors.LinearSegmentedColormap.from_list('my_colormap', ['blue', 'white', 'red'], 256)

    # modify data layout so that it displays as expected (x horizontal and y vertical, with origin in bottom left corner)
    u = reorientate(state.u)
    v = reorientate(state.v)
    h = reorientate(state.h)

    x = y = np.linspace(0, np.size(u, axis=0)-1, np.size(u, axis=0))
    xx, yy = np.meshgrid(x, y)

    # downsample velocity vector field to make it easier to read
    xx = downsample(xx, 20)
    yy = downsample(yy, 20)
    u = downsample(u, 20)
    v = downsample(v, 20)

    fig, ax = plt.subplots()
    # tell imshow about color map so that only set colors are used
    img = ax.imshow(h, interpolation='nearest', cmap=cmap, origin='lower')
    ax.quiver(xx,yy,u,v)
    plt.colorbar(img,cmap=cmap)
    plt.grid(True,color='black')
    plt.savefig(filename)


def save_global_state_domain_on_root(s, geometry: ParGeometry, mpi4py_comm, root, filename, msg):
    s_global = gather_global_state_domain(s, geometry, mpi4py_comm, root)
    if geometry.local_pg.rank == root:
        save_state_figure(s_global, filename)
        print(msg)

@ffi.def_extern()
def step_model_impl(geom_ptr, nmx, nmy, u_ptr, v_ptr, h_ptr):
    u = as_ndarray(u_ptr, (nmx, nmy))
    v = as_ndarray(v_ptr, (nmx, nmy))
    h = as_ndarray(h_ptr, (nmx, nmy))
    py_geom = to_py_geometry(geom_ptr)
    py_s0 = State(u, v, h)
    comm_int = MPI.Comm.py2f(MPI.COMM_WORLD)
    root = 0
    step_model(py_geom, py_s0, comm_int, root)

def step_model(geometry: ParGeometry, s0: State, comm_int, root):
    assert geometry.global_domain.extent.x == geometry.global_domain.extent.y
    assert geometry.global_domain.grid_extent.x == geometry.global_domain.grid_extent.y

    mpi4jax_comm = MPI.Comm.f2py(comm_int)
    mpi4jax_comm_wrapped = HashableMPIType(mpi4jax_comm)
    mpi4py_comm = mpi4jax_comm.Clone()

    rank = mpi4jax_comm.Get_rank()
    
    xmax = geometry.global_domain.extent.x
    nx = geometry.global_domain.grid_extent.x

    dx = dy = xmax / (nx - 1.0)
    g = 9.81
    dt = 0.68 * dx / sqrt(g * 5030)
    tmax = 150
    num_steps = ceil(tmax / dt)

    token = jnp.empty((1,))
    b = create_local_field_zeros(geometry, jnp.float64)

    save_global_state_domain_on_root(s0, geometry, mpi4py_comm, root, "step-0.png", "Saved initial condition.")


    if rank == root:
        print(f"Starting compilation.")
        start = time.perf_counter()


    model_compiled = shallow_water_model_w_padding.lower(s0, geometry, mpi4jax_comm_wrapped, b, num_steps, dt, dx, dy, token).compile()


    if rank == root:
        end = time.perf_counter()
        print(f"Compilation completed in {end - start} seconds.")
        print(f"Starting simulation with {num_steps} steps...")
        start = time.perf_counter()


    sN, _ = model_compiled(s0, b, dt, dx, dy, token)
    sN.u.block_until_ready()


    if rank == root:
        end = time.perf_counter()
        print(f"Simulation completed in {end - start} seconds, with an average time per step of {(end - start) / num_steps} seconds.")


    save_global_state_domain_on_root(sN, geometry, mpi4py_comm, root, f"step-{num_steps}.png", "Saved final condition.")
    


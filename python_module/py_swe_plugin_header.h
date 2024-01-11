struct Vec2i_c {
    int x, y;
};
struct Vec2d_c {
    double x, y;
};
struct ParGeometry_c {
    struct ProcessGridGlobalInfo_c {
        int size;
        int nxprocs;
        int nyprocs;
    } global_pg;
    struct ProcessGridLocalInfo_c {
        int rank;
        struct ProcessGridLocalTopology_c {
            int north, south, east, west;
        } topology;
    } local_pg;
    struct DomainGlobalInfo_c {
        struct Vec2d_c origin;
        struct Vec2d_c extent;
        struct Vec2i_c grid_extent;
    } global_domain;
    struct DomainLocalInfo_c {
        struct Vec2i_c grid_origin;
        struct Vec2i_c grid_extent;
        struct HaloDepth_c {
            int north, south, east, west;
        } halo_depth;
        struct GhostDepth_c {
            int north, south, east, west;
        } ghost_depth;
    } local_domain;
};

extern void init_geometry(int, int, double, double, struct ParGeometry_c*);

extern void init_tsunami_pulse_initial_condition_impl(struct ParGeometry_c*, int, int, double*, double*, double*);

extern void step_model_impl(struct ParGeometry_c*, int, int, double*, double*, double*);
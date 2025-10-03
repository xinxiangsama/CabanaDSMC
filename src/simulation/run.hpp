#pragma once
#include "particle.hpp"
#include "mesh/mesh.hpp"
#include "cell/cell.hpp"
#include "cell/sample.hpp"
#include "initialize.hpp"
#include "boundary/boundary.hpp"
#include "move_old_version.hpp"
#include "sort/sort.hpp"
#include "collide/collide.hpp"
#include "output/output.hpp"
#include <chrono>
#include <Cabana_Grid.hpp>
#include "../input/input.hpp"
#include "../geometry/distinguish_node.hpp"
#include "../geometry/cut.hpp"
namespace CabanaDSMC{

template<class ExecutionSpace = Kokkos::DefaultExecutionSpace,
            typename MemorySpace = typename ExecutionSpace::memory_space>
class Run{
public:
    Run() = default;
    using memory_space = MemorySpace;
    using exec_space = ExecutionSpace;
    using particle_list_type = UserSpecfic::particle_list_type;
    using particle_type = typename particle_list_type::particle_type;
    using species_list_type = Particle::SpeciesList<memory_space>;
    using mesh_manager_type = Mesh::MeshManager<memory_space>;
    using mesh_type = typename mesh_manager_type::mesh_type;
    using scalar_type = typename mesh_manager_type::scalar_type;
    using cell_data_type = Cell::CellData<scalar_type, memory_space, mesh_type>;
    using boundary_container_type = UserSpecfic::boundaryContainer_t;
    using collision_model = UserSpecfic::collisionModel;
    using node_data_type = UserSpecfic::node_data_type;
    using stl_type = UserSpecfic::stl_type;
    using sampling_t = UserSpecfic::sampling;
    using write_type = UserSpecfic::write_type;
    using writer_type = UserSpecfic::writer_type;

    static constexpr int dim = UserSpecfic::dim;

    void init(const Input::SimulationConfig& config);
    void run();
protected:
    particle_list_type particles {"particles"};
    species_list_type species;
    std::shared_ptr<mesh_manager_type> mesh_manager;
    std::shared_ptr<cell_data_type> cell_data;
    std::shared_ptr<node_data_type> node_data;
    stl_type stl;
    // static boundary
    UserSpecfic::xminBoundary_t xmin_boundary;
    UserSpecfic::xmaxBoundary_t xmax_boundary;
    UserSpecfic::yminBoundary_t ymin_boundary;
    UserSpecfic::ymaxBoundary_t ymax_boundary;
    UserSpecfic::zminBoundary_t zmin_boundary;
    UserSpecfic::zmaxBoundary_t zmax_boundary;

    int rank;
    int num_procs;

    size_t steps;
    uint8_t seed;
    uint8_t step_average;
    uint8_t step_write;
};


//========== initialize ============================

template <class ExecutionSpace, typename MemorySpace>
inline void Run<ExecutionSpace, MemorySpace>::init(const Input::SimulationConfig& config)
{
    //-----------------------------------------
    // global
    //------------------------------------------
    steps = config.steps;
    seed = config.seed;
    step_average = config.step_average;
    step_write = config.step_write;
    //-----------------------------------------
    // mesh
    //------------------------------------------
    auto mesh_config = config.mesh_config;
    mesh_manager = std::make_shared<mesh_manager_type>(mesh_config);
    auto local_grid = mesh_manager->getLocalGrid();
    auto global_mesh = mesh_manager->getGlobalMesh();
    //-----------------------------------------
    // stl
    //------------------------------------------
    auto h_stl = config.h_stl;
    stl =Kokkos::create_mirror_view_and_copy(exec_space{}, h_stl);
    //-----------------------------------------
    // species list
    //------------------------------------------
    auto h_species  = config.species_list;
    species = Kokkos::create_mirror_view_and_copy(exec_space {}, h_species);

    //-----------------------------------------
    // cell data
    //------------------------------------------
    cell_data = std::make_shared<cell_data_type>(local_grid);
    auto field_initial = config.initial;
    initializeField(
        exec_space {},
        field_initial,
        cell_data,
        species
        );
    //-----------------------------------------
    // node data
    //------------------------------------------
    node_data = std::make_shared<node_data_type>(local_grid);
    // node_data->setAllNodesToOutside();
    // Geometry::distinguish_node(exec_space {}, stl, node_data);
    // Kokkos::fence("");
    // Cabana::Grid::Experimental::BovWriter::writeTimeStep(
    //     ExecutionSpace {},
    //     "",
    //     1,
    //     1,
    //     *node_data->is_inside);
    //
    //-----------------------------------------
    // cut cell
    //------------------------------------------
    Geometry::cutcell(
        exec_space {},
        stl,
        cell_data
    );
    Kokkos::fence("");
    Cabana::Grid::Experimental::BovWriter::writeTimeStep(
        ExecutionSpace {},
        "cut_cell_num",
        1,
        1,
        *cell_data->Num_cut_faces);
    //-----------------------------------------
    // particle list
    //------------------------------------------
    createParticles(
        Cabana::InitRandom {},
        exec_space {},
        UserSpecfic::particle_factory_t {},
        particles,
        local_grid,
        cell_data,
        species,
        0,
        456789
        );
    //-----------------------------------------
    // boundary
    //------------------------------------------
    auto boundary_config = config.boundary_config;
    xmin_boundary = Boundary::BoundaryFactory<UserSpecfic::xminBoundary_t, mesh_type>::create(
        boundary_config[0], global_mesh);
    xmax_boundary = Boundary::BoundaryFactory<UserSpecfic::xmaxBoundary_t, mesh_type>::create(
        boundary_config[1], global_mesh);
    ymin_boundary = Boundary::BoundaryFactory<UserSpecfic::yminBoundary_t, mesh_type>::create(
        boundary_config[2], global_mesh);
    ymax_boundary = Boundary::BoundaryFactory<UserSpecfic::ymaxBoundary_t, mesh_type>::create(
        boundary_config[3], global_mesh);
    zmin_boundary = Boundary::BoundaryFactory<UserSpecfic::zminBoundary_t, mesh_type>::create(
        boundary_config[4], global_mesh);
    zmax_boundary = Boundary::BoundaryFactory<UserSpecfic::zmaxBoundary_t, mesh_type>::create(
        boundary_config[5], global_mesh);

}

template <class ExecutionSpace, typename MemorySpace>
inline void Run<ExecutionSpace, MemorySpace>::run()
{
    auto local_grid = mesh_manager->getLocalGrid();
    auto global_grid = mesh_manager->getGlobalGrid();
    auto owned_cells = local_grid->indexSpace( Cabana::Grid::Own(), Cabana::Grid::Cell(), Cabana::Grid::Local() );
    const auto local_seed = global_grid->blockId() + ( seed % ( global_grid->blockId() + 1 ) );
    Kokkos::Random_XorShift64_Pool<ExecutionSpace> pool( local_seed, owned_cells.size() );
    for (size_t i = 0; i < steps; ++i){
        moveParticles(
            exec_space {},
            particles,
            local_grid,
            cell_data,
            stl,
            xmin_boundary,
            xmax_boundary,
            ymin_boundary,
            ymax_boundary,
            zmin_boundary,
            zmax_boundary
            );

        Kokkos::fence("move");

        sort(
            exec_space {},
            particles,
            cell_data
            );
        Kokkos::fence("sort");
        collide(
            exec_space {},
            particles,
            local_grid,
            cell_data,
            species,
            collision_model  {},
            pool
            );
        Kokkos::fence("collide");

        Cell::sample(
            i,
            step_average,
            particles,
            cell_data,
            species,
            sampling_t {}
            );
        Kokkos::fence("sample");

        if (i % step_write == 0) {
            write(
                write_type {},
                "",
                i,
                i * 1.0e-5,
                std::make_shared<writer_type>(),
                cell_data
                );
        }
    }
}
}

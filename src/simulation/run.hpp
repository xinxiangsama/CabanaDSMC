#pragma once
#include "particle.hpp"
#include "mesh/mesh.hpp"
#include "cell/cell.hpp"
#include "cell/sample.hpp"
#include "initialize.hpp"
#include "boundary/boundary.hpp"
#include "move.hpp"
#include "sort/sort.hpp"
#include "collide/collide.hpp"
#include "output/output.hpp"
#include <chrono>
#include <Cabana_Grid.hpp>
namespace CabanaDSMC{

template<class ExecutionSpace = Kokkos::DefaultExecutionSpace,
            typename MemorySpace = typename ExecutionSpace::memory_space>
class Run{
public:
    Run() = default;
    using memory_space = MemorySpace;
    using exec_space = ExecutionSpace;
    using particle_list_type = Particle::GridParticleList<memory_space, 16>;
    using particle_type = typename particle_list_type::particle_type;
    using species_list_type = Particle::SpeciesList<memory_space>;
    using mesh_manager_type = Mesh::MeshManager<memory_space>;
    using mesh_type = typename mesh_manager_type::mesh_type;
    using scalar_type = typename mesh_manager_type::scalar_type;
    using cell_data_type = Cell::CellData<scalar_type, memory_space, mesh_type>;

    void init();
    void run();
protected:
    particle_list_type particles {"particles"};
    species_list_type species;
    std::shared_ptr<mesh_manager_type> mesh_manager;
    std::shared_ptr<cell_data_type> cell_data;

    int rank;
    int numprocs;
};


//========== initialize ============================

template <class ExecutionSpace, typename MemorySpace>
inline void Run<ExecutionSpace, MemorySpace>::init()
{

}

template <class ExecutionSpace, typename MemorySpace>
inline void Run<ExecutionSpace, MemorySpace>::run()
{
}
}

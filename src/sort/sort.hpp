#pragma once
#include "../cell/cell.hpp"
#include <Cabana_Core.hpp>
namespace CabanaDSMC{
template<class ExecutionSpace, class ParticleListType, class CellDataType>
requires (Cabana::is_particle_list<ParticleListType>::value ||
            Cabana::Grid::is_particle_list<ParticleListType>::value)

void sort(
    const ExecutionSpace& exec_space,
    ParticleListType& particle_list,
    const std::shared_ptr<CellDataType>& cell_data 
)
{       
    using memory_space = ExecutionSpace::memory_space;
    using mesh_type = typename CellDataType::mesh_type;

    // get particle position value, it will be the key to sort
    auto position = particle_list.slice(Particle::Field::Position {});

    // get local mesh
    auto local_grid = cell_data->localgrid();
    auto local_mesh = Cabana::Grid::createLocalMesh<memory_space, mesh_type>(*local_grid);

    //create linked_cell_list based on the local gird
    double grid_min[3], grid_max[3], grid_delta[3];
    for (int d = 0; d < 3; ++d) {
        grid_min[d] = local_mesh.lowCorner(Cabana::Grid::Ghost(), d);
        grid_max[d] = local_mesh.highCorner(Cabana::Grid::Ghost(), d);
        grid_delta[d]  = local_grid->globalGrid().globalMesh().cellSize(d);
    }
    // set the particle position as the key to sort
    auto linked_cell_list = Cabana::createLinkedCellList<memory_space>(
        position, grid_delta, grid_min, grid_max);
    // based the sort res to permute particle (i don't known whether it is a expensive process---2025 9 27)
    Cabana::permute(linked_cell_list, particle_list.aosoa());
    Kokkos::fence();

    // reset cell data
    Cabana::Grid::ArrayOp::assign(
        *cell_data->Num_particles,
        0,
        Cabana::Grid::Own()
    );
    Cabana::Grid::ArrayOp::assign(
        *cell_data->Offset_particle_idx,
        0,
        Cabana::Grid::Own()
    );
    // fill in cell data
    auto particle_num_view = cell_data->Num_particles->view();
    auto offset_index_view = cell_data->Offset_particle_idx->view();
    auto owned_cells_is = local_grid->indexSpace(
                Cabana::Grid::Own(), Cabana::Grid::Cell(), Cabana::Grid::Local());
 
    Cabana::Grid::grid_parallel_for(
        "sort_particles_into_cells",
        ExecutionSpace{},
        owned_cells_is,
        KOKKOS_LAMBDA(const int i, const int j, const int k) {
            particle_num_view(i, j, k, 0) = linked_cell_list.binSize(i, j, k);
            offset_index_view(i, j, k, 0) = linked_cell_list.binOffset(i, j, k);
        });
    
    // set particle cell id
    auto cell_id_slice = particle_list.slice(Particle::Field::CellID{});
    std::size_t num_particles = particle_list.size();
    
    // Update cell IDs for all particles
    Kokkos::parallel_for(
        "update_particle_cell_ids",
        Kokkos::RangePolicy<ExecutionSpace>(0, particle_list.size()),
        KOKKOS_LAMBDA(const std::size_t p) {
            // Get the bin index for this particle n
            int bin_index = linked_cell_list.getParticleBin(p);
            
            // Convert the 1D bin index to 3D grid coordinates
            int i, j, k;
            linked_cell_list.ijkBinIndex(bin_index, i, j, k);
            
            // Set the cell ID for this particle
            cell_id_slice(p, 0) = static_cast<uint32_t>(i);
            cell_id_slice(p, 1) = static_cast<uint32_t>(j);
            cell_id_slice(p, 2) = static_cast<uint32_t>(k);
        });
    Kokkos::fence();
}
}
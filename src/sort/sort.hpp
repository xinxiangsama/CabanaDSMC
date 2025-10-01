#pragma once
#include "../cell/cell.hpp"
#include <Cabana_Core.hpp>
#include <mpi.h>
#include <iostream>
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
    
    // 获取MPI信息
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    
    // 记录排序前的粒子总数（本地和全局）
    std::size_t local_particles_before = particle_list.size();
    std::size_t global_particles_before = 0;
    MPI_Allreduce(&local_particles_before, &global_particles_before, 1, MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);
    
    if (rank == 0) {
        std::cout << "=== Before Sort ===" << std::endl;
        std::cout << "Global particle count: " << global_particles_before << std::endl;
    }
    
    // 输出每个进程的粒子数
    std::cout << "Rank " << rank << " particles before sort: " << local_particles_before << std::endl;

    particle_list.redistribute(
        *(cell_data->localgrid()),
        Particle::Field::Position {}
    );
    Kokkos::fence();
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
    auto linked_cell_list = Cabana::createLinkedCellList(
        position, grid_delta, grid_min, grid_max);
    // based the sort res to permute particle (i don't known whether it is a expensive process---2025 9 27)
    Cabana::permute(linked_cell_list.binningData(), particle_list.aosoa());
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
    Kokkos::fence();
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
    
    // 记录排序后的粒子总数（本地和全局）
    std::size_t local_particles_after = particle_list.size();
    std::size_t global_particles_after = 0;
    MPI_Allreduce(&local_particles_after, &global_particles_after, 1, MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);
    
    // 输出每个进程的粒子数
    std::cout << "Rank " << rank << " particles after sort: " << local_particles_after << std::endl;
    
    if (rank == 0) {
        std::cout << "=== After Sort ===" << std::endl;
        std::cout << "Global particle count: " << global_particles_after << std::endl;
        
        // 检测粒子总数变化
        if (global_particles_after < global_particles_before) {
            std::cout << "WARNING: Global particle count decreased! "
                      << "Before: " << global_particles_before 
                      << ", After: " << global_particles_after 
                      << ", Lost: " << (global_particles_before - global_particles_after) 
                      << std::endl;
        } else if (global_particles_after > global_particles_before) {
            std::cout << "WARNING: Global particle count increased! "
                      << "Before: " << global_particles_before 
                      << ", After: " << global_particles_after 
                      << ", Gained: " << (global_particles_after - global_particles_before) 
                      << std::endl;
        } else {
            std::cout << "Global particle count unchanged." << std::endl;
        }
        std::cout << "=================================" << std::endl;
    }
}
}

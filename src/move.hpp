#pragma once
#include "particle.hpp"
namespace CabanaDSMC{
template <class ExecutionSpace, class ParticleListType, class LocalGridType,
          class CellDataType, class ... BoundaryTypes>
requires (Cabana::is_particle_list<ParticleListType>::value ||
            Cabana::Grid::is_particle_list<ParticleListType>::value)
void moveParticles(
    const ExecutionSpace& exec_space, ParticleListType& particle_list,
    const std::shared_ptr<LocalGridType>& local_grid, 
    const std::shared_ptr<CellDataType>& cell_data,
    const BoundaryTypes&... boundaries
)
{   
    // using memory_space = typename ParticleListType::memory_space;
    // using boundary_tuple_type = std::tuple<BoundaryTypes...>;
    // if constexpr (sizeof...(BoundaryTypes) > 0){
    //     boundary_conditions = std::make_tuple(boundaries...);
    // }
    auto boundary_conditions = std::make_tuple(boundaries...);

    //get global grid
    // const auto& global_grid = local_grid->globalGrid();

    //get local set of owned cell indices
    // auto owned_cells = local_grid->indexSpace( Cabana::Grid::Own(), Cabana::Grid::Cell(), Cabana::Grid::Local() );

    // get particle data view
    auto position = particle_list.slice(Particle::Field::Position {});
    auto velocity = particle_list.slice(Particle::Field::Velocity {});
    auto cell_id = particle_list.slice(Particle::Field::CellID {});

    // get cell data view
    // to be thought: what we need from cell data 
    // 1. cell time step (even get from ghost cell)
    // 2. cell cut cell faces
    auto cell_dt = cell_data->Dt->view();
    // auto cut_cell_faces = cell_data->Cut_cell_faces->view();

    // move particles
    Kokkos::parallel_for("move particles",
        Kokkos::RangePolicy<ExecutionSpace>(ExecutionSpace {}, 0, particle_list.size()),
        KOKKOS_LAMBDA(const int idx){
            auto dt = cell_dt(cell_id(idx,0), cell_id(idx,1), cell_id(idx,2), 0);
            
            // move particle
            for(int d = 0; d < 3; ++d){
                position(idx, d) += velocity(idx, d) * dt;
            }

            // apply boundary conditions
            auto particle = particle_list.getParticle(idx);
            std::apply([&](const auto&... boundary){
                (boundary.apply(particle), ...);
            }, boundary_conditions);
            particle_list.setParticle(particle, idx);
        }
    );

    // to be implemented:
    // 1. check boundary condition
    // 2. variable time step
    // 3. hit with cut cell faces        
}

}
